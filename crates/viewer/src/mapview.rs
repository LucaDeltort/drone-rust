//! Online map in the viewer: measurement integration, 3D overlay, egui minimap.

use bevy::{
    asset::RenderAssetUsages,
    light::{NotShadowCaster, NotShadowReceiver},
    camera::visibility::RenderLayers,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_egui::egui;
use mapping::{Agreement, Cell, Counts, OccupancyGrid};

use crate::{
    CurrentWorld, WorldRebuilt,
    drone::{DRONE_LAYER, SimDrone},
    pose::{Estimator, estimate},
    sensing::Suite,
};

/// Map cell size, in m.
const CELL_M: f32 = 0.25;
/// Half-thickness of the mapped slab, in m.
const HALF_SLAB: f32 = 0.5;
/// Refresh rates of the display and of the consistency measurement.
const REDRAW_S: f32 = 0.2;
const AGREEMENT_S: f32 = 2.0;

#[derive(Resource)]
pub struct Map {
    pub grid: OccupancyGrid,
    pub image: Handle<Image>,
    /// Time of the last integrated measurement, per sensor.
    last_t: Vec<f64>,
    dirty: bool,
    counts: Counts,
    agreement: Option<Agreement>,
    /// The grid is projected from the 3D map (3D flight): measurements are not
    /// integrated into it directly.
    pub projected: bool,
}

#[derive(Resource)]
pub struct MapSettings {
    overlay: bool,
    /// Altitude of the slab above the ground at the origin (starting point), in m.
    altitude: f32,
}

#[derive(Component)]
pub struct MapOverlay;

pub struct MapPlugin;

impl Map {
    /// Clears the map (pose mode change).
    pub fn reset(&mut self) {
        self.grid.clear();
        self.last_t.clear();
        self.dirty = true;
        self.agreement = None;
        // A projected grid must be projected again in full.
        self.projected = false;
    }

    /// The grid changed outside of `integrate`: redraw it.
    pub fn touch(&mut self) {
        self.dirty = true;
    }

    /// Cell counts at the last refresh.
    pub fn counts(&self) -> Counts {
        self.counts
    }
}

impl Plugin for MapPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MapSettings { overlay: true, altitude: 2.0 })
            .add_systems(Startup, create_map)
            .add_systems(FixedUpdate, integrate.after(estimate))
            .add_systems(Update, (reset_on_rebuild, redraw, measure_agreement).chain())
            ;
    }
}

fn new_grid(world: &world::World, settings: &MapSettings) -> OccupancyGrid {
    let y = world.terrain.height_at(0.0, 0.0) + settings.altitude;
    OccupancyGrid::new(world.terrain.half_size() * 2.0, CELL_M, y - HALF_SLAB, y + HALF_SLAB)
}

fn blank_image(grid: &OccupancyGrid) -> Image {
    let (nx, nz) = grid.size();
    let mut image = Image::new_fill(
        Extent3d { width: nx as u32, height: nz as u32, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    // Crisp pixels: one cell = one square.
    image.sampler = bevy::image::ImageSampler::nearest();
    image
}

/// Horizontal quad covering the map; pixel (i, j) falls on cell (i, j).
fn overlay_mesh(grid: &OccupancyGrid) -> Mesh {
    let (nx, nz) = grid.size();
    let (x0, z0) = (grid.center(0, 0) - grid.cell_m() / 2.0).into();
    let (x1, z1) = (grid.center(nx - 1, nz - 1) + grid.cell_m() / 2.0).into();
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]],
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 4])
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        .with_inserted_indices(Indices::U32(vec![0, 2, 1, 0, 3, 2]))
}

fn spawn_overlay(
    commands: &mut Commands,
    grid: &OccupancyGrid,
    image: &Handle<Image>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(overlay_mesh(grid))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(image.clone()),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_xyz(0.0, (grid.y_lo + grid.y_hi) / 2.0, 0.0),
        // Outside view only: the onboard camera does not see the map.
        RenderLayers::layer(DRONE_LAYER),
        // Transparent overlay: it must not darken the ground.
        NotShadowCaster,
        NotShadowReceiver,
        MapOverlay,
    ));
}

fn create_map(
    mut commands: Commands,
    world: Res<CurrentWorld>,
    settings: Res<MapSettings>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let grid = new_grid(&world.0, &settings);
    let image = images.add(blank_image(&grid));
    spawn_overlay(&mut commands, &grid, &image, &mut meshes, &mut materials);
    commands.insert_resource(Map {
        counts: grid.counts(),
        grid,
        image,
        last_t: Vec::new(),
        dirty: true,
        agreement: None,
        projected: false,
    });
}

/// New empty map for the current world (world or altitude change).
#[allow(clippy::too_many_arguments)]
fn rebuild_map(
    map: &mut Map,
    world: &world::World,
    settings: &MapSettings,
    commands: &mut Commands,
    overlays: &Query<Entity, With<MapOverlay>>,
    images: &mut Assets<Image>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    map.grid = new_grid(world, settings);
    map.projected = false;
    let _ = images.insert(&map.image, blank_image(&map.grid));
    for e in overlays {
        commands.entity(e).despawn();
    }
    spawn_overlay(commands, &map.grid, &map.image, meshes, materials);
    map.last_t.clear();
    map.counts = map.grid.counts();
    map.agreement = None;
    map.dirty = true;
}

#[allow(clippy::too_many_arguments)]
fn reset_on_rebuild(
    mut rebuilt: MessageReader<WorldRebuilt>,
    mut last_altitude: Local<Option<f32>>,
    mut map: ResMut<Map>,
    world: Res<CurrentWorld>,
    settings: Res<MapSettings>,
    mut commands: Commands,
    overlays: Query<Entity, With<MapOverlay>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let altitude_changed = last_altitude.replace(settings.altitude).is_some_and(|a| a != settings.altitude);
    if rebuilt.read().count() == 0 && !altitude_changed {
        return;
    }
    rebuild_map(&mut map, &world.0, &settings, &mut commands, &overlays, &mut images, &mut meshes, &mut materials);
}

/// Integrates each new measurement, once, from the pose the drone believes it
/// has (the measurement was taken from the real one).
fn integrate(
    mut map: ResMut<Map>,
    est: Res<Estimator>,
    config: Res<crate::drone::DroneConfig>,
    suites: Query<(&SimDrone, &Suite)>,
) {
    let map = map.as_mut();
    for (d, suite) in &suites {
        let truth = sensors::Pose::from(&d.state);
        let believed = sensors::Pose::from(&est.belief);
        // Where the drone's body is, there is no obstacle; on the first step of a
        // new map, the take-off area is assumed clear.
        let at = Vec2::new(believed.position.x, believed.position.z);
        let radius = if map.last_t.is_empty() { sim::TAKEOFF_CLEAR_M } else { config.0.airframe.radius };
        map.grid.mark_free_disk(at, radius);
        map.dirty = true;
        map.last_t.resize(suite.0.latest.len(), f64::NEG_INFINITY);
        for (k, m) in suite.0.latest.iter().enumerate() {
            if let Some(m) = m
                && m.t > map.last_t[k]
            {
                if map.projected {
                    // 3D flight: the 3D map fills the grid.
                } else if est.ekf.is_some() {
                    map.grid.integrate(&m.reposed(&truth, &believed));
                } else {
                    map.grid.integrate(m);
                }
                map.last_t[k] = m.t;
                map.dirty = true;
            }
        }
    }
}

fn redraw(time: Res<Time>, mut timer: Local<f32>, mut map: ResMut<Map>, mut images: ResMut<Assets<Image>>) {
    *timer += time.delta_secs();
    if !map.dirty || *timer < REDRAW_S {
        return;
    }
    *timer = 0.0;
    map.dirty = false;
    map.counts = map.grid.counts();
    let Some(mut image) = images.get_mut(&map.image) else { return };
    let Some(data) = image.data.as_mut() else { return };
    let (nx, nz) = map.grid.size();
    for j in 0..nz {
        for i in 0..nx {
            let px: [u8; 4] = match map.grid.cell(i, j) {
                Cell::Unknown => [0, 0, 0, 0],
                Cell::Free => [90, 170, 255, 40],
                Cell::Occupied => [235, 45, 35, 235],
            };
            let k = 4 * (j * nx + i);
            data[k..k + 4].copy_from_slice(&px);
        }
    }
}

fn measure_agreement(time: Res<Time>, mut timer: Local<f32>, mut map: ResMut<Map>, world: Res<CurrentWorld>) {
    *timer += time.delta_secs();
    if *timer < AGREEMENT_S {
        return;
    }
    *timer = 0.0;
    let (lo, hi) = (map.grid.y_lo, map.grid.y_hi);
    let a = map.grid.agreement(|x, z| {
        let c = world.0.column(x, z, lo, hi);
        (c.matter, c.space)
    });
    map.agreement = Some(a);
}

/// Map section of the side panel.
pub fn panel(
    ui: &mut egui::Ui,
    map: &mut Map,
    settings: &mut MapSettings,
    overlays: &mut Query<&mut Visibility, With<MapOverlay>>,
    drone: Option<&SimDrone>,
    exploration: &crate::explore::Exploration,
    texture: egui::TextureId,
) {

            let mut overlay = settings.overlay;
            if ui.checkbox(&mut overlay, "Show the map in the scene").changed() {
                settings.overlay = overlay;
                for mut v in overlays.iter_mut() {
                    *v = if overlay { Visibility::Inherited } else { Visibility::Hidden };
                }
            }
            // Changing the altitude restarts from an empty map.
            let mut alt = settings.altitude;
            let slider = egui::Slider::new(&mut alt, 0.5..=10.0).step_by(0.5).text("slab altitude (m)");
            if ui.add(slider).on_hover_text("above the ground at the starting point").changed() {
                settings.altitude = alt;
            }
            if ui.button("Clear the map").clicked() {
                map.grid.clear();
                map.dirty = true;
                map.agreement = None;
            }

            let (lo, hi) = (map.grid.y_lo, map.grid.y_hi);
            if map.projected {
                ui.label("3D flight: obstacles too tall to fly over (from the 3D map)");
            } else if let Some(d) = drone {
                let y = d.state.position.y;
                if (lo..=hi).contains(&y) {
                    ui.label(format!("Drone in the slab ({lo:.1} to {hi:.1} m)"));
                } else {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 160, 60),
                        format!("Drone outside the slab ({lo:.1} to {hi:.1} m): the 2D lidar sees nothing there"),
                    );
                }
            }
            let c = map.counts;
            let area = |n: usize| n as f32 * map.grid.cell_m().powi(2);
            ui.label(format!(
                "Known: {:.0} m² free, {:.0} m² occupied",
                area(c.free),
                area(c.occupied)
            ));
            match map.agreement {
                Some(a) if a.free + a.occupied > 0 => {
                    ui.label(format!(
                        "Consistency with the terrain: occupied {:.1} %, free {:.2} %",
                        100.0 * a.occupied_ratio(),
                        100.0 * a.free_ratio()
                    ));
                }
                _ => {
                    ui.label("Consistency with the terrain: pending");
                }
            }

            // Top-down minimap, north (-Z) up, with the drone's position.
            let side = 260.0;
            let response = ui.image((texture, egui::vec2(side, side)));
            let rect = response.rect;
            ui.painter().rect_stroke(rect, 0.0, (1.0, egui::Color32::GRAY), egui::StrokeKind::Outside);
            if let Some(d) = drone {
                let (nx, _) = map.grid.size();
                let extent = nx as f32 * map.grid.cell_m();
                let to_ui = |p: Vec3| {
                    rect.min + egui::vec2((p.x / extent + 0.5) * side, (p.z / extent + 0.5) * side)
                };
                // Explorer plan: frontiers, path, target.
                if let Some(e) = &exploration.explorer {
                    let flat = |p: Vec2| to_ui(Vec3::new(p.x, 0.0, p.y));
                    for &f in &e.frontiers {
                        ui.painter().circle_filled(flat(f), 2.5, egui::Color32::from_rgb(255, 230, 30));
                    }
                    let pts: Vec<egui::Pos2> = e.path.iter().map(|&p| flat(p)).collect();
                    ui.painter().add(egui::Shape::line(pts, (1.5, egui::Color32::WHITE)));
                    if let Some(t) = e.target {
                        ui.painter().circle_stroke(flat(t), 4.0, (2.0, egui::Color32::from_rgb(255, 60, 230)));
                    }
                }
                let pos = to_ui(d.state.position);
                let nose = to_ui(d.state.position + d.state.forward() * extent * 0.03);
                ui.painter().circle_filled(pos, 4.0, egui::Color32::from_rgb(255, 140, 0));
                ui.painter().line_segment([pos, nose], (2.0, egui::Color32::from_rgb(255, 140, 0)));
            }
}
