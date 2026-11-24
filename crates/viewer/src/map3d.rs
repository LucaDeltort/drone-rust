//! 3D map in the viewer: every range measurement feeds a voxel map, drawn as
//! cubes colored by height. `M` shows or hides it, `V` hides the real world to
//! see only what the drone has mapped.
//!
//! 3D flight (on by default for a drone with a 3D lidar): the missions plan on
//! the 3D map projected flat (only what is too tall to fly over blocks), and
//! [`planning::Flight3d`] sets the altitude from the surfaces ahead and holds the
//! drone back from what stands at its height.

use rustc_hash::FxHashMap as HashMap;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::RenderLayers,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bevy_egui::egui;
use drone_spec::SensorKind;
use mapping::{CHUNK, VoxelMap};
use planning::{Flight3d, Flight3dConfig};

use crate::{
    OuterCamera, WorldRebuilt,
    drone::{DRONE_LAYER, DroneConfig, PilotCommand, SimDrone, step_drone},
    explore::{Exploration, explore},
    mapview::Map,
    mission::{Mission, navigate},
    pose::{Estimator, estimate},
    sensing::Suite,
};

/// Voxel side, in m.
const VOXEL_M: f32 = 0.4;
/// Mesh refresh period, in s. Only the chunks that changed are remeshed.
const REDRAW_S: f32 = 0.2;

#[derive(Resource)]
pub struct Map3d {
    pub map: VoxelMap,
    last_t: Vec<f64>,
    /// One mesh entity per chunk that has occupied voxels.
    chunks: HashMap<IVec3, Entity>,
    material: Handle<StandardMaterial>,
    /// Height range (y keys) of the color scale. It only grows (until the map is
    /// emptied): a wider range recolors every chunk.
    y_range: Option<(i32, i32)>,
    show: bool,
    show_world: bool,
    mode: Option<sim::PoseMode>,
    /// 3D flight: climb over what can be flown over.
    pub three_d: bool,
    flight: Flight3d,
}

#[derive(Component)]
struct VoxelMesh;

pub struct Map3dPlugin;

impl Plugin for Map3dPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(FixedUpdate, (integrate.after(estimate), guard.after(explore).after(navigate).after(crate::inspect::InspectSet).before(step_drone)))
            .add_systems(Update, (reset_on_rebuild, keys, redraw).chain())
;
    }
}

fn setup(mut commands: Commands, config: Res<DroneConfig>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.insert_resource(Map3d {
        map: VoxelMap::new(VOXEL_M),
        last_t: Vec::new(),
        chunks: HashMap::default(),
        material,
        y_range: None,
        show: true,
        // Non-interactive check: `VIEWER_MAP_ONLY=1` starts with the real world hidden.
        show_world: std::env::var_os("VIEWER_MAP_ONLY").is_none(),
        mode: None,
        three_d: config.0.sensors.iter().any(|s| matches!(s.kind, SensorKind::Lidar3D { .. })),
        flight: Flight3d::new(Flight3dConfig::default()),
    });
}

/// Integrates each new range measurement, from the pose the drone believes it has;
/// in 3D flight, projects the changed columns onto the planning grid.
fn integrate(mut m3: ResMut<Map3d>, mut map: ResMut<Map>, est: Res<Estimator>, suites: Query<(&SimDrone, &Suite)>) {
    let m3 = m3.as_mut();
    // A new pose mode rebuilds the map from scratch, like the 2D one.
    if m3.mode.replace(est.mode).is_some_and(|m| m != est.mode) {
        m3.map.clear();
        m3.last_t.clear();
    }
    for (d, suite) in &suites {
        let truth = sensors::Pose::from(&d.state);
        let believed = sensors::Pose::from(&est.belief);
        m3.last_t.resize(suite.0.latest.len(), f64::NEG_INFINITY);
        for (k, m) in suite.0.latest.iter().enumerate() {
            let Some(m) = m else { continue };
            if m.t <= m3.last_t[k] {
                continue;
            }
            m3.last_t[k] = m.t;
            if est.ekf.is_some() {
                m3.map.integrate(&m.reposed(&truth, &believed));
            } else {
                m3.map.integrate(m);
            }
        }
    }
    let dirty = m3.map.take_dirty();
    match (m3.three_d, map.projected) {
        (true, true) => m3.flight.project(&m3.map, &mut map.grid, &dirty),
        (true, false) => {
            // Switched on, or the grid was reset: project everything known.
            map.grid.clear();
            let all: Vec<_> = m3.map.known_columns().collect();
            m3.flight.project(&m3.map, &mut map.grid, &all);
            map.projected = true;
        }
        (false, true) => {
            // Switched off: back to the slab, built from the next measurements.
            map.reset();
        }
        (false, false) => {}
    }
    if m3.three_d && !dirty.is_empty() {
        map.touch();
    }
}

/// In 3D flight, during a mission, the planner's command gets its altitude and
/// its horizontal speed limit from the 3D map.
#[allow(clippy::too_many_arguments)]
fn guard(
    time: Res<Time>,
    mut m3: ResMut<Map3d>,
    est: Res<Estimator>,
    config: Res<DroneConfig>,
    ex: Res<Exploration>,
    mission: Res<Mission>,
    insp: Res<crate::inspect::Inspection>,
    mut cmd: ResMut<PilotCommand>,
) {
    let m3 = m3.as_mut();
    if !m3.three_d || !(ex.active() || mission.active() || insp.inspector.is_some()) {
        m3.flight.target = None;
        return;
    }
    cmd.0 = m3.flight.guard(&m3.map, &est.belief, &config.0.airframe, cmd.0, time.delta_secs(), insp.wanted);
}

fn reset_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, mut m3: ResMut<Map3d>) {
    if rebuilt.read().count() > 0 {
        m3.map.clear();
        m3.last_t.clear();
    }
}

fn keys(input: Res<ButtonInput<KeyCode>>, mut m3: ResMut<Map3d>) {
    if input.just_pressed(KeyCode::KeyM) {
        m3.show = !m3.show;
    }
    if input.just_pressed(KeyCode::KeyV) {
        m3.show_world = !m3.show_world;
    }
}

/// Height color: deep blue low, through teal and yellow, to red high.
fn height_color(k: f32) -> [f32; 4] {
    let k = k.clamp(0.0, 1.0);
    let c = Color::hsl(230.0 - 230.0 * k, 0.7, 0.35 + 0.2 * k).to_linear();
    [c.red, c.green, c.blue, 1.0]
}

/// Cube faces of the occupied voxels of a chunk, only where they face a
/// non-occupied voxel. `None` if the chunk has no occupied voxel.
fn chunk_mesh(map: &VoxelMap, chunk: IVec3, (lo, hi): (i32, i32)) -> Option<Mesh> {
    let span = (hi - lo).max(1) as f32;
    let s = map.voxel_m();
    // Slightly smaller than the voxel: the cubes read as cubes.
    let half = 0.46 * s;
    let faces: [(IVec3, Vec3, Vec3); 6] = [
        (IVec3::X, Vec3::Y, Vec3::Z),
        (IVec3::NEG_X, Vec3::Z, Vec3::Y),
        (IVec3::Y, Vec3::Z, Vec3::X),
        (IVec3::NEG_Y, Vec3::X, Vec3::Z),
        (IVec3::Z, Vec3::X, Vec3::Y),
        (IVec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let mut any = false;
    for k in map.chunk_occupied(chunk) {
        any = true;
        let c = map.center(k);
        let color = height_color((k.y - lo) as f32 / span);
        for (n, u, v) in faces {
            if map.is_occupied(k + n) {
                continue;
            }
            let nf = n.as_vec3();
            let base = positions.len() as u32;
            let o = c + nf * half;
            // u × v = n: counter-clockwise seen from outside.
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                positions.push((o + u * a * half + v * b * half).to_array());
                normals.push(nf.to_array());
                colors.push(color);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    any.then(|| {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
            .with_inserted_indices(Indices::U32(indices))
    })
}

/// Height range (y keys) of the occupied voxels of some chunks.
fn y_range_of(map: &VoxelMap, chunks: &[IVec3]) -> Option<(i32, i32)> {
    chunks
        .iter()
        .flat_map(|&c| map.chunk_occupied(c))
        .fold(None, |r, k| Some(r.map_or((k.y, k.y), |(lo, hi): (i32, i32)| (lo.min(k.y), hi.max(k.y)))))
}

#[allow(clippy::too_many_arguments)]
fn redraw(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: Local<f32>,
    mut m3: ResMut<Map3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut shown: Query<(&mut Visibility, &Mesh3d), With<VoxelMesh>>,
    mut cams: Query<&mut RenderLayers, With<OuterCamera>>,
) {
    let visibility = if m3.show { Visibility::Inherited } else { Visibility::Hidden };
    for (mut v, _) in &mut shown {
        if *v != visibility {
            *v = visibility;
        }
    }
    // Hiding the real world: the outside camera stops rendering layer 0.
    let layers = if m3.show_world { RenderLayers::from_layers(&[0, DRONE_LAYER]) } else { RenderLayers::layer(DRONE_LAYER) };
    for mut l in &mut cams {
        if *l != layers {
            *l = layers.clone();
        }
    }
    *timer += time.delta_secs();
    if *timer < REDRAW_S || !m3.show {
        return;
    }
    *timer = 0.0;
    let m3 = m3.as_mut();
    let mut dirty = m3.map.take_dirty_chunks();
    if dirty.is_empty() {
        return;
    }
    // A wider height range recolors everything already drawn.
    if let Some((lo, hi)) = y_range_of(&m3.map, &dirty) {
        let range = m3.y_range.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi)));
        if m3.y_range != Some(range) {
            if m3.y_range.is_some() {
                dirty.extend(m3.chunks.keys().copied());
                dirty.sort_unstable_by_key(|c| c.to_array());
                dirty.dedup();
            }
            m3.y_range = Some(range);
        }
    }
    let range = m3.y_range.unwrap_or((0, CHUNK));
    for c in dirty {
        match (chunk_mesh(&m3.map, c, range), m3.chunks.get(&c)) {
            (Some(mesh), Some(&e)) => {
                if let Ok((_, h)) = shown.get(e) {
                    let _ = meshes.insert(&h.0, mesh);
                }
            }
            (Some(mesh), None) => {
                let e = commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(m3.material.clone()),
                        Transform::IDENTITY,
                        visibility,
                        // Outside view only: the onboard camera sees the real world.
                        RenderLayers::layer(DRONE_LAYER),
                        NotShadowCaster,
                        VoxelMesh,
                    ))
                    .id();
                m3.chunks.insert(c, e);
            }
            (None, Some(_)) => {
                let e = m3.chunks.remove(&c).unwrap();
                commands.entity(e).despawn();
            }
            (None, None) => {}
        }
    }
    if m3.chunks.is_empty() {
        m3.y_range = None;
    }
}

/// 3D map and flight section of the side panel.
pub fn panel(ui: &mut egui::Ui, m3: &mut Map3d, ex: &mut Exploration, mission: &mut Mission) {

            ui.checkbox(&mut m3.show, "Show the voxel map (M)");
            ui.checkbox(&mut m3.show_world, "Show the real world (V)");
            let c = m3.map.counts();
            let v = m3.map.voxel_m();
            ui.label(format!(
                "{} occupied voxels ({:.0} cm, {} chunks), {:.0} m³ known free",
                c.occupied,
                v * 100.0,
                m3.chunks.len(),
                c.free as f32 * v.powi(3)
            ));
            if ui.button("Clear the 3D map").clicked() {
                m3.map.clear();
            }
            ui.separator();
            ui.checkbox(&mut m3.three_d, "3D flight: climb over what can be flown over")
                .on_hover_text("missions plan on the 3D map; off: flat flight on the 2D slab");
            let three_d = m3.three_d;
            let cfg = &mut m3.flight.cfg;
            ui.add_enabled(
                three_d,
                egui::Slider::new(&mut cfg.clearance_v_m, 1.0..=8.0).text("height above surfaces (m)"),
            );
            ui.add_enabled(
                three_d,
                egui::Slider::new(&mut cfg.max_overfly_m, 2.0..=30.0).text("fly over obstacles up to (m)"),
            );
            let mut clearance = ex.clearance_m;
            if ui
                .add(egui::Slider::new(&mut clearance, 0.2..=3.0).text("distance from walls (m)"))
                .on_hover_text("beyond the drone's radius; applies at the next mission start")
                .changed()
            {
                ex.clearance_m = clearance;
                mission.clearance_m = clearance;
            }
            cfg.clearance_h_m = (clearance * 0.75).max(0.2);
            if let Some(t) = m3.flight.target {
                ui.label(format!("Target altitude: {t:.1} m"));
            }
}
