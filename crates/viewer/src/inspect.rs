//! Inspection in the viewer: `T` places the target under the cursor, `I` starts
//! (or stops) the inspection of the structure there. The exploration can also be
//! kept within a radius of the target.

use bevy::prelude::*;
use bevy_egui::egui;
use planning::{InspectConfig, Inspector, Phase, Zone};

use crate::{
    CurrentWorld, OuterCamera, WorldRebuilt,
    drone::{DroneConfig, PilotCommand, step_drone},
    explore::{Exploration, explore},
    map3d::Map3d,
    mapview::Map,
    mission::{Mission, navigate},
    pose::Estimator,
};

#[derive(Resource)]
pub struct Inspection {
    pub inspector: Option<Inspector>,
    /// Target point (x, z).
    pub point: Vec2,
    /// Radius kept by the exploration when it is limited to the target, in m.
    pub zone_radius: f32,
    /// Altitude asked by the inspector at the last step, for the 3D flight guard.
    pub wanted: Option<f32>,
    pub elapsed: f32,
    pub finished_in: Option<f32>,
}

impl Inspection {
    pub fn active(&self) -> bool {
        self.inspector.as_ref().is_some_and(|i| !matches!(i.phase, Phase::Done | Phase::NotFound))
    }

    pub fn zone(&self) -> Zone {
        Zone { center: self.point, radius: self.zone_radius }
    }
}

/// The inspector's step, before the 3D flight guard.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct InspectSet;

pub struct InspectPlugin;

impl Plugin for InspectPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Inspection {
            inspector: None,
            point: Vec2::new(-22.0, -18.0),
            zone_radius: 15.0,
            wanted: None,
            elapsed: 0.0,
            finished_in: None,
        })
        .add_systems(FixedUpdate, run.in_set(InspectSet).after(explore).after(navigate).before(step_drone))
        .add_systems(Update, (keys, pick_target, reset_on_rebuild, draw));
    }
}

/// Starts the inspection (and stops the other missions: one at a time).
pub fn start(insp: &mut Inspection, ex: &mut Exploration, mission: &mut Mission, home: Vec2) {
    ex.explorer = None;
    mission.navigator = None;
    insp.inspector = Some(Inspector::new(InspectConfig::default(), insp.point, home));
    insp.elapsed = 0.0;
    insp.finished_in = None;
}

fn keys(
    input: Res<ButtonInput<KeyCode>>,
    mut frames: Local<u32>,
    mut insp: ResMut<Inspection>,
    mut ex: ResMut<Exploration>,
    mut mission: ResMut<Mission>,
    m3: Res<Map3d>,
) {
    // Non-interactive check: `VIEWER_INSPECT=1` starts the inspection at start-up.
    *frames = frames.saturating_add(1);
    let auto = *frames == 10 && std::env::var_os("VIEWER_INSPECT").is_some();
    if !(auto || input.just_pressed(KeyCode::KeyI)) {
        return;
    }
    if insp.active() && !auto {
        insp.inspector = None;
    } else if m3.three_d {
        start(&mut insp, &mut ex, &mut mission, Vec2::ZERO);
    }
}

/// Target = world point under the cursor (ray from the outside camera).
fn pick_target(
    input: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<OuterCamera>>,
    world: Res<CurrentWorld>,
    mut insp: ResMut<Inspection>,
) {
    if !input.just_pressed(KeyCode::KeyT) {
        return;
    }
    let (Ok(window), Ok((camera, transform))) = (windows.single(), cameras.single()) else { return };
    let Some(cursor) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(transform, cursor) else { return };
    if let Some(hit) = world.0.raycast(ray.origin, *ray.direction, 1000.0) {
        insp.point = Vec2::new(hit.point.x, hit.point.z);
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    time: Res<Time>,
    mut insp: ResMut<Inspection>,
    ex: Res<Exploration>,
    mission: Res<Mission>,
    map: Res<Map>,
    m3: Res<Map3d>,
    est: Res<Estimator>,
    config: Res<DroneConfig>,
    mut cmd: ResMut<PilotCommand>,
) {
    let dt = time.delta_secs();
    let insp = insp.as_mut();
    insp.wanted = None;
    // Another mission was started: it takes over.
    if ex.active() || mission.active() {
        insp.inspector = None;
        return;
    }
    let Some(inspector) = insp.inspector.as_mut() else { return };
    let (c, wanted) = inspector.update(&map.grid, &m3.map, &est.belief, &config.0.airframe, dt);
    if matches!(inspector.phase, Phase::Done | Phase::NotFound) {
        insp.finished_in.get_or_insert(insp.elapsed);
    } else {
        insp.elapsed += dt;
    }
    cmd.0 = c;
    insp.wanted = wanted;
}

fn reset_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, mut insp: ResMut<Inspection>) {
    if rebuilt.read().count() > 0 {
        insp.inspector = None;
    }
}

fn draw(insp: Res<Inspection>, ex: Res<Exploration>, world: Res<CurrentWorld>, mut gizmos: Gizmos) {
    let flat = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let orange = Color::srgb(1.0, 0.6, 0.1);
    let ground = world.0.terrain.height_at(insp.point.x, insp.point.y);
    let p = Vec3::new(insp.point.x, ground, insp.point.y);
    gizmos.line(p, p + Vec3::Y * 8.0, orange);
    gizmos.circle(Isometry3d::new(p + Vec3::Y * 0.05, flat), 0.8, orange);
    if ex.zone.is_some() {
        gizmos.circle(Isometry3d::new(p + Vec3::Y * 0.1, flat), insp.zone_radius, orange.with_alpha(0.5));
    }
    let Some(i) = &insp.inspector else { return };
    if let Some(t) = i.target {
        // Footprint, at the ground and at the top.
        for y in [t.ground, t.top] {
            let c = Vec3::new(t.center.x, y, t.center.y);
            gizmos.rect(Isometry3d::new(c, flat), t.half * 2.0, Color::srgb(1.0, 0.85, 0.2));
        }
    }
    let path: Vec<Vec3> = i.plan.iter().skip(i.next).map(|w| w.pos).collect();
    if i.phase == Phase::Survey && !path.is_empty() {
        gizmos.linestrip(path.iter().copied(), Color::srgba(1.0, 1.0, 1.0, 0.5));
        gizmos.sphere(Isometry3d::from_translation(path[0]), 0.3, Color::srgb(1.0, 0.3, 0.9));
    }
}

/// Inspection section of the side panel.
pub fn panel(ui: &mut egui::Ui, insp: &mut Inspection, ex: &mut Exploration, mission: &mut Mission, three_d: bool) {
    ui.horizontal(|ui| {
        ui.label("Target:");
        ui.add(egui::DragValue::new(&mut insp.point.x).speed(0.5).prefix("x "));
        ui.add(egui::DragValue::new(&mut insp.point.y).speed(0.5).prefix("z "));
    });
    ui.label("Target under the cursor: key T");
    let running = insp.active();
    let label = if running { "Stop (I)" } else { "Inspect the target (I)" };
    if ui.add_enabled(three_d || running, egui::Button::new(label)).clicked() {
        if running {
            insp.inspector = None;
        } else {
            start(insp, ex, mission, Vec2::ZERO);
        }
    }
    if !three_d {
        ui.weak("Needs 3D flight (a drone with a 3D lidar).");
    }
    if let Some(i) = &insp.inspector {
        let status = match i.phase {
            Phase::Approach => "flying to the target, looking for it".to_string(),
            Phase::Survey => format!("circling it: {:.0} %", 100.0 * i.progress()),
            Phase::Return => "coming back".to_string(),
            Phase::Done => format!("done in {:.0} s", insp.finished_in.unwrap_or(insp.elapsed)),
            Phase::NotFound => "nothing to inspect there".to_string(),
        };
        ui.label(format!("Status: {status}"));
        if let Some(t) = i.target {
            ui.label(format!(
                "Structure: {:.1} × {:.1} m, {:.1} m high",
                2.0 * t.half.x,
                2.0 * t.half.y,
                t.top - t.ground
            ));
        }
        if i.skipped > 0 {
            ui.label(format!("Waypoints skipped: {}", i.skipped));
        }
    }
    ui.separator();
    let mut limited = ex.zone.is_some();
    ui.add_enabled_ui(!ex.active(), |ui| {
        if ui.checkbox(&mut limited, "Explore only around the target").changed() {
            ex.zone = limited.then(|| insp.zone());
        }
        if ui.add(egui::Slider::new(&mut insp.zone_radius, 5.0..=60.0).text("radius (m)")).changed() && limited {
            ex.zone = Some(insp.zone());
        }
    });
    if limited {
        ex.zone = Some(insp.zone());
    }
}
