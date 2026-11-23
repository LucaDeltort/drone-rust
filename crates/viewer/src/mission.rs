//! A-to-B mission in the viewer: choosing B, fixed-step navigator, trajectory
//! as gizmos, unexpected obstacle on demand, control panel.
//!
//! Keys: G = start / stop the mission, B = place B under the cursor,
//! O = make an obstacle appear in front of the drone.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use planning::{NavStatus, Navigator, NavigatorConfig};
use world::ObstacleShape;

use crate::{
    CurrentWorld, OuterCamera, WorldRebuilt,
    drone::{DroneConfig, PilotCommand, SimDrone, step_drone},
    explore::Exploration,
    mapview::Map,
    scene::spawn_obstacle,
};

#[derive(Resource)]
pub struct Mission {
    /// Active navigator; `None` = no mission.
    pub navigator: Option<Navigator>,
    /// Point B (x, z).
    pub goal: Vec2,
    /// Simulation time elapsed since start, in s.
    pub elapsed: f32,
    /// Time predicted by the first trajectory, in s.
    pub planned: Option<f32>,
    pub finished_in: Option<f32>,
    pub distance_m: f32,
    /// Obstacles added during the session (for display).
    pub dropped: u32,
}

impl Mission {
    pub fn active(&self) -> bool {
        self.navigator.is_some()
    }
}

/// Default B: beyond the far wall of the corridor.
const DEFAULT_GOAL: Vec2 = Vec2::new(0.0, -38.0);

pub struct MissionPlugin;

impl Plugin for MissionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Mission {
            navigator: None,
            goal: DEFAULT_GOAL,
            elapsed: 0.0,
            planned: None,
            finished_in: None,
            distance_m: 0.0,
            dropped: 0,
        })
        .add_systems(FixedUpdate, navigate.before(step_drone))
        .add_systems(Update, (keys, pick_goal, drop_obstacle, reset_on_rebuild, draw))
        .add_systems(EguiPrimaryContextPass, panel);
    }
}

/// Starts the mission (and stops the exploration: only one mission at a time).
pub fn start(mission: &mut Mission, ex: &mut Exploration, map: &Map, config: &DroneConfig) {
    ex.explorer = None;
    let altitude = (map.grid.y_lo + map.grid.y_hi) / 2.0;
    let cfg = NavigatorConfig::new(&config.0.airframe);
    mission.navigator = Some(Navigator::new(cfg, altitude, mission.goal));
    mission.elapsed = 0.0;
    mission.planned = None;
    mission.finished_in = None;
    mission.distance_m = 0.0;
}

fn keys(
    input: Res<ButtonInput<KeyCode>>,
    mut frames: Local<u32>,
    mut mission: ResMut<Mission>,
    mut ex: ResMut<Exploration>,
    map: Res<Map>,
    config: Res<DroneConfig>,
) {
    // Non-interactive check: `VIEWER_MISSION=1` starts the mission at start-up.
    *frames = frames.saturating_add(1);
    let auto = *frames == 10 && std::env::var_os("VIEWER_MISSION").is_some();
    if auto || input.just_pressed(KeyCode::KeyG) {
        if mission.active() && !auto {
            mission.navigator = None;
        } else {
            start(&mut mission, &mut ex, &map, &config);
        }
    }
}

/// B = world point under the cursor (ray from the outside camera).
fn pick_goal(
    input: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<OuterCamera>>,
    world: Res<CurrentWorld>,
    mut mission: ResMut<Mission>,
) {
    if !input.just_pressed(KeyCode::KeyB) {
        return;
    }
    let (Ok(window), Ok((camera, transform))) = (windows.single(), cameras.single()) else { return };
    let Some(cursor) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(transform, cursor) else { return };
    if let Some(hit) = world.0.raycast(ray.origin, *ray.direction, 1000.0) {
        mission.goal = Vec2::new(hit.point.x, hit.point.z);
        // A running mission heads to the new B.
        if let Some(nav) = &mission.navigator {
            let (cfg, altitude) = (nav.cfg, nav.altitude);
            let goal = mission.goal;
            mission.navigator = Some(Navigator::new(cfg, altitude, goal));
        }
    }
}

/// Unexpected obstacle: a 6 m wall across, 8 m in front of the drone.
fn drop_obstacle(
    input: Res<ButtonInput<KeyCode>>,
    drones: Query<&SimDrone>,
    mut world: ResMut<CurrentWorld>,
    mut mission: ResMut<Mission>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !input.just_pressed(KeyCode::KeyO) {
        return;
    }
    let Ok(d) = drones.single() else { return };
    let ahead = d.state.position + d.state.forward() * 8.0;
    let shape = ObstacleShape::Box { size: Vec3::new(6.0, 5.0, 0.5) };
    let o = world.0.add_obstacle(shape, ahead.x, ahead.z, d.state.yaw);
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.3, 0.2),
        perceptual_roughness: 0.8,
        ..default()
    });
    spawn_obstacle(&o, material, &mut commands, &mut meshes);
    mission.dropped += 1;
}

fn navigate(
    time: Res<Time>,
    mut mission: ResMut<Mission>,
    map: Res<Map>,
    config: Res<DroneConfig>,
    drones: Query<&SimDrone>,
    est: Res<crate::pose::Estimator>,
    mut cmd: ResMut<PilotCommand>,
) {
    let dt = time.delta_secs();
    let m = mission.as_mut();
    let (Some(nav), Ok(d)) = (m.navigator.as_mut(), drones.single()) else {
        return;
    };
    cmd.0 = nav.update(&map.grid, &est.belief, &config.0.airframe, dt);
    if m.planned.is_none() {
        m.planned = nav.eta();
    }
    if nav.status == NavStatus::Reached {
        m.finished_in.get_or_insert(m.elapsed);
    } else {
        m.elapsed += dt;
        m.distance_m += d.state.velocity.length() * dt;
    }
}

fn reset_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, mut mission: ResMut<Mission>) {
    if rebuilt.read().count() == 0 {
        return;
    }
    mission.navigator = None;
    mission.finished_in = None;
}

/// B (green), smoothed path (white), trajectory: one point every 0.5 s of predicted
/// flight (cyan); tight points = slow, spread out = fast.
fn draw(mission: Res<Mission>, world: Res<CurrentWorld>, drones: Query<&SimDrone>, mut gizmos: Gizmos) {
    let Ok(d) = drones.single() else { return };
    let y = d.state.position.y;
    let at = |p: Vec2| Vec3::new(p.x, y, p.y);
    let green = Color::srgb(0.2, 1.0, 0.4);
    let ground = world.0.terrain.height_at(mission.goal.x, mission.goal.y);
    let goal = Vec3::new(mission.goal.x, ground, mission.goal.y);
    gizmos.line(goal, goal + Vec3::Y * 6.0, green);
    let flat = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    gizmos.circle(Isometry3d::new(goal + Vec3::Y * 0.05, flat), 0.6, green);

    let Some(nav) = &mission.navigator else { return };
    for w in nav.path.windows(2) {
        gizmos.line(at(w[0]), at(w[1]), Color::WHITE);
    }
    if let Some(traj) = &nav.trajectory {
        let cyan = Color::srgb(0.2, 0.9, 1.0);
        let mut t = 0.5;
        while t < traj.duration() {
            gizmos.sphere(Isometry3d::from_translation(at(traj.sample(t))), 0.12, cyan);
            t += 0.5;
        }
    }
}

fn panel(
    mut contexts: EguiContexts,
    mut mission: ResMut<Mission>,
    mut ex: ResMut<Exploration>,
    map: Res<Map>,
    config: Res<DroneConfig>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("Mission A -> B")
        .anchor(egui::Align2::CENTER_TOP, [0.0, 12.0])
        .resizable(false)
        .show(ctx, |ui| {
            let running = mission.active();
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    ui.label("B:");
                    ui.add(egui::DragValue::new(&mut mission.goal.x).speed(0.5).prefix("x "));
                    ui.add(egui::DragValue::new(&mut mission.goal.y).speed(0.5).prefix("z "));
                });
            });
            ui.label("B under the cursor: key B");
            let label = if running { "Stop (G)" } else { "Go to B (G)" };
            if ui.button(label).clicked() {
                if running {
                    mission.navigator = None;
                } else {
                    start(&mut mission, &mut ex, &map, &config);
                }
            }
            ui.label(format!("Unexpected obstacle in front of the drone: key O ({} placed)", mission.dropped));

            let Some(nav) = &mission.navigator else {
                ui.label("Status: no mission");
                return;
            };
            let status = match (nav.status, mission.finished_in) {
                (NavStatus::Reached, Some(s)) => format!("B reached in {s:.1} s"),
                (NavStatus::NoPath, _) => "no path to B for now".to_string(),
                _ => format!("flying, {:.1} s", mission.elapsed),
            };
            ui.label(format!("Status: {status}"));
            if let Some(p) = mission.planned {
                ui.label(format!("Predicted time at start: {p:.1} s"));
            }
            if let Some(eta) = nav.eta() {
                ui.label(format!("Predicted remaining time: {eta:.1} s"));
            }
            ui.label(format!("Distance travelled: {:.0} m", mission.distance_m));
            let s = nav.stats;
            ui.label(format!(
                "D* Lite: {} replans, {} cells expanded",
                s.replans, s.expanded
            ));
            let text = format!("Collisions: {}", ex.collisions);
            if ex.collisions == 0 {
                ui.label(text);
            } else {
                ui.colored_label(egui::Color32::from_rgb(255, 90, 70), text);
            }
        });
    Ok(())
}
