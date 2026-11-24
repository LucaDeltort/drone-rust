//! Autonomous exploration in the viewer: fixed-step explorer, path as gizmos,
//! control panel, collision counter.

use bevy::prelude::*;
use bevy_egui::egui;
use planning::{Explorer, ExplorerConfig, Status, Strategy};

use crate::{
    WorldRebuilt,
    drone::{DroneConfig, PilotCommand, SimDrone, step_drone},
    mapview::Map,
    mission::Mission,
};

#[derive(Resource)]
pub struct Exploration {
    /// Active explorer; `None` = manual piloting.
    pub explorer: Option<Explorer>,
    pub strategy: Strategy,
    pub lambda: f32,
    /// Simulation time elapsed since start, in s.
    pub elapsed: f32,
    /// Duration of the full exploration, if it is finished.
    pub finished_in: Option<f32>,
    pub distance_m: f32,
    /// Collisions (contact entries) since the last world change.
    pub collisions: u32,
    touching: bool,
    /// Distance kept from obstacles, beyond the drone's radius, in m.
    pub clearance_m: f32,
}

impl Exploration {
    pub fn active(&self) -> bool {
        self.explorer.is_some()
    }
}

/// Default distance kept from obstacles, beyond the drone's radius, in m.
pub const DEFAULT_CLEARANCE_M: f32 = 0.8;

pub struct ExplorePlugin;

impl Plugin for ExplorePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Exploration {
            explorer: None,
            strategy: Strategy::InfoGain,
            lambda: 10.0,
            elapsed: 0.0,
            finished_in: None,
            distance_m: 0.0,
            collisions: 0,
            touching: false,
            clearance_m: DEFAULT_CLEARANCE_M,
        })
        .add_systems(FixedUpdate, (explore.before(step_drone), count_collisions.after(step_drone)))
        .add_systems(Update, (toggle_key, reset_on_rebuild, draw_plan))
;
    }
}

/// Starts the exploration (and stops the A-to-B mission: only one at a time).
fn start(ex: &mut Exploration, mission: &mut Mission, map: &Map, config: &DroneConfig) {
    mission.navigator = None;
    let mut cfg = ExplorerConfig::new(ex.strategy, &config.0.airframe);
    cfg.lambda = ex.lambda;
    cfg.inflate_m = config.0.airframe.radius + ex.clearance_m;
    cfg.unknown_margin_m = cfg.inflate_m;
    let altitude = (map.grid.y_lo + map.grid.y_hi) / 2.0;
    ex.explorer = Some(Explorer::new(cfg, altitude));
    ex.elapsed = 0.0;
    ex.finished_in = None;
    ex.distance_m = 0.0;
}

fn toggle_key(
    input: Res<ButtonInput<KeyCode>>,
    mut frames: Local<u32>,
    mut ex: ResMut<Exploration>,
    mut mission: ResMut<Mission>,
    map: Res<Map>,
    config: Res<DroneConfig>,
) {
    // Non-interactive check: `VIEWER_EXPLORE=1` starts the exploration at start-up,
    // once the world is built (building resets the exploration).
    *frames = frames.saturating_add(1);
    if *frames == 10 && std::env::var_os("VIEWER_EXPLORE").is_some() {
        start(&mut ex, &mut mission, &map, &config);
        return;
    }
    if input.just_pressed(KeyCode::KeyX) {
        if ex.active() {
            ex.explorer = None;
        } else {
            start(&mut ex, &mut mission, &map, &config);
        }
    }
}

pub fn explore(
    time: Res<Time>,
    mut ex: ResMut<Exploration>,
    map: Res<Map>,
    config: Res<DroneConfig>,
    drones: Query<&SimDrone>,
    est: Res<crate::pose::Estimator>,
    mut cmd: ResMut<PilotCommand>,
) {
    let dt = time.delta_secs();
    let ex = ex.as_mut();
    let (Some(explorer), Ok(d)) = (ex.explorer.as_mut(), drones.single()) else {
        return;
    };
    cmd.0 = explorer.update(&map.grid, &est.belief, &config.0.airframe, dt);
    if explorer.status == Status::Exploring {
        ex.elapsed += dt;
        ex.distance_m += d.state.velocity.length() * dt;
    } else if ex.finished_in.is_none() {
        ex.finished_in = Some(ex.elapsed);
    }
}

fn count_collisions(mut ex: ResMut<Exploration>, drones: Query<&SimDrone>) {
    let Ok(d) = drones.single() else { return };
    let hit = d.state.flight.contact;
    if hit && !ex.touching {
        ex.collisions += 1;
    }
    ex.touching = hit;
}

fn reset_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, mut ex: ResMut<Exploration>) {
    if rebuilt.read().count() == 0 {
        return;
    }
    ex.explorer = None;
    ex.collisions = 0;
    ex.touching = false;
    ex.finished_in = None;
}

/// Planned path (white), target (magenta), frontier centers (yellow).
fn draw_plan(ex: Res<Exploration>, drones: Query<&SimDrone>, mut gizmos: Gizmos) {
    let (Some(explorer), Ok(d)) = (ex.explorer.as_ref(), drones.single()) else { return };
    let y = d.state.position.y;
    let at = |p: Vec2| Vec3::new(p.x, y, p.y);
    for w in explorer.path.windows(2) {
        gizmos.line(at(w[0]), at(w[1]), Color::WHITE);
    }
    if let Some(t) = explorer.target {
        gizmos.sphere(Isometry3d::from_translation(at(t)), 0.3, Color::srgb(1.0, 0.2, 0.9));
    }
    for &f in &explorer.frontiers {
        let iso = Isometry3d::new(at(f), Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
        gizmos.circle(iso, 0.4, Color::srgb(1.0, 0.9, 0.1));
    }
}

/// Exploration section of the side panel.
pub fn panel(ui: &mut egui::Ui, ex: &mut Exploration, mission: &mut Mission, map: &Map, config: &DroneConfig) {

            let running = ex.active();
            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    ui.radio_value(&mut ex.strategy, Strategy::Nearest, "Nearest frontier");
                    ui.radio_value(&mut ex.strategy, Strategy::InfoGain, "Information gain");
                });
                ui.add_enabled(
                    ex.strategy == Strategy::InfoGain,
                    egui::Slider::new(&mut ex.lambda, 0.5..=50.0)
                        .logarithmic(true)
                        .text("λ (m² of gain per m)"),
                );
            });
            let label = if running { "Stop (X)" } else { "Start exploration (X)" };
            if ui.button(label).clicked() {
                if running {
                    ex.explorer = None;
                } else {
                    start(ex, mission, map, config);
                }
            }

            let status = match (&ex.explorer, ex.finished_in) {
                (None, _) => "manual piloting".to_string(),
                (Some(_), Some(s)) => format!("done in {s:.0} s: no reachable frontier left"),
                (Some(e), None) => format!("running, {:.0} s, {} frontier(s)", ex.elapsed, e.frontiers.len()),
            };
            ui.label(format!("Status: {status}"));
            if ex.active() {
                ui.label(format!("Distance travelled: {:.0} m", ex.distance_m));
            }
            let c = map_counts(map);
            ui.label(format!("Known area: {:.0} m² ({:.1} % of the map)", c.0, c.1));
            let text = format!("Collisions: {}", ex.collisions);
            if ex.collisions == 0 {
                ui.label(text);
            } else {
                ui.colored_label(egui::Color32::from_rgb(255, 90, 70), text);
            }
}

/// Known area (free + occupied) in m², and in % of the map.
fn map_counts(map: &Map) -> (f32, f32) {
    let c = map.counts();
    let cell = map.grid.cell_m().powi(2);
    let known = (c.free + c.occupied) as f32;
    let total = (c.free + c.occupied + c.unknown) as f32;
    (known * cell, 100.0 * known / total.max(1.0))
}
