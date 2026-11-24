//! Drone pose as seen by the AI: ground truth, or Kalman filter estimate
//! (with or without GPS). The map and the missions only use [`Estimator::belief`].
//!
//! In estimated mode, each range scan is also matched on the map to correct the pose
//! (scan matching, see [`sim::scan_correct`]). A magenta ghost shows where the drone
//! believes it is, surrounded by a 2σ circle (uncertainty announced by the filter).

use bevy::prelude::*;
use bevy_egui::egui;
use estimation::{Ekf, EkfConfig};
use sim::{PoseMode, sensors_for};
use world::drone::DroneState;

use crate::{
    WorldRebuilt,
    drone::{DroneConfig, SimDrone},
    explore::Exploration,
    mapview::Map,
    mission::Mission,
    sensing::{SENSOR_SEED, Suite, sense},
};

#[derive(Resource)]
pub struct Estimator {
    pub mode: PoseMode,
    pub ekf: Option<Ekf>,
    /// What the drone believes about its state.
    pub belief: DroneState,
    /// Real horizontal error of the belief, in m (display only).
    pub error_m: f32,
    /// Match each range scan on the map to correct the pose.
    pub scan_match: bool,
    /// Scan matches accepted by the EKF, and attempted, since the last reset.
    pub scan_matches: u32,
    pub scan_attempts: u32,
    /// Navigation sensor, by index in the suite.
    nav: Vec<bool>,
    last_t: Vec<f64>,
    /// Mode requested by the UI, applied at the next step.
    pending: Option<PoseMode>,
    /// The drone has just been reset: restart from a certain estimate.
    reset: bool,
}

impl Estimator {
    pub fn new(mode: PoseMode) -> Self {
        Self {
            mode,
            ekf: None,
            belief: DroneState {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
                yaw: 0.0,
                acceleration: Vec3::ZERO,
                yaw_rate: 0.0,
                flight: Default::default(),
            },
            error_m: 0.0,
            scan_match: true,
            scan_matches: 0,
            scan_attempts: 0,
            nav: Vec::new(),
            last_t: Vec::new(),
            pending: Some(mode),
            reset: true,
        }
    }
}

pub struct PosePlugin(pub PoseMode);

impl Plugin for PosePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Estimator::new(self.0))
            .add_systems(FixedUpdate, estimate.after(sense))
            .add_systems(Update, (reset_on_rebuild, draw))
;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn estimate(
    time: Res<Time>,
    mut est: ResMut<Estimator>,
    config: Res<DroneConfig>,
    mut drones: Query<(&SimDrone, &mut Suite)>,
    mut map: ResMut<Map>,
    mut ex: ResMut<Exploration>,
    mut mission: ResMut<Mission>,
) {
    let Ok((d, mut suite)) = drones.single_mut() else { return };
    let t = time.elapsed_secs_f64();
    let est = est.as_mut();

    if let Some(mode) = est.pending.take() {
        // New mode: the mode's sensors, map and missions start from scratch (the
        // map was built with the old pose).
        let sensors = sensors_for(&config.0, mode);
        est.nav = sensors.iter().map(|s| s.kind.is_navigation()).collect();
        suite.0 = sensors::SensorSuite::new(&sensors, SENSOR_SEED);
        est.last_t = vec![f64::NEG_INFINITY; sensors.len()];
        est.mode = mode;
        est.reset = true;
        map.reset();
        ex.explorer = None;
        mission.navigator = None;
    }
    if est.reset {
        est.reset = false;
        est.ekf = match est.mode {
            PoseMode::Truth => None,
            PoseMode::Ekf { .. } => EkfConfig::from_spec(&config.0).map(|cfg| Ekf::new(cfg, &d.state, t)),
        };
        est.scan_matches = 0;
        est.scan_attempts = 0;
    }

    if let Some(ekf) = &mut est.ekf {
        for (k, m) in suite.0.latest.iter().enumerate() {
            if let Some(m) = m
                && est.nav.get(k).copied().unwrap_or(false)
                && m.t > est.last_t[k]
            {
                ekf.feed(m);
                est.last_t[k] = m.t;
            }
        }
        // Then the new range scans, aligned on the map before they are added to it.
        if est.scan_match && est.mode == (PoseMode::Ekf { gps: false }) {
            let truth = sensors::Pose::from(&d.state);
            let (mut matches, mut attempts) = (0, 0);
            for (k, m) in suite.0.latest.iter().enumerate() {
                if let Some(m) = m
                    && !est.nav.get(k).copied().unwrap_or(true)
                    && m.t > est.last_t[k]
                {
                    est.last_t[k] = m.t;
                    if let Some(accepted) = sim::scan_correct(ekf, &map.grid, m, &truth, &mapping::MatchConfig::default()) {
                        attempts += 1;
                        matches += accepted as u32;
                    }
                }
            }
            est.scan_matches += matches;
            est.scan_attempts += attempts;
        }
    }
    est.belief = est.ekf.as_ref().map_or(d.state, Ekf::state);
    let e = est.belief.position - d.state.position;
    est.error_m = (e.x * e.x + e.z * e.z).sqrt();
}

fn reset_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, mut est: ResMut<Estimator>) {
    if rebuilt.read().count() > 0 {
        est.reset = true;
    }
}

fn draw(est: Res<Estimator>, mut gizmos: Gizmos) {
    let Some(ekf) = &est.ekf else { return };
    let b = &est.belief;
    let magenta = Color::srgb(1.0, 0.3, 0.9);
    gizmos.sphere(Isometry3d::from_translation(b.position), 0.3, magenta);
    gizmos.line(b.position, b.position + b.forward() * 0.8, magenta);
    let flat = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let radius = (2.0 * ekf.horizontal_std()).max(0.05);
    gizmos.circle(Isometry3d::new(b.position, flat), radius, magenta.with_alpha(0.7));
}

/// Pose section of the side panel.
pub fn panel(ui: &mut egui::Ui, est: &mut Estimator, config: &DroneConfig) {

            let mut mode = est.mode;
            ui.horizontal(|ui| {
                ui.radio_value(&mut mode, PoseMode::Truth, "Truth");
                ui.radio_value(&mut mode, PoseMode::Ekf { gps: true }, "EKF + GPS");
                ui.radio_value(&mut mode, PoseMode::Ekf { gps: false }, "EKF without GPS");
            });
            if mode != est.mode && est.pending.is_none() {
                est.pending = Some(mode);
            }
            ui.label("Changing mode clears the map and stops the mission.");
            let mut scan_match = est.scan_match;
            match &est.ekf {
                None if est.mode != PoseMode::Truth => {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 90, 70),
                        format!("{} has no IMU: perfect pose used", config.0.name),
                    );
                }
                None => {
                    ui.label("Perfect pose (ground truth)");
                }
                Some(ekf) => {
                    ui.label(format!(
                        "Real error: {:.2} m   Announced uncertainty (1σ): {:.2} m",
                        est.error_m,
                        ekf.horizontal_std()
                    ));
                    if est.mode == (PoseMode::Ekf { gps: false }) {
                        ui.checkbox(&mut scan_match, "Scan matching (lidar / depth on the map)");
                    }
                    if scan_match && est.mode == (PoseMode::Ekf { gps: false }) {
                        ui.label(format!("Scan matches accepted: {} / {}", est.scan_matches, est.scan_attempts));
                    }
                    let (ba, bg) = ekf.biases();
                    ui.label(format!(
                        "Yaw ±{:.1}°   estimated biases: accel {:.3} m/s², gyro {:.2} °/s",
                        ekf.yaw_std().to_degrees(),
                        ba.length(),
                        bg.to_degrees()
                    ));
                }
            }
            est.scan_match = scan_match;
}
