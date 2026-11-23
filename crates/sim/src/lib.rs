//! Headless simulation: real world, drone, sensors, pose estimation and
//! map. The viewer shows the same thing with Bevy; here, nothing but pure Rust,
//! for mission tests and benchmarks.
//!
//! The boundary between the world and the AI is strict: the AI receives [`Sim::belief`] (what
//! the drone believes about its state) and [`Sim::grid`] (its map), never the truth.

use std::time::{Duration, Instant};

use drone_spec::{DroneSpec, Mount, SensorKind, SensorSpec};
use estimation::{Ekf, EkfConfig};
use mapping::{Cell, OccupancyGrid};
use sensors::{Pose, SensorSuite};
use world::World;
use world::drone::{Command, DroneState};

/// Where the pose used by the map and the planner comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoseMode {
    /// Perfect pose (ground truth).
    Truth,
    /// Extended Kalman filter. `gps`: with a GPS (added if the drone lacks one),
    /// or without (removed if it has one).
    Ekf { gps: bool },
}

impl PoseMode {
    pub fn label(&self) -> &'static str {
        match self {
            PoseMode::Truth => "truth",
            PoseMode::Ekf { gps: true } => "ekf-gps",
            PoseMode::Ekf { gps: false } => "ekf",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "truth" => Some(PoseMode::Truth),
            "ekf-gps" => Some(PoseMode::Ekf { gps: true }),
            "ekf" => Some(PoseMode::Ekf { gps: false }),
            _ => None,
        }
    }
}

/// Take-off area assumed clear around point A, in m.
pub const TAKEOFF_CLEAR_M: f32 = 1.0;

/// GPS added to drones that lack one, in `Ekf { gps: true }` mode.
pub fn default_gps() -> SensorSpec {
    SensorSpec { kind: SensorKind::Gps { noise_std: 1.0 }, mount: Mount::default(), rate_hz: 5.0 }
}

/// Effective sensors of the drone depending on the pose mode.
pub fn sensors_for(spec: &DroneSpec, mode: PoseMode) -> Vec<SensorSpec> {
    let mut sensors = spec.sensors.clone();
    if let PoseMode::Ekf { gps } = mode {
        let has_gps = sensors.iter().any(|s| matches!(s.kind, SensorKind::Gps { .. }));
        if gps && !has_gps {
            sensors.push(default_gps());
        }
        if !gps {
            sensors.retain(|s| !matches!(s.kind, SensorKind::Gps { .. }));
        }
    }
    sensors
}

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub mode: PoseMode,
    /// Sensor noise seed.
    pub seed: u64,
    /// Flight height at start, above the ground, in m.
    pub altitude: f32,
    /// Map cell size, in m.
    pub cell_m: f32,
    /// Half-thickness of the mapped slab, in m.
    pub half_slab: f32,
    /// Simulation step, in s.
    pub dt: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { mode: PoseMode::Truth, seed: 1, altitude: 2.0, cell_m: 0.25, half_slab: 0.5, dt: 0.01 }
    }
}

pub struct Sim {
    pub cfg: SimConfig,
    pub world: World,
    pub spec: DroneSpec,
    /// Real state of the drone.
    pub truth: DroneState,
    pub grid: OccupancyGrid,
    pub suite: SensorSuite,
    pub ekf: Option<Ekf>,
    last_t: Vec<f64>,
    /// Simulation time, in s.
    pub t: f32,
    /// Collisions (contact entries).
    pub collisions: u32,
    touching: bool,
    pub distance_m: f32,
    error_sq_sum: f64,
    error_samples: u64,
    pub max_error_m: f32,
    /// Cumulative compute time of estimation and mapping.
    pub compute: Duration,
}

impl Sim {
    /// Drone at point A = (0, 0). Error if the mode requires an EKF and the drone
    /// has no IMU.
    pub fn new(world: World, spec: &DroneSpec, cfg: SimConfig) -> Result<Self, String> {
        let spec = DroneSpec { sensors: sensors_for(spec, cfg.mode), ..spec.clone() };
        let truth = DroneState::spawn(&world, 0.0, 0.0, cfg.altitude);
        let ekf = match cfg.mode {
            PoseMode::Truth => None,
            PoseMode::Ekf { .. } => {
                let ekf_cfg = EkfConfig::from_spec(&spec).ok_or_else(|| format!("{}: no IMU", spec.name))?;
                Some(Ekf::new(ekf_cfg, &truth, 0.0))
            }
        };
        let y = truth.position.y;
        let mut grid = OccupancyGrid::new(world.terrain.half_size() * 2.0, cfg.cell_m, y - cfg.half_slab, y + cfg.half_slab);
        grid.mark_free_disk(glam::Vec2::new(truth.position.x, truth.position.z), TAKEOFF_CLEAR_M);
        let suite = SensorSuite::new(&spec.sensors, cfg.seed);
        let n = spec.sensors.len();
        Ok(Self {
            cfg,
            world,
            spec,
            truth,
            grid,
            suite,
            ekf,
            last_t: vec![f64::NEG_INFINITY; n],
            t: 0.0,
            collisions: 0,
            touching: false,
            distance_m: 0.0,
            error_sq_sum: 0.0,
            error_samples: 0,
            max_error_m: 0.0,
            compute: Duration::ZERO,
        })
    }

    /// What the drone believes about its state: estimate, or truth in perfect mode.
    pub fn belief(&self) -> DroneState {
        self.ekf.as_ref().map_or(self.truth, Ekf::state)
    }

    /// Targeted absolute flight height (center of the mapped slab).
    pub fn altitude(&self) -> f32 {
        (self.grid.y_lo + self.grid.y_hi) / 2.0
    }

    /// Current horizontal error of the estimated pose, in m.
    pub fn pose_error(&self) -> f32 {
        let e = self.belief().position - self.truth.position;
        (e.x * e.x + e.z * e.z).sqrt()
    }

    /// Root mean square horizontal error since the start, in m.
    pub fn pose_rms(&self) -> f32 {
        if self.error_samples == 0 { 0.0 } else { (self.error_sq_sum / self.error_samples as f64).sqrt() as f32 }
    }

    /// One step: physics, collisions, sensors, estimation, map.
    pub fn step(&mut self, cmd: Command) {
        let dt = self.cfg.dt;
        let before = self.truth.position;
        self.truth.step(cmd, &self.spec.airframe, &self.world, dt);
        self.t += dt;
        self.distance_m += before.distance(self.truth.position);
        let hit = self.world.collides(self.truth.position, self.spec.airframe.radius);
        self.collisions += (hit && !self.touching) as u32;
        self.touching = hit;

        let body = Pose::from(&self.truth);
        self.suite.update(&self.world, &body, self.t as f64);

        let start = Instant::now();
        // Navigation first (the current pose), then the map seen from that pose.
        let fresh: Vec<usize> = (0..self.suite.latest.len())
            .filter(|&k| self.suite.latest[k].as_ref().is_some_and(|m| m.t > self.last_t[k]))
            .collect();
        if let Some(ekf) = &mut self.ekf {
            for &k in &fresh {
                let m = self.suite.latest[k].as_ref().unwrap();
                if self.spec.sensors[k].kind.is_navigation() {
                    ekf.feed(m);
                }
            }
        }
        let believed = Pose::from(&self.belief());
        // Where the drone's body is, there is no obstacle.
        let at = glam::Vec2::new(believed.position.x, believed.position.z);
        self.grid.mark_free_disk(at, self.spec.airframe.radius);
        for &k in &fresh {
            let m = self.suite.latest[k].as_ref().unwrap();
            self.last_t[k] = m.t;
            if self.spec.sensors[k].kind.is_navigation() {
                continue;
            }
            if self.ekf.is_some() {
                self.grid.integrate(&m.reposed(&body, &believed));
            } else {
                self.grid.integrate(m);
            }
        }
        self.compute += start.elapsed();

        let e = self.pose_error();
        self.error_sq_sum += (e * e) as f64;
        self.error_samples += 1;
        self.max_error_m = self.max_error_m.max(e);
    }
}

/// Map cells whose column is entirely empty within the slab (ground
/// truth): the coverage denominator. Computed once per world.
pub fn empty_cells(world: &World, grid: &OccupancyGrid) -> Vec<bool> {
    let (nx, nz) = grid.size();
    let mut out = vec![false; nx * nz];
    for j in 0..nz {
        for i in 0..nx {
            let c = grid.center(i, j);
            out[j * nx + i] = !world.column(c.x, c.y, grid.y_lo, grid.y_hi).matter;
        }
    }
    out
}

/// Share of the really empty area that the map knows as free.
pub fn coverage(grid: &OccupancyGrid, empty: &[bool]) -> f32 {
    let (nx, nz) = grid.size();
    let (mut total, mut known) = (0usize, 0usize);
    for j in 0..nz {
        for i in 0..nx {
            if empty[j * nx + i] {
                total += 1;
                known += (grid.cell(i, j) == Cell::Free) as usize;
            }
        }
    }
    known as f32 / total.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use world::WorldSpec;

    fn corridor() -> World {
        World::build(&WorldSpec::from_ron(include_str!("../../../maps/corridor.ron")).unwrap()).unwrap()
    }

    #[test]
    fn modes_add_or_remove_the_gps() {
        let scout = DroneSpec::scout();
        let gps = |v: &[SensorSpec]| v.iter().filter(|s| matches!(s.kind, SensorKind::Gps { .. })).count();
        assert_eq!(gps(&sensors_for(&scout, PoseMode::Truth)), 0);
        assert_eq!(gps(&sensors_for(&scout, PoseMode::Ekf { gps: true })), 1);
        let pro = DroneSpec::from_ron(include_str!("../../../drones/pro.ron")).unwrap();
        assert_eq!(gps(&sensors_for(&pro, PoseMode::Ekf { gps: true })), 1);
        assert_eq!(gps(&sensors_for(&pro, PoseMode::Ekf { gps: false })), 0);
    }

    #[test]
    fn truth_mode_has_no_pose_error() {
        let mut sim = Sim::new(corridor(), &DroneSpec::scout(), SimConfig::default()).unwrap();
        for _ in 0..300 {
            sim.step(Command { forward: 0.5, ..Default::default() });
        }
        assert_eq!(sim.max_error_m, 0.0);
        assert!(sim.grid.counts().free > 100);
    }

    #[test]
    fn ekf_with_gps_maps_the_corridor_consistently() {
        let cfg = SimConfig { mode: PoseMode::Ekf { gps: true }, ..Default::default() };
        let mut sim = Sim::new(corridor(), &DroneSpec::scout(), cfg).unwrap();
        for k in 0..2000 {
            let yaw = if k % 400 < 200 { 0.3 } else { -0.3 };
            sim.step(Command { forward: 0.3, yaw, ..Default::default() });
        }
        println!("EKF + GPS: RMS error {:.2} m, max {:.2} m", sim.pose_rms(), sim.max_error_m);
        assert!(sim.pose_rms() < 0.6 && sim.max_error_m < 1.5);
        assert_eq!(sim.collisions, 0);
    }
}
