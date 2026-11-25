//! Simulated sensors.
//!
//! The AI never sees the truth of the world, only timestamped measurements with
//! their defects. Each sensor implements [`Sensor`] and runs at its own
//! rate (handled by [`SensorSuite`]). Range sensors read the world
//! through raycasts; IMU, barometer and GPS read the real drone's state.

mod depth;
mod lidar;
mod lidar3d;
mod nav;
mod sonar;

pub use depth::DepthCamera;
pub use lidar::Lidar2D;
pub use lidar3d::Lidar3D;
pub use nav::{Baro, Gps, Imu};
pub use sonar::Sonar;

use std::collections::VecDeque;

use drone_spec::{Mount, SensorKind, SensorSpec};
use glam::{EulerRot, Quat, Vec3};
use world::{Material, World, drone::DroneState, rng::SplitMix64};

/// Pose and motion of the drone (ground truth), from which the pose of each
/// sensor and what the navigation sensors measure are derived.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub position: Vec3,
    /// Heading only: the frame of the stabilized sensors.
    pub rotation: Quat,
    /// Full attitude of the airframe (heading and tilt): the frame of the sensors
    /// fixed to it.
    pub attitude: Quat,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    /// Yaw rate, in rad/s.
    pub yaw_rate: f32,
    /// Angular velocity of the airframe, world frame, in rad/s.
    pub angular_velocity: Vec3,
}

impl Pose {
    /// Stationary pose.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            attitude: rotation,
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            yaw_rate: 0.0,
            angular_velocity: Vec3::ZERO,
        }
    }
}

impl From<&DroneState> for Pose {
    fn from(s: &DroneState) -> Self {
        Self {
            position: s.position,
            rotation: s.rotation(),
            attitude: s.attitude(),
            velocity: s.velocity,
            acceleration: s.acceleration,
            yaw_rate: s.yaw_rate,
            angular_velocity: s.angular_velocity,
        }
    }
}

/// Measurement defects, tunable at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Noise {
    /// Base standard deviation, in m.
    pub std_m: f32,
    /// Growth of the standard deviation with distance: std = std_m + growth × d².
    pub growth_per_m2: f32,
    /// Probability that a measurement is lost.
    pub dropout: f32,
}

impl Noise {
    /// Noisy measurement of a true distance, or `None` if it is lost.
    pub fn apply(&self, rng: &mut SplitMix64, true_distance: f32) -> Option<f32> {
        if self.dropout > 0.0 && rng.next_f32() < self.dropout {
            return None;
        }
        let std = self.std_m + self.growth_per_m2 * true_distance * true_distance;
        Some((true_distance + std * rng.gaussian()).max(0.0))
    }
}

/// One ray of a measurement: direction in the sensor frame, measured distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beam {
    pub dir: Vec3,
    /// `None`: nothing up to max range.
    pub range: Option<f32>,
    /// `false`: invalid return (lost, or surface below the minimum range); the
    /// sensor knows it knows nothing, which is not "nothing in sight".
    pub valid: bool,
}

impl Beam {
    pub fn hit(dir: Vec3, range: f32) -> Self {
        Self { dir, range: Some(range), valid: true }
    }
    pub fn nothing(dir: Vec3) -> Self {
        Self { dir, range: None, valid: true }
    }
    pub fn invalid(dir: Vec3) -> Self {
        Self { dir, range: None, valid: false }
    }

    /// Measured ray: surface too close or lost return = invalid; noise pushing
    /// beyond max range = nothing.
    pub(crate) fn measured(dir: Vec3, hit: Option<f32>, min_range: f32, max_range: f32, noise: &Noise, rng: &mut SplitMix64) -> Self {
        match hit {
            None => Self::nothing(dir),
            Some(d) if d < min_range => Self::invalid(dir),
            Some(d) => match noise.apply(rng, d) {
                None => Self::invalid(dir),
                Some(m) if m <= max_range => Self::hit(dir, m),
                Some(_) => Self::nothing(dir),
            },
        }
    }
}

/// Casts a fan of rays (directions in the sensor frame) and measures each one.
/// `pose(i)` is the sensor pose when ray `i` is fired: fixed for a camera, moving
/// along the sweep of a spinning lidar.
///
/// Each ray draws its noise from its own generator, seeded from one draw of the
/// sensor's: the rays can be cast in parallel and the result stays the same,
/// whatever the order in which they run.
pub(crate) fn cast_beams(
    world: &World,
    pose: impl Fn(usize) -> (Vec3, Quat) + Sync,
    dirs: &[Vec3],
    (min_range, max_range): (f32, f32),
    noise: &Noise,
    rng: &mut SplitMix64,
) -> Vec<Beam> {
    let base = rng.next_u64();
    let one = |(i, &dir): (usize, &Vec3)| {
        let mut rng = SplitMix64::new(base ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let (origin, rotation) = pose(i);
        match optical_return(world, origin, rotation * dir, max_range, &mut rng) {
            Return::Nothing => Beam::nothing(dir),
            Return::Lost => Beam::invalid(dir),
            Return::Hit { distance, noise_scale } => {
                let noise = Noise { std_m: noise.std_m * noise_scale, ..*noise };
                Beam::measured(dir, Some(distance), min_range, max_range, &noise, &mut rng)
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        dirs.par_iter().enumerate().with_min_len(64).map(one).collect()
    }
    #[cfg(not(feature = "parallel"))]
    dirs.iter().enumerate().map(one).collect()
}

#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    /// Fan of rays (2D lidar).
    Scan { beams: Vec<Beam>, max_range: f32 },
    /// A single distance within a cone (sonar).
    Echo {
        range: Option<f32>,
        cone_half_angle: f32,
        max_range: f32,
    },
    /// Depth image, row by row from the top left.
    DepthImage {
        width: u32,
        height: u32,
        beams: Vec<Beam>,
        max_range: f32,
    },
    /// Acceleration in the drone frame (m/s²) and yaw rate (rad/s).
    Imu { accel: Vec3, yaw_rate: f32 },
    /// Absolute height, in m.
    Baro { height: f32 },
    /// Absolute position, in m, and its announced accuracy (1σ per axis, in m),
    /// which degrades when buildings mask the sky. Infinite: no fix.
    Gps { position: Vec3, std: f32 },
}

/// What a light ray (lidar, depth camera) brings back, before the noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Return {
    Nothing,
    /// No usable return (weak or scattered signal).
    Lost,
    Hit { distance: f32, noise_scale: f32 },
}

/// Fraction of the light rays that go through glass.
const GLASS_PASS: f32 = 0.7;
/// Dark surfaces only return within this share of the max range...
const DARK_RANGE: f32 = 0.35;
/// ...and lose this share of their returns even there.
const DARK_LOSS: f32 = 0.2;
/// Metal: noise multiplier, and share of phantom echoes (bounced away and back).
const METAL_NOISE: f32 = 3.0;
const METAL_PHANTOM: f32 = 0.05;
/// Foliage: returns come from up to this depth inside the canopy, in m.
const FOLIAGE_DEPTH: f32 = 1.0;

/// Follows a light ray through the world, surface by surface, as its material
/// decides: through glass, lost on dark paint, echoing off metal, scattered in leaves.
pub(crate) fn optical_return(world: &World, origin: Vec3, dir: Vec3, max_range: f32, rng: &mut SplitMix64) -> Return {
    let mut start = 0.0;
    let mut in_pane = false;
    // A few panes of glass at most, crossed 5 cm at a time.
    for _ in 0..32 {
        let Some(hit) = world.raycast(origin + dir * start, dir, max_range - start) else { return Return::Nothing };
        let d = start + hit.distance;
        let found = |distance: f32, noise_scale: f32| Return::Hit { distance, noise_scale };
        // Still inside the pane it went through (a hit right at the start): no new
        // chance to reflect.
        if in_pane && hit.material == Material::Glass && hit.distance < 1e-4 {
            start = d + 0.05;
            continue;
        }
        return match hit.material {
            Material::Concrete => found(d, 1.0),
            Material::Glass if rng.next_f32() < GLASS_PASS => {
                start = d + 0.02;
                in_pane = true;
                continue;
            }
            Material::Glass => found(d, 1.0),
            Material::Dark if d > DARK_RANGE * max_range || rng.next_f32() < DARK_LOSS => Return::Lost,
            Material::Dark => found(d, 1.0),
            Material::Metal if rng.next_f32() < METAL_PHANTOM => {
                let phantom = d * rng.range(1.2, 2.0);
                if phantom <= max_range { found(phantom, METAL_NOISE) } else { Return::Nothing }
            }
            Material::Metal => found(d, METAL_NOISE),
            Material::Foliage => found(d + rng.range(0.0, FOLIAGE_DEPTH), 1.0),
        };
    }
    Return::Lost
}

/// Pose of a sensor at a fraction `f` of its sweep, from its pose at the start of
/// the sweep to the one at the end.
pub(crate) fn sweep_pose(start: (Vec3, Quat), end: (Vec3, Quat), f: f32) -> (Vec3, Quat) {
    (start.0.lerp(end.0, f), start.1.slerp(end.1, f))
}

/// Timestamped measurement, with the sensor pose at measurement time (for a
/// spinning lidar, at the end of its sweep).
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Simulation time, in s.
    pub t: f64,
    pub origin: Vec3,
    pub rotation: Quat,
    /// Sensor pose at the start of the sweep of a spinning lidar; `None` when all
    /// the rays are fired at once (or the motion during the sweep is unknown).
    pub sweep_start: Option<(Vec3, Quat)>,
    pub data: Data,
}

impl Measurement {
    /// Sensor pose when ray `i` was fired: along the sweep, column by column, for a
    /// spinning lidar. Using it instead of (`origin`, `rotation`) undoes the
    /// distortion of the sweep ("deskew").
    pub fn beam_pose(&self, i: usize) -> (Vec3, Quat) {
        let end = (self.origin, self.rotation);
        let Some(start) = self.sweep_start else { return end };
        let f = match &self.data {
            Data::Scan { beams, .. } => (i as f32 + 0.5) / beams.len() as f32,
            Data::DepthImage { width, .. } => ((i % *width as usize) as f32 + 0.5) / *width as f32,
            _ => 1.0,
        };
        sweep_pose(start, end, f)
    }

    /// The same measurement seen from another drone pose: `truth` is the pose where it
    /// was taken, `estimate` the one the drone believes it has. The map, which only knows
    /// the estimate, thus places the rays where the drone believes they start.
    pub fn reposed(&self, truth: &Pose, estimate: &Pose) -> Measurement {
        let q = estimate.rotation * truth.rotation.inverse();
        let moved = |(o, r): (Vec3, Quat)| (estimate.position + q * (o - truth.position), q * r);
        let (origin, rotation) = moved((self.origin, self.rotation));
        Measurement { t: self.t, origin, rotation, sweep_start: self.sweep_start.map(moved), data: self.data.clone() }
    }
}

pub trait Sensor: Send + Sync {
    fn label(&self) -> &str;
    fn rate_hz(&self) -> f32;
    fn mount(&self) -> &Mount;
    fn noise(&self) -> &Noise;
    fn noise_mut(&mut self) -> &mut Noise;
    /// Measures from the sensor pose (already composed with its mount); `body` is
    /// the drone's state, for navigation sensors.
    fn measure(&mut self, world: &World, body: &Pose, origin: Vec3, rotation: Quat) -> Data;
    /// A spinning sensor takes a whole period to sweep: `start` is its pose when the
    /// sweep began, `end` the current one. The rays are fired along the way, but
    /// the measurement only says `end`: a drone moving or turning during the sweep
    /// sees a distorted scan, like a real lidar. Others measure from `end`.
    /// Does it fire its rays along a sweep (see [`Sensor::measure_sweep`])?
    fn spinning(&self) -> bool {
        false
    }
    fn measure_sweep(&mut self, world: &World, body: &Pose, start: (Vec3, Quat), end: (Vec3, Quat)) -> Data {
        let _ = start;
        self.measure(world, body, end.0, end.1)
    }
}

/// Builds a sensor from its spec. `seed` makes its noise reproducible.
pub fn build(spec: &SensorSpec, seed: u64) -> Box<dyn Sensor> {
    let rng = SplitMix64::new(seed);
    match spec.kind {
        SensorKind::Lidar2D { rays, range, noise_std, dropout } => Box::new(Lidar2D::new(
            spec.mount,
            spec.rate_hz,
            rays,
            range,
            Noise { std_m: noise_std, growth_per_m2: 0.0, dropout },
            rng,
        )),
        SensorKind::Lidar3D { channels, rays, fov_up_deg, fov_down_deg, range, noise_std, dropout } => {
            Box::new(Lidar3D::new(
                spec.mount,
                spec.rate_hz,
                channels,
                rays,
                fov_up_deg,
                fov_down_deg,
                range,
                Noise { std_m: noise_std, growth_per_m2: 0.0, dropout },
                rng,
            ))
        }
        SensorKind::Sonar { cone_deg, range, noise_std, max_incidence_deg, multipath } => Box::new(
            Sonar::new(
            spec.mount,
            spec.rate_hz,
            cone_deg,
            range,
            max_incidence_deg,
            Noise { std_m: noise_std, growth_per_m2: 0.0, dropout: 0.0 },
            rng,
            )
            .with_multipath(multipath),
        ),
        SensorKind::Depth { width, height, fov_deg, range, noise_std, noise_growth, dropout } => {
            Box::new(DepthCamera::new(
                spec.mount,
                spec.rate_hz,
                width,
                height,
                fov_deg,
                range,
                Noise { std_m: noise_std, growth_per_m2: noise_growth, dropout },
                rng,
            ))
        }
        SensorKind::Imu { accel_std, accel_bias, gyro_std_deg, gyro_bias_deg } => Box::new(Imu::new(
            spec.mount,
            spec.rate_hz,
            accel_std,
            accel_bias,
            gyro_std_deg.to_radians(),
            gyro_bias_deg.to_radians(),
            rng,
        )),
        SensorKind::Baro { noise_std, drift } => Box::new(Baro::new(spec.mount, spec.rate_hz, noise_std, drift, rng)),
        SensorKind::Gps { noise_std } => Box::new(Gps::new(spec.mount, spec.rate_hz, noise_std, rng)),
    }
}

/// World pose of a sensor mounted on a drone. The mount point moves with the
/// airframe; the sensor's orientation follows the airframe, or only its heading
/// if it is stabilized.
pub fn sensor_pose(body: &Pose, mount: &Mount) -> (Vec3, Quat) {
    let local = Quat::from_euler(EulerRot::YXZ, mount.yaw_deg.to_radians(), mount.pitch_deg.to_radians(), 0.0);
    let frame = if mount.stabilized { body.rotation } else { body.attitude };
    (body.position + body.attitude * Vec3::from(mount.pos), frame * local)
}

/// All the sensors of a drone, each at its own rate and with its own latency.
pub struct SensorSuite {
    pub sensors: Vec<Box<dyn Sensor>>,
    /// Last measurement delivered by each sensor.
    pub latest: Vec<Option<Measurement>>,
    next_t: Vec<f64>,
    /// Delivery delay of each sensor, in s.
    latency: Vec<f64>,
    /// Measurements taken but not delivered yet, oldest first.
    pending: Vec<VecDeque<Measurement>>,
    /// Time and pose of each sensor at its previous measurement: where its sweep
    /// starts, if that measurement was the previous period's.
    last_pose: Vec<Option<(f64, (Vec3, Quat))>>,
}

impl SensorSuite {
    /// `seed`: the drone's seed; each sensor derives its own from it.
    pub fn new(specs: &[SensorSpec], seed: u64) -> Self {
        let sensors: Vec<_> = specs
            .iter()
            .enumerate()
            .map(|(i, s)| build(s, seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
            .collect();
        let n = sensors.len();
        Self {
            sensors,
            latest: vec![None; n],
            next_t: vec![0.0; n],
            latency: specs.iter().map(|s| s.latency_s.max(0.0) as f64).collect(),
            pending: vec![VecDeque::new(); n],
            last_pose: vec![None; n],
        }
    }

    /// Triggers the sensors that are due, and delivers the measurements whose
    /// latency has elapsed. Returns the number of measurements delivered.
    pub fn update(&mut self, world: &World, body: &Pose, t: f64) -> usize {
        let mut fresh = 0;
        for (i, sensor) in self.sensors.iter_mut().enumerate() {
            let period = 1.0 / sensor.rate_hz() as f64;
            // Tolerance of 1 % of the period: a clock that is a hair early (float
            // rounding) must not skip a whole step.
            if t + period * 0.01 >= self.next_t[i] {
                // Rate kept without drift; if we fell behind, restart from t.
                self.next_t[i] = (self.next_t[i] + period).max(t + period * 0.5);
                let end = sensor_pose(body, sensor.mount());
                // A gap (first measurement, teleport, paused clock): no motion known
                // during the sweep.
                let start = match self.last_pose[i] {
                    Some((t0, p)) if t - t0 <= 1.5 * period => p,
                    _ => end,
                };
                self.last_pose[i] = Some((t, end));
                let data = sensor.measure_sweep(world, body, start, end);
                let sweep_start = sensor.spinning().then_some(start);
                self.pending[i].push_back(Measurement { t, origin: end.0, rotation: end.1, sweep_start, data });
            }
            // Same tolerance for the delivery: a zero latency delivers at once.
            while self.pending[i].front().is_some_and(|m| m.t + self.latency[i] <= t + period * 0.01) {
                self.latest[i] = self.pending[i].pop_front();
                fresh += 1;
            }
        }
        fresh
    }

    /// Delivery delay of sensor `i`, in s.
    pub fn latency(&self, i: usize) -> f64 {
        self.latency[i]
    }

    /// Forgets the measurements in flight and the sweep starts (after a teleport or
    /// a world rebuild).
    pub fn reset(&mut self) {
        self.latest.iter_mut().for_each(|m| *m = None);
        self.pending.iter_mut().for_each(VecDeque::clear);
        self.last_pose.iter_mut().for_each(|p| *p = None);
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use world::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainSpec, World, WorldSpec};

    /// Flat ground at 0 m and a 2 m thick wall whose front face is 10 m ahead (-Z).
    pub fn wall_world() -> World {
        World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::List(vec![ObstacleSpec {
                shape: ObstacleShape::Box { size: glam::Vec3::new(40.0, 10.0, 2.0) },
                x: 0.0,
                z: -11.0,
                yaw_deg: 0.0,
                elevation_m: 0.0,
                material: Default::default(),
            }]),
            wind: Default::default(),
        })
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lidar_spec(rate_hz: f32) -> SensorSpec {
        SensorSpec {
            kind: SensorKind::Lidar2D { rays: 36, range: 12.0, noise_std: 0.02, dropout: 0.0 },
            mount: Mount::default(),
            rate_hz,
            latency_s: 0.0,
        }
    }

    #[test]
    fn suite_respects_rates() {
        let w = test_util::wall_world();
        let sonar = SensorSpec {
            kind: SensorKind::Sonar { cone_deg: 20.0, range: 4.0, noise_std: 0.0, max_incidence_deg: 60.0, multipath: 0.0 },
            mount: Mount::default(),
            rate_hz: 20.0,
            latency_s: 0.0,
        };
        let mut suite = SensorSuite::new(&[lidar_spec(10.0), sonar], 1);
        let body = Pose::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY);
        let mut counts = [0usize; 2];
        let dt = 0.01;
        for k in 0..100 {
            let before: Vec<_> = suite.latest.iter().map(|m| m.as_ref().map(|m| m.t)).collect();
            suite.update(&w, &body, k as f64 * dt);
            for i in 0..2 {
                if suite.latest[i].as_ref().map(|m| m.t) != before[i] {
                    counts[i] += 1;
                }
            }
        }
        assert_eq!(counts, [10, 20]);
    }

    #[test]
    fn same_seed_same_noise() {
        let w = test_util::wall_world();
        let body = Pose::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY);
        let run = |seed| {
            let mut s = SensorSuite::new(&[lidar_spec(10.0)], seed);
            s.update(&w, &body, 0.0);
            s.latest[0].clone().unwrap().data
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn mount_pitch_points_sensor_down() {
        let body = Pose::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(1.0));
        let mount = Mount { pos: [0.0, -0.1, 0.0], yaw_deg: 0.0, pitch_deg: -90.0, stabilized: true };
        let (origin, rot) = sensor_pose(&body, &mount);
        assert!((origin - Vec3::new(1.0, 1.9, 3.0)).length() < 1e-5);
        assert!((rot * Vec3::NEG_Z - Vec3::NEG_Y).length() < 1e-5);
    }

    #[test]
    fn reposed_measurement_follows_the_estimated_pose() {
        let truth = Pose::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(0.3));
        let est = Pose::new(Vec3::new(1.5, 2.0, 2.0), Quat::from_rotation_y(0.5));
        let mount = Mount { pos: [0.0, -0.3, -0.2], yaw_deg: 10.0, pitch_deg: 0.0, stabilized: true };
        let (o, r) = sensor_pose(&truth, &mount);
        let m = Measurement { t: 0.0, origin: o, rotation: r, sweep_start: None, data: Data::Baro { height: 0.0 } };
        let (o2, r2) = sensor_pose(&est, &mount);
        let moved = m.reposed(&truth, &est);
        assert!((moved.origin - o2).length() < 1e-5);
        assert!(moved.rotation.angle_between(r2) < 1e-4);
    }

    #[test]
    fn noise_statistics_match_settings() {
        let mut rng = SplitMix64::new(3);
        let n = Noise { std_m: 0.1, growth_per_m2: 0.0, dropout: 0.25 };
        let samples: Vec<_> = (0..20_000).map(|_| n.apply(&mut rng, 5.0)).collect();
        let kept: Vec<f32> = samples.iter().flatten().copied().collect();
        let lost = 1.0 - kept.len() as f32 / samples.len() as f32;
        assert!((lost - 0.25).abs() < 0.02, "losses {lost}");
        let mean = kept.iter().sum::<f32>() / kept.len() as f32;
        let std = (kept.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / kept.len() as f32).sqrt();
        assert!((mean - 5.0).abs() < 0.01 && (std - 0.1).abs() < 0.01, "{mean} {std}");
    }

    /// Front ray of a 2D lidar (beam 0 looks forward, -Z).
    fn front_range(m: &Measurement) -> Option<f32> {
        let Data::Scan { beams, .. } = &m.data else { panic!() };
        beams[0].range
    }

    #[test]
    fn fixed_lidar_tilts_with_the_airframe() {
        let w = test_util::wall_world();
        // Nose down by 30°, 2 m above the ground, 10 m from the wall.
        let pitch = Quat::from_rotation_x(-30f32.to_radians());
        let body = Pose { attitude: pitch, ..Pose::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY) };
        let run = |stabilized: bool| {
            let mut spec = lidar_spec(10.0);
            spec.kind = SensorKind::Lidar2D { rays: 36, range: 12.0, noise_std: 0.0, dropout: 0.0 };
            spec.mount.stabilized = stabilized;
            let mut suite = SensorSuite::new(&[spec], 1);
            suite.update(&w, &body, 0.0);
            front_range(suite.latest[0].as_ref().unwrap()).unwrap()
        };
        // On a gimbal: level, it sees the wall.
        assert!((run(true) - 10.0).abs() < 1e-3);
        // Fixed: it looks down and hits the ground 4 m ahead (2 m / sin 30°).
        assert!((run(false) - 4.0).abs() < 1e-3, "{}", run(false));
    }

    #[test]
    fn latency_delays_delivery_not_the_timestamp() {
        let w = test_util::wall_world();
        let mut spec = lidar_spec(10.0);
        spec.latency_s = 0.25;
        let mut suite = SensorSuite::new(&[spec], 1);
        let body = Pose::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY);
        let mut delivered = Vec::new();
        for k in 0..60 {
            let t = k as f64 * 0.01;
            if suite.update(&w, &body, t) > 0 {
                delivered.push((t, suite.latest[0].as_ref().unwrap().t));
            }
        }
        // Taken at 0, 0.1, 0.2, ...; each one arrives 0.25 s later.
        assert_eq!(delivered.len(), 4, "{delivered:?}");
        for (k, (at, taken)) in delivered.iter().enumerate() {
            assert!((taken - k as f64 * 0.1).abs() < 1e-9 && (at - taken - 0.25).abs() < 1e-6, "{delivered:?}");
        }
    }

    #[test]
    fn a_sweep_while_turning_is_distorted_and_deskew_undoes_it() {
        let w = test_util::wall_world();
        let mut spec = lidar_spec(10.0);
        spec.kind = SensorKind::Lidar2D { rays: 360, range: 30.0, noise_std: 0.0, dropout: 0.0 };
        let mut suite = SensorSuite::new(&[spec], 1);
        // Turning at 2 rad/s: 0.2 rad during one sweep.
        let pose = |t: f64| Pose::new(Vec3::new(0.0, 2.0, 0.0), Quat::from_rotation_y(2.0 * t as f32));
        suite.update(&w, &pose(0.0), 0.0);
        suite.update(&w, &pose(0.1), 0.1);
        let m = suite.latest[0].clone().unwrap();
        assert!(m.sweep_start.is_some());
        let Data::Scan { beams, .. } = &m.data else { panic!() };
        // Hit points on the wall face (z = -10), placed with the end pose only, then
        // with each ray's own pose.
        let (mut raw, mut deskewed) = (0f32, 0f32);
        for (i, b) in beams.iter().enumerate() {
            let Some(r) = b.range else { continue };
            let p = m.origin + m.rotation * b.dir * r;
            if (p.z + 10.0).abs() > 1.5 || p.y < 1.0 {
                continue;
            }
            raw = raw.max((p.z + 10.0).abs());
            let (o, q) = m.beam_pose(i);
            deskewed = deskewed.max(((o + q * b.dir * r).z + 10.0).abs());
        }
        assert!(raw > 0.1, "no distortion: {raw}");
        assert!(deskewed < 1e-3, "deskewed off the wall by {deskewed}");
    }

    /// A wall 10 m ahead (-Z) made of `material`, a concrete one behind it at 20 m.
    fn material_world(material: Material) -> World {
        use world::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainSpec, WorldSpec};
        let wall = |z: f32, material| ObstacleSpec {
            shape: ObstacleShape::Box { size: Vec3::new(40.0, 10.0, 0.2) },
            x: 0.0,
            z,
            yaw_deg: 0.0,
            elevation_m: 0.0,
            material,
        };
        World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::List(vec![wall(-10.1, material), wall(-20.1, Material::Concrete)]),
            wind: Default::default(),
        })
        .unwrap()
    }

    /// What 2000 straight-ahead rays from 2 m up bring back.
    fn returns(material: Material, range: f32) -> Vec<Return> {
        let w = material_world(material);
        let mut rng = SplitMix64::new(7);
        (0..2000).map(|_| optical_return(&w, Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Z, range, &mut rng)).collect()
    }

    fn share(r: &[Return], f: impl Fn(&Return) -> bool) -> f32 {
        r.iter().filter(|x| f(x)).count() as f32 / r.len() as f32
    }

    #[test]
    fn glass_lets_most_rays_through() {
        let r = returns(Material::Glass, 30.0);
        let at = |d: f32| move |x: &Return| matches!(x, Return::Hit { distance, .. } if (distance - d).abs() < 0.01);
        assert!((share(&r, at(10.0)) - (1.0 - GLASS_PASS)).abs() < 0.04);
        assert!((share(&r, at(20.0)) - GLASS_PASS).abs() < 0.04);
    }

    #[test]
    fn dark_surfaces_only_answer_up_close() {
        // 10 m is beyond 35 % of a 25 m range: always lost.
        assert_eq!(share(&returns(Material::Dark, 25.0), |x| matches!(x, Return::Lost)), 1.0);
        // Within reach (40 m): only some losses.
        let lost = share(&returns(Material::Dark, 40.0), |x| matches!(x, Return::Lost));
        assert!((lost - DARK_LOSS).abs() < 0.04, "{lost}");
    }

    #[test]
    fn metal_is_noisier_and_sometimes_echoes_behind() {
        let r = returns(Material::Metal, 30.0);
        let phantom = share(&r, |x| matches!(x, Return::Hit { distance, .. } if *distance > 10.5));
        assert!((phantom - METAL_PHANTOM).abs() < 0.02, "{phantom}");
        assert!(r.iter().all(|x| matches!(x, Return::Hit { noise_scale, .. } if *noise_scale == METAL_NOISE)));
    }

    #[test]
    fn foliage_returns_from_inside_the_canopy() {
        let r = returns(Material::Foliage, 30.0);
        assert!(r.iter().all(|x| matches!(x, Return::Hit { distance, .. } if (10.0..=10.0 + FOLIAGE_DEPTH).contains(distance))));
        let deep = share(&r, |x| matches!(x, Return::Hit { distance, .. } if *distance > 10.5));
        assert!((deep - 0.5).abs() < 0.05, "{deep}");
    }
}
