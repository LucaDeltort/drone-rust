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

use drone_spec::{Mount, SensorKind, SensorSpec};
use glam::{EulerRot, Quat, Vec3};
use world::{World, drone::DroneState, rng::SplitMix64};

/// Pose and motion of the drone (ground truth), from which the pose of each
/// sensor and what the navigation sensors measure are derived.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    /// Yaw rate, in rad/s.
    pub yaw_rate: f32,
}

impl Pose {
    /// Stationary pose.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation, velocity: Vec3::ZERO, acceleration: Vec3::ZERO, yaw_rate: 0.0 }
    }
}

impl From<&DroneState> for Pose {
    fn from(s: &DroneState) -> Self {
        Self {
            position: s.position,
            rotation: s.rotation(),
            velocity: s.velocity,
            acceleration: s.acceleration,
            yaw_rate: s.yaw_rate,
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
///
/// Each ray draws its noise from its own generator, seeded from one draw of the
/// sensor's: the rays can be cast in parallel and the result stays the same,
/// whatever the order in which they run.
pub(crate) fn cast_beams(
    world: &World,
    origin: Vec3,
    rotation: Quat,
    dirs: &[Vec3],
    (min_range, max_range): (f32, f32),
    noise: &Noise,
    rng: &mut SplitMix64,
) -> Vec<Beam> {
    let base = rng.next_u64();
    let one = |(i, &dir): (usize, &Vec3)| {
        let mut rng = SplitMix64::new(base ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let hit = world.raycast(origin, rotation * dir, max_range).map(|h| h.distance);
        Beam::measured(dir, hit, min_range, max_range, noise, &mut rng)
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
    /// Absolute position, in m.
    Gps { position: Vec3 },
}

/// Timestamped measurement, with the sensor pose at measurement time.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Simulation time, in s.
    pub t: f64,
    pub origin: Vec3,
    pub rotation: Quat,
    pub data: Data,
}

impl Measurement {
    /// The same measurement seen from another drone pose: `truth` is the pose where it
    /// was taken, `estimate` the one the drone believes it has. The map, which only knows
    /// the estimate, thus places the rays where the drone believes they start.
    pub fn reposed(&self, truth: &Pose, estimate: &Pose) -> Measurement {
        let q = estimate.rotation * truth.rotation.inverse();
        Measurement {
            t: self.t,
            origin: estimate.position + q * (self.origin - truth.position),
            rotation: q * self.rotation,
            data: self.data.clone(),
        }
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
        SensorKind::Sonar { cone_deg, range, noise_std, max_incidence_deg } => Box::new(Sonar::new(
            spec.mount,
            spec.rate_hz,
            cone_deg,
            range,
            max_incidence_deg,
            Noise { std_m: noise_std, growth_per_m2: 0.0, dropout: 0.0 },
            rng,
        )),
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

/// World pose of a sensor mounted on a drone.
pub fn sensor_pose(body: &Pose, mount: &Mount) -> (Vec3, Quat) {
    let local = Quat::from_euler(EulerRot::YXZ, mount.yaw_deg.to_radians(), mount.pitch_deg.to_radians(), 0.0);
    (body.position + body.rotation * Vec3::from(mount.pos), body.rotation * local)
}

/// All the sensors of a drone, each at its own rate.
pub struct SensorSuite {
    pub sensors: Vec<Box<dyn Sensor>>,
    /// Last measurement of each sensor.
    pub latest: Vec<Option<Measurement>>,
    next_t: Vec<f64>,
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
        }
    }

    /// Triggers the sensors that are due. Returns the number of new measurements.
    pub fn update(&mut self, world: &World, body: &Pose, t: f64) -> usize {
        let mut fresh = 0;
        for (i, sensor) in self.sensors.iter_mut().enumerate() {
            let period = 1.0 / sensor.rate_hz() as f64;
            // Tolerance of 1 % of the period: a clock that is a hair early (float
            // rounding) must not skip a whole step.
            if t + period * 0.01 < self.next_t[i] {
                continue;
            }
            // Rate kept without drift; if we fell behind, restart from t.
            self.next_t[i] = (self.next_t[i] + period).max(t + period * 0.5);
            let (origin, rotation) = sensor_pose(body, sensor.mount());
            let data = sensor.measure(world, body, origin, rotation);
            self.latest[i] = Some(Measurement { t, origin, rotation, data });
            fresh += 1;
        }
        fresh
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
        }
    }

    #[test]
    fn suite_respects_rates() {
        let w = test_util::wall_world();
        let sonar = SensorSpec {
            kind: SensorKind::Sonar { cone_deg: 20.0, range: 4.0, noise_std: 0.0, max_incidence_deg: 60.0 },
            mount: Mount::default(),
            rate_hz: 20.0,
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
        let mount = Mount { pos: [0.0, -0.1, 0.0], yaw_deg: 0.0, pitch_deg: -90.0 };
        let (origin, rot) = sensor_pose(&body, &mount);
        assert!((origin - Vec3::new(1.0, 1.9, 3.0)).length() < 1e-5);
        assert!((rot * Vec3::NEG_Z - Vec3::NEG_Y).length() < 1e-5);
    }

    #[test]
    fn reposed_measurement_follows_the_estimated_pose() {
        let truth = Pose::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(0.3));
        let est = Pose::new(Vec3::new(1.5, 2.0, 2.0), Quat::from_rotation_y(0.5));
        let mount = Mount { pos: [0.0, -0.3, -0.2], yaw_deg: 10.0, pitch_deg: 0.0 };
        let (o, r) = sensor_pose(&truth, &mount);
        let m = Measurement { t: 0.0, origin: o, rotation: r, data: Data::Baro { height: 0.0 } };
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
}
