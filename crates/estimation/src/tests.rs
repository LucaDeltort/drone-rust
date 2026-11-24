use super::*;
use drone_spec::{Mount, SensorSpec};
use glam::Quat;
use sensors::{Pose, SensorSuite};
use world::drone::Command;
use world::{ObstaclesSpec, TerrainSpec, World, WorldSpec};

fn flat() -> World {
    World::build(&WorldSpec {
        size_m: 400.0,
        cell_m: 2.0,
        terrain: TerrainSpec::Flat { height: 0.0 },
        obstacles: ObstaclesSpec::None,
        wind: Default::default(),
    })
    .unwrap()
}

fn nav_sensors(imu: (f32, f32, f32, f32), gps: bool) -> Vec<SensorSpec> {
    let spec = |kind, rate_hz| SensorSpec { kind, mount: Mount::default(), rate_hz };
    let (accel_std, accel_bias, gyro_std_deg, gyro_bias_deg) = imu;
    let mut v = vec![
        spec(SensorKind::Imu { accel_std, accel_bias, gyro_std_deg, gyro_bias_deg }, 100.0),
        spec(SensorKind::Baro { noise_std: 0.1, drift: 0.0 }, 25.0),
    ];
    if gps {
        v.push(spec(SensorKind::Gps { noise_std: 1.0 }, 5.0));
    }
    v
}

/// Flight with yaw turns and jerks for `seconds`; returns, every second, the real
/// horizontal error and the announced uncertainty (1σ).
fn fly(sensors: &[SensorSpec], seconds: f32, seed: u64) -> (Vec<(f32, f32)>, Ekf) {
    const DT: f32 = 0.01;
    let w = flat();
    let spec = DroneSpec { sensors: sensors.to_vec(), ..DroneSpec::scout() };
    let mut state = DroneState::spawn(&w, 0.0, 0.0, 5.0);
    let mut ekf = Ekf::new(EkfConfig::from_spec(&spec).unwrap(), &state, 0.0);
    let mut suite = SensorSuite::new(sensors, seed);
    let mut last_t = vec![f64::NEG_INFINITY; sensors.len()];
    let mut out = Vec::new();
    let mut t = 0.0f32;
    while t < seconds {
        let phase = (t / 4.0) as i32 % 4;
        let cmd = Command {
            forward: [1.0, 0.5, 0.0, -0.5][phase as usize],
            right: [0.0, 0.5, 1.0, 0.0][phase as usize],
            up: 0.0,
            yaw: (t * 0.7).sin(),
        };
        state.step(cmd, &spec.airframe, &w, DT);
        t += DT;
        suite.update(&w, &Pose::from(&state), t as f64);
        for (k, m) in suite.latest.iter().enumerate() {
            if let Some(m) = m
                && m.t > last_t[k]
            {
                ekf.feed(m);
                last_t[k] = m.t;
            }
        }
        if (t * 100.0).round() as i32 % 100 == 0 {
            let e = ekf.position() - state.position;
            out.push((Vec2Len::xz(e), ekf.horizontal_std()));
        }
    }
    (out, ekf)
}

struct Vec2Len;
impl Vec2Len {
    fn xz(v: Vec3) -> f32 {
        (v.x * v.x + v.z * v.z).sqrt()
    }
}

#[test]
fn rotation_matches_glam() {
    for yaw in [0.0, 0.7, -2.0, 3.0] {
        let v = Vec3::new(0.3, -1.0, 2.0);
        assert!((rot_y(yaw, v) - Quat::from_rotation_y(yaw) * v).length() < 1e-5);
    }
}

/// Perfect IMU: integration reproduces the simulation (same numerical scheme).
#[test]
fn perfect_imu_dead_reckoning_is_exact() {
    let (errors, _) = fly(&nav_sensors((0.0, 0.0, 0.0, 0.0), false), 30.0, 1);
    let worst = errors.iter().map(|e| e.0).fold(0.0, f32::max);
    assert!(worst < 0.05, "max error {worst}");
}

#[test]
fn gps_keeps_the_error_bounded() {
    let (errors, ekf) = fly(&nav_sensors((0.05, 0.05, 0.2, 0.5), true), 120.0, 2);
    let rms = (errors.iter().map(|e| e.0 * e.0).sum::<f32>() / errors.len() as f32).sqrt();
    let worst = errors.iter().map(|e| e.0).fold(0.0, f32::max);
    println!("GPS: RMS error {rms:.2} m, max {worst:.2} m, final σ {:.2} m", ekf.horizontal_std());
    // The GPS has 1 m of noise per axis: the filter does clearly better than a single measurement.
    assert!(rms < 0.6 && worst < 1.5, "RMS {rms}, max {worst}");
    // The announced uncertainty has the right order (filter consistency).
    let inside = errors.iter().filter(|(e, s)| *e < 3.0 * s.max(0.05)).count();
    assert!(inside as f32 / errors.len() as f32 > 0.9, "{inside}/{}", errors.len());
}

/// Without GPS, the error drifts (IMU biases), but the filter knows it: its
/// uncertainty grows at the same pace.
#[test]
fn without_gps_the_drift_is_reported() {
    let mut drift_seen = 0.0f32;
    let mut inside = 0;
    let runs = 6usize;
    for seed in 0..runs {
        let (errors, ekf) = fly(&nav_sensors((0.05, 0.05, 0.2, 0.5), false), 30.0, 10 + seed as u64);
        let last = errors.last().unwrap().0;
        drift_seen = drift_seen.max(last);
        inside += (last < 3.0 * ekf.horizontal_std()) as usize;
        assert!(ekf.horizontal_std() > 1.0, "σ {}", ekf.horizontal_std());
    }
    println!("without GPS: max drift at 30 s {drift_seen:.1} m, {inside}/{runs} within 3σ");
    assert!(drift_seen > 1.0);
    assert!(inside >= runs - 1);
}

#[test]
fn gps_reveals_the_accelerometer_bias() {
    let sensors = nav_sensors((0.02, 0.2, 0.1, 0.0), true);
    let seed = 4;
    let (_, ekf) = fly(&sensors, 120.0, seed);
    // Real initial bias: same seed as the suite's IMU (sensor #0).
    let imu_seed = seed ^ 0x9E37_79B9_7F4A_7C15;
    let imu = sensors::Imu::new(Mount::default(), 100.0, 0.02, 0.2, 0.0, 0.0, world::rng::SplitMix64::new(imu_seed));
    let (truth, _) = imu.biases();
    let (est, _) = ekf.biases();
    println!("real bias {truth}, estimated {est}");
    // The real bias has walked since the start (≈ 0.2 × 0.02 × √120 ≈ 0.04 m/s² per axis):
    // ask for the right order of magnitude, axis by axis, and the right sign where it is clear.
    let err = est - truth;
    assert!(err.abs().max_element() < 0.1, "real {truth}, estimated {est}");
    assert!(est.x * truth.x > 0.0 && est.z * truth.z > 0.0);
}
