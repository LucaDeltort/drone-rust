//! Navigation sensors: they read the real drone's state, not the world.
//!
//! - IMU: acceleration in the drone frame and yaw rate, with white noise and
//!   bias (drawn at start-up, then a slow random walk);
//! - barometer: absolute height, noise and slow drift;
//! - GPS: noisy absolute position.

use drone_spec::Mount;
use glam::{Quat, Vec3};
use world::{World, rng::SplitMix64};

use crate::{Data, Noise, Pose, Sensor};

/// Speed of the random walk of the IMU biases, as a fraction of the initial bias
/// per √s.
const BIAS_WALK: f32 = 0.02;

fn gaussian3(rng: &mut SplitMix64) -> Vec3 {
    Vec3::new(rng.gaussian(), rng.gaussian(), rng.gaussian())
}

pub struct Imu {
    mount: Mount,
    rate_hz: f32,
    /// `std_m`: accelerometer noise, in m/s².
    noise: Noise,
    gyro_std: f32,
    accel_bias_std: f32,
    gyro_bias_std: f32,
    accel_bias: Vec3,
    gyro_bias: f32,
    rng: SplitMix64,
}

impl Imu {
    pub fn new(
        mount: Mount,
        rate_hz: f32,
        accel_std: f32,
        accel_bias_std: f32,
        gyro_std: f32,
        gyro_bias_std: f32,
        mut rng: SplitMix64,
    ) -> Self {
        let accel_bias = gaussian3(&mut rng) * accel_bias_std;
        let gyro_bias = rng.gaussian() * gyro_bias_std;
        Self {
            mount,
            rate_hz,
            noise: Noise { std_m: accel_std, growth_per_m2: 0.0, dropout: 0.0 },
            gyro_std,
            accel_bias_std,
            gyro_bias_std,
            accel_bias,
            gyro_bias,
            rng,
        }
    }

    /// Current biases (accelerometer, gyro), for tests and display.
    pub fn biases(&self) -> (Vec3, f32) {
        (self.accel_bias, self.gyro_bias)
    }
}

impl Sensor for Imu {
    fn label(&self) -> &str {
        "IMU"
    }
    fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn mount(&self) -> &Mount {
        &self.mount
    }
    fn noise(&self) -> &Noise {
        &self.noise
    }
    fn noise_mut(&mut self) -> &mut Noise {
        &mut self.noise
    }

    fn measure(&mut self, _world: &World, body: &Pose, _origin: Vec3, _rotation: Quat) -> Data {
        let sqrt_dt = (1.0 / self.rate_hz).sqrt();
        self.accel_bias += gaussian3(&mut self.rng) * self.accel_bias_std * BIAS_WALK * sqrt_dt;
        self.gyro_bias += self.rng.gaussian() * self.gyro_bias_std * BIAS_WALK * sqrt_dt;
        let accel = body.rotation.inverse() * body.acceleration
            + self.accel_bias
            + gaussian3(&mut self.rng) * self.noise.std_m;
        let yaw_rate = body.yaw_rate + self.gyro_bias + self.rng.gaussian() * self.gyro_std;
        Data::Imu { accel, yaw_rate }
    }
}

pub struct Baro {
    mount: Mount,
    rate_hz: f32,
    noise: Noise,
    /// Drift, in m per √s.
    drift_rate: f32,
    drift: f32,
    rng: SplitMix64,
}

impl Baro {
    pub fn new(mount: Mount, rate_hz: f32, noise_std: f32, drift_rate: f32, rng: SplitMix64) -> Self {
        Self {
            mount,
            rate_hz,
            noise: Noise { std_m: noise_std, growth_per_m2: 0.0, dropout: 0.0 },
            drift_rate,
            drift: 0.0,
            rng,
        }
    }
}

impl Sensor for Baro {
    fn label(&self) -> &str {
        "Baro"
    }
    fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn mount(&self) -> &Mount {
        &self.mount
    }
    fn noise(&self) -> &Noise {
        &self.noise
    }
    fn noise_mut(&mut self) -> &mut Noise {
        &mut self.noise
    }

    fn measure(&mut self, _world: &World, _body: &Pose, origin: Vec3, _rotation: Quat) -> Data {
        self.drift += self.rng.gaussian() * self.drift_rate * (1.0 / self.rate_hz).sqrt();
        Data::Baro { height: origin.y + self.drift + self.rng.gaussian() * self.noise.std_m }
    }
}

pub struct Gps {
    mount: Mount,
    rate_hz: f32,
    noise: Noise,
    rng: SplitMix64,
}

impl Gps {
    pub fn new(mount: Mount, rate_hz: f32, noise_std: f32, rng: SplitMix64) -> Self {
        Self { mount, rate_hz, noise: Noise { std_m: noise_std, growth_per_m2: 0.0, dropout: 0.0 }, rng }
    }
}

impl Sensor for Gps {
    fn label(&self) -> &str {
        "GPS"
    }
    fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn mount(&self) -> &Mount {
        &self.mount
    }
    fn noise(&self) -> &Noise {
        &self.noise
    }
    fn noise_mut(&mut self) -> &mut Noise {
        &mut self.noise
    }

    fn measure(&mut self, _world: &World, _body: &Pose, origin: Vec3, _rotation: Quat) -> Data {
        Data::Gps { position: origin + gaussian3(&mut self.rng) * self.noise.std_m }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::wall_world;

    fn moving_body() -> Pose {
        Pose {
            position: Vec3::new(1.0, 2.0, 3.0),
            // Turned 90° to the left: the drone's nose is -X.
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            attitude: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            velocity: Vec3::ZERO,
            acceleration: Vec3::new(-2.0, 0.0, 0.0),
            yaw_rate: 0.5,
            angular_velocity: Vec3::Y * 0.5,
        }
    }

    #[test]
    fn imu_measures_body_frame_acceleration() {
        let w = wall_world();
        let mut imu = Imu::new(Mount::default(), 100.0, 0.0, 0.0, 0.0, 0.0, SplitMix64::new(1));
        let b = moving_body();
        let Data::Imu { accel, yaw_rate } = imu.measure(&w, &b, b.position, b.rotation) else { panic!() };
        // Acceleration towards world -X = towards the drone's front (-Z).
        assert!((accel - Vec3::new(0.0, 0.0, -2.0)).length() < 1e-5, "{accel}");
        assert!((yaw_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn imu_noise_and_bias_have_the_requested_size() {
        let w = wall_world();
        let b = moving_body();
        let mut biases = Vec::new();
        for seed in 0..400 {
            let imu = Imu::new(Mount::default(), 100.0, 0.0, 0.1, 0.0, 0.01, SplitMix64::new(seed));
            biases.push(imu.biases().0.x);
        }
        let std = (biases.iter().map(|b| b * b).sum::<f32>() / biases.len() as f32).sqrt();
        assert!((std - 0.1).abs() < 0.015, "bias {std}");

        let mut imu = Imu::new(Mount::default(), 100.0, 0.2, 0.0, 0.0, 0.0, SplitMix64::new(9));
        let xs: Vec<f32> = (0..5000)
            .map(|_| match imu.measure(&w, &b, b.position, b.rotation) {
                Data::Imu { accel, .. } => accel.x,
                _ => unreachable!(),
            })
            .collect();
        let std = (xs.iter().map(|x| x * x).sum::<f32>() / xs.len() as f32).sqrt();
        assert!((std - 0.2).abs() < 0.01, "noise {std}");
    }

    #[test]
    fn baro_and_gps_center_on_the_truth() {
        let w = wall_world();
        let b = moving_body();
        let mut baro = Baro::new(Mount::default(), 25.0, 0.1, 0.0, SplitMix64::new(2));
        let mut gps = Gps::new(Mount::default(), 5.0, 0.5, SplitMix64::new(3));
        let (mut h, mut p) = (0.0, Vec3::ZERO);
        let n = 4000;
        for _ in 0..n {
            if let Data::Baro { height } = baro.measure(&w, &b, b.position, b.rotation) {
                h += height / n as f32;
            }
            if let Data::Gps { position } = gps.measure(&w, &b, b.position, b.rotation) {
                p += position / n as f32;
            }
        }
        assert!((h - 2.0).abs() < 0.01 && (p - b.position).length() < 0.05, "{h} {p}");
    }
}
