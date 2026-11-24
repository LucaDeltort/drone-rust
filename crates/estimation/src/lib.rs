//! Pose estimation: extended Kalman filter (EKF) fusing IMU, barometer
//! and, if there is one, GPS.
//!
//! State (11): position (3), velocity (3) in the world frame, yaw, accelerometer
//! bias (3, drone frame), gyro bias. The drone neither rolls nor
//! pitches in the simulation: only yaw matters.
//!
//! - Prediction on each IMU measurement: integrate the debiased acceleration, rotated
//!   into the world frame by the estimated yaw.
//! - Correction by the barometer (height), the GPS (position) and scan matching
//!   (planar pose, see [`Ekf::update_pose`]).
//!
//! Without GPS or scan matching, nothing observes the horizontal position or the yaw:
//! the estimate drifts with the IMU biases, and the filter's uncertainty says so.
//!
//! Pure Rust, no Bevy and no access to the ground truth (except the known start pose).

use drone_spec::{DroneSpec, SensorKind};
use glam::Vec3;
use nalgebra::{SMatrix, SVector};
use sensors::{Data, Measurement};
use world::drone::DroneState;

const N: usize = 11;
type Vector = SVector<f32, N>;
type Matrix = SMatrix<f32, N, N>;

// Indices in the state.
const P: usize = 0;
const V: usize = 3;
const YAW: usize = 6;
const BA: usize = 7;
const BG: usize = 10;

/// Noise model of the filter, taken from the drone's sensors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EkfConfig {
    /// Accelerometer noise, in m/s².
    pub accel_std: f32,
    /// Initial accelerometer bias (1σ), in m/s².
    pub accel_bias: f32,
    /// Gyro noise, in rad/s.
    pub gyro_std: f32,
    /// Initial gyro bias (1σ), in rad/s.
    pub gyro_bias: f32,
    /// Barometer noise, in m.
    pub baro_std: f32,
    /// GPS noise, in m.
    pub gps_std: f32,
}

/// Random walk of the biases, as a fraction of the initial bias per √s (like the sensors).
const BIAS_WALK: f32 = 0.02;

impl EkfConfig {
    /// Reads the navigation sensors of the spec. `None` without an IMU: nothing to predict with.
    pub fn from_spec(spec: &DroneSpec) -> Option<Self> {
        let mut cfg = None;
        let (mut baro_std, mut gps_std) = (1.0, 1.0);
        for s in &spec.sensors {
            match s.kind {
                SensorKind::Imu { accel_std, accel_bias, gyro_std_deg, gyro_bias_deg } => {
                    cfg = Some((accel_std, accel_bias, gyro_std_deg.to_radians(), gyro_bias_deg.to_radians()));
                }
                SensorKind::Baro { noise_std, .. } => baro_std = noise_std,
                SensorKind::Gps { noise_std } => gps_std = noise_std,
                _ => {}
            }
        }
        let (accel_std, accel_bias, gyro_std, gyro_bias) = cfg?;
        Some(Self { accel_std, accel_bias, gyro_std, gyro_bias, baro_std, gps_std })
    }
}

pub struct Ekf {
    pub cfg: EkfConfig,
    x: Vector,
    p: Matrix,
    /// Last IMU measurement (debiased, drone frame) and its time, for the exposed state.
    last_accel: Vec3,
    last_yaw_rate: f32,
    last_imu_t: Option<f64>,
}

impl Ekf {
    /// Known start (point A) at time `t0`: the initial pose is certain, the biases
    /// are not.
    pub fn new(cfg: EkfConfig, start: &DroneState, t0: f64) -> Self {
        let mut x = Vector::zeros();
        for k in 0..3 {
            x[P + k] = start.position[k];
            x[V + k] = start.velocity[k];
        }
        x[YAW] = start.yaw;
        let mut p = Matrix::zeros();
        for k in 0..3 {
            p[(P + k, P + k)] = 0.01f32.powi(2);
            p[(V + k, V + k)] = 0.01f32.powi(2);
            p[(BA + k, BA + k)] = cfg.accel_bias.max(1e-4).powi(2);
        }
        p[(YAW, YAW)] = 0.5f32.to_radians().powi(2);
        p[(BG, BG)] = cfg.gyro_bias.max(1e-5).powi(2);
        Self { cfg, x, p, last_accel: Vec3::ZERO, last_yaw_rate: 0.0, last_imu_t: Some(t0) }
    }

    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x[P], self.x[P + 1], self.x[P + 2])
    }

    pub fn velocity(&self) -> Vec3 {
        Vec3::new(self.x[V], self.x[V + 1], self.x[V + 2])
    }

    pub fn yaw(&self) -> f32 {
        self.x[YAW]
    }

    /// Horizontal uncertainty (1σ), in m.
    pub fn horizontal_std(&self) -> f32 {
        (self.p[(P, P)] + self.p[(P + 2, P + 2)]).max(0.0).sqrt()
    }

    /// Yaw uncertainty (1σ), in rad.
    pub fn yaw_std(&self) -> f32 {
        self.p[(YAW, YAW)].max(0.0).sqrt()
    }

    /// Estimated biases (accelerometer, gyro).
    pub fn biases(&self) -> (Vec3, f32) {
        (Vec3::new(self.x[BA], self.x[BA + 1], self.x[BA + 2]), self.x[BG])
    }

    /// The state as the drone believes it, for the map and the planner.
    pub fn state(&self) -> DroneState {
        DroneState {
            position: self.position(),
            velocity: self.velocity(),
            yaw: self.yaw(),
            acceleration: rot_y(self.yaw(), self.last_accel),
            yaw_rate: self.last_yaw_rate,
            angular_velocity: Vec3::Y * self.last_yaw_rate,
            flight: Default::default(),
        }
    }

    /// Integrates a navigation sensor measurement; ignores the others.
    pub fn feed(&mut self, m: &Measurement) {
        match m.data {
            Data::Imu { accel, yaw_rate } => {
                if let Some(t0) = self.last_imu_t {
                    let dt = (m.t - t0) as f32;
                    if dt > 0.0 {
                        self.predict(accel, yaw_rate, dt);
                    }
                }
                self.last_imu_t = Some(m.t);
            }
            Data::Baro { height } => self.update_height(height),
            Data::Gps { position, std } if std.is_finite() => self.update_position(position, std),
            _ => {}
        }
    }

    /// Prediction: same scheme as the simulation (yaw, then velocity, then position).
    pub fn predict(&mut self, accel_body: Vec3, yaw_rate: f32, dt: f32) {
        let omega = yaw_rate - self.x[BG];
        let yaw = wrap(self.x[YAW] + omega * dt);
        self.x[YAW] = yaw;
        let (s, c) = yaw.sin_cos();
        let ba = Vec3::new(self.x[BA], self.x[BA + 1], self.x[BA + 2]);
        let a_b = accel_body - ba;
        let a_w = rot_y(yaw, a_b);
        // Derivative of the yaw rotation, applied to a_b.
        let d_yaw = Vec3::new(-a_b.x * s + a_b.z * c, 0.0, -a_b.x * c - a_b.z * s);
        let rot = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];

        for k in 0..3 {
            self.x[V + k] += a_w[k] * dt;
            self.x[P + k] += self.x[V + k] * dt;
        }
        self.last_accel = a_b;
        self.last_yaw_rate = omega;

        // Jacobian. The yaw used is the new one: it also depends on the gyro
        // bias (∂yaw/∂bg = -dt).
        let mut f = Matrix::identity();
        let dt2 = dt * dt;
        for k in 0..3 {
            f[(P + k, V + k)] = dt;
            f[(V + k, YAW)] = d_yaw[k] * dt;
            f[(P + k, YAW)] = d_yaw[k] * dt2;
            f[(V + k, BG)] = -d_yaw[k] * dt * dt;
            f[(P + k, BG)] = -d_yaw[k] * dt2 * dt;
            for j in 0..3 {
                f[(V + k, BA + j)] = -rot[k][j] * dt;
                f[(P + k, BA + j)] = -rot[k][j] * dt2;
            }
        }
        f[(YAW, BG)] = -dt;

        // Noise: white acceleration (propagated to velocity and position), gyro,
        // bias walk, and a floor for model errors.
        let mut q = Matrix::zeros();
        let sa2 = self.cfg.accel_std.powi(2) + 0.01;
        for k in 0..3 {
            q[(P + k, P + k)] = sa2 * dt2 * dt2;
            q[(P + k, V + k)] = sa2 * dt2 * dt;
            q[(V + k, P + k)] = sa2 * dt2 * dt;
            q[(V + k, V + k)] = sa2 * dt2;
            q[(BA + k, BA + k)] = (self.cfg.accel_bias * BIAS_WALK).powi(2) * dt;
        }
        q[(YAW, YAW)] = self.cfg.gyro_std.powi(2) * dt2;
        q[(BG, BG)] = (self.cfg.gyro_bias * BIAS_WALK).powi(2) * dt;

        self.p = f * self.p * f.transpose() + q;
    }

    /// Correction by an absolute height (barometer).
    pub fn update_height(&mut self, height: f32) {
        let mut h = SMatrix::<f32, 1, N>::zeros();
        h[(0, P + 1)] = 1.0;
        let r = SMatrix::<f32, 1, 1>::new(self.cfg.baro_std.powi(2));
        let z = SVector::<f32, 1>::new(height - self.x[P + 1]);
        self.correct(h, r, z);
    }

    /// Correction by an absolute position (GPS) of accuracy `std` (1σ per axis, m).
    /// The filter trusts it no more than the GPS's configured noise.
    pub fn update_position(&mut self, position: Vec3, std: f32) {
        let mut h = SMatrix::<f32, 3, N>::zeros();
        for k in 0..3 {
            h[(k, P + k)] = 1.0;
        }
        let r = SMatrix::<f32, 3, 3>::identity() * std.max(self.cfg.gps_std).powi(2);
        let z = SVector::<f32, 3>::from_fn(|k, _| position[k] - self.x[P + k]);
        self.correct(h, r, z);
    }

    /// Correction by a planar pose (x, z, yaw) with its covariance, e.g. from scan
    /// matching. Rejected (returns `false`) if it is inconsistent with the current
    /// estimate (Mahalanobis gate).
    pub fn update_pose(&mut self, x: f32, z: f32, yaw: f32, covariance: [[f32; 3]; 3]) -> bool {
        let mut h = SMatrix::<f32, 3, N>::zeros();
        h[(0, P)] = 1.0;
        h[(1, P + 2)] = 1.0;
        h[(2, YAW)] = 1.0;
        // Floor: the match is never better than this, whatever its Hessian says.
        let floor = [0.02f32.powi(2), 0.02f32.powi(2), 0.3f32.to_radians().powi(2)];
        let r = SMatrix::<f32, 3, 3>::from_fn(|i, j| covariance[i][j] + if i == j { floor[i] } else { 0.0 });
        let z = SVector::<f32, 3>::new(x - self.x[P], z - self.x[P + 2], wrap(yaw - self.x[YAW]));
        let s = h * self.p * h.transpose() + r;
        let Some(s_inv) = s.try_inverse() else { return false };
        // 99.9 % of a χ² with 3 degrees of freedom.
        if (z.transpose() * s_inv * z)[(0, 0)] > 16.3 {
            return false;
        }
        self.correct(h, r, z);
        true
    }

    fn correct<const M: usize>(
        &mut self,
        h: SMatrix<f32, M, N>,
        r: SMatrix<f32, M, M>,
        innovation: SVector<f32, M>,
    ) {
        let s = h * self.p * h.transpose() + r;
        let Some(s_inv) = s.try_inverse() else { return };
        let k = self.p * h.transpose() * s_inv;
        self.x += k * innovation;
        self.x[YAW] = wrap(self.x[YAW]);
        // Joseph form: stays symmetric and positive despite f32 rounding.
        let i_kh = Matrix::identity() - k * h;
        self.p = i_kh * self.p * i_kh.transpose() + k * r * k.transpose();
    }
}

/// Rotation of a vector from the drone frame to the world by the yaw.
fn rot_y(yaw: f32, v: Vec3) -> Vec3 {
    let (s, c) = yaw.sin_cos();
    Vec3::new(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)
}

fn wrap(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let a = a.rem_euclid(TAU);
    if a > PI { a - TAU } else { a }
}

#[cfg(test)]
mod tests;
