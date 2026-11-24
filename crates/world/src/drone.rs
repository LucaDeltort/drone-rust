//! Flight dynamics of the real drone (ground truth).
//!
//! A quadrotor: it only pushes along its thrust axis. To accelerate, brake or
//! hold against the wind, it tilts that axis.
//!
//! - **Flight controller** (at the caller's `dt`). The pilot command is a velocity
//!   setpoint in the drone frame. A PI loop turns the velocity error into a wanted
//!   acceleration (the integral learns the steady push of the wind), with a
//!   feed-forward of the drag in still air. The acceleration is bounded by
//!   `max_accel`, then by the tilt limit: it gives a wanted thrust axis and thrust.
//! - **Attitude, mixer and motors** (at 500 Hz, in sub-steps). P on the tilt angle
//!   gives a wanted tilt rate, P on the rate a wanted torque. An X mixer turns
//!   (thrust, roll, pitch, yaw torques) into four motor thrusts, keeping the thrust
//!   first and scaling the torques down when a motor saturates. Each motor follows
//!   its command with the `motor_tau_s` lag (its ESC overdrives it within
//!   `[0, max]`), and the torques turn the airframe through its inertia.
//! - **Forces.** Thrust (stronger near the ground: ground effect), gravity, and
//!   drag on the airspeed (velocity relative to the wind): linear in the rotor
//!   plane (induced rotor drag) plus quadratic (body).
//! - **Battery** (optional). The motors draw a power from momentum theory; the
//!   voltage falls with the charge and sags with the current, and the motors' max
//!   thrust scales with its square.
//! - **Contacts** (at the caller's `dt`). The ground and the map edges stop the
//!   drone; an obstacle pushes it back out and absorbs most of the impact.
//!
//! Yaw stays kinematic (the commanded rate is applied at once); its torque only
//! loads the motors.

use drone_spec::Airframe;
use glam::{Quat, Vec3};

use crate::World;

/// Gravity, in m/s².
pub const G: f32 = 9.81;

/// Pilot command, each axis in [-1, 1].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Command {
    /// Forward (+) / backward (-).
    pub forward: f32,
    /// Right (+) / left (-).
    pub right: f32,
    /// Up (+) / down (-).
    pub up: f32,
    /// Yaw: left (+) / right (-), counter-clockwise seen from above.
    pub yaw: f32,
}

/// Internal state of the flight dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flight {
    /// Thrust axis (the drone's "up"), unit vector in the world frame.
    pub up: Vec3,
    /// Total motor thrust per unit mass, in m/s² (before the ground effect).
    pub thrust: f32,
    /// Thrust of each motor per unit mass, in m/s². Order: front right, rear left,
    /// front left, rear right.
    pub motors: [f32; 4],
    /// Health of each motor: fraction of its max thrust still available (failure
    /// injection; 1 = healthy, 0 = dead).
    pub motor_health: [f32; 4],
    /// Tilt rate of the thrust axis, world frame, in rad/s (perpendicular to `up`).
    pub tilt_rate: Vec3,
    /// Integral of the velocity error, in m: what the controller learned about the
    /// steady disturbances (wind).
    pub integral: Vec3,
    /// Wind felt at the last step, in m/s.
    pub wind: Vec3,
    /// Thrust multiplier of the ground effect at the last step (1 = none).
    pub ground_effect: f32,
    /// Battery state of charge, in [0, 1] (stays 1 without battery).
    pub soc: f32,
    /// Battery voltage under load, in V (0 without battery).
    pub voltage: f32,
    /// Battery current, in A (0 without battery).
    pub current: f32,
    /// Electrical energy drawn since the start, in Wh.
    pub energy_wh: f32,
    /// Flight time, in s (the wind depends on it).
    pub t: f32,
    /// Was the drone touching an obstacle at the last step?
    pub contact: bool,
    /// Is the drone resting on the ground?
    pub grounded: bool,
}

impl Default for Flight {
    fn default() -> Self {
        Self {
            up: Vec3::Y,
            thrust: G,
            motors: [G / 4.0; 4],
            motor_health: [1.0; 4],
            tilt_rate: Vec3::ZERO,
            integral: Vec3::ZERO,
            wind: Vec3::ZERO,
            ground_effect: 1.0,
            soc: 1.0,
            voltage: 0.0,
            current: 0.0,
            energy_wh: 0.0,
            t: 0.0,
            contact: false,
            grounded: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneState {
    pub position: Vec3,
    pub velocity: Vec3,
    /// Yaw in radians; 0 = facing -Z.
    pub yaw: f32,
    /// Actual acceleration over the last step, in m/s² (what an IMU measures once
    /// gravity is compensated).
    pub acceleration: Vec3,
    /// Yaw rate over the last step, in rad/s.
    pub yaw_rate: f32,
    /// Angular velocity of the airframe, world frame, in rad/s: tilt rate plus yaw
    /// rate around the vertical.
    pub angular_velocity: Vec3,
    pub flight: Flight,
}

/// Velocity error beyond which the integral grows no faster, in m/s.
const INTEGRAL_ZONE: f32 = 0.5;
/// Fraction of the normal speed kept after hitting an obstacle.
const RESTITUTION: f32 = 0.2;
/// Fraction of the tangential speed kept after hitting an obstacle.
const SCRAPE: f32 = 0.7;
/// Physics step, in s: `step` splits its `dt` into sub-steps no longer than this.
pub const PHYSICS_DT: f32 = 0.002;
/// Roll/pitch inertia over m·radius² (motors at the arm tips, light body).
const INERTIA: f32 = 0.07;
/// Yaw inertia over m·radius².
const INERTIA_YAW: f32 = 0.12;
/// Yaw (drag) torque of a rotor over its thrust, per m of rotor radius.
const YAW_TORQUE: f32 = 0.2;
/// The ESC makes a motor follow its command this much faster than `motor_tau_s`,
/// as long as the command stays within `[0, max]` (it overdrives it).
const ESC_SPEEDUP: f32 = 3.0;
/// Rate loop gain over angle loop gain.
const RATE_GAIN: f32 = 4.0;
/// Share of the max thrust kept for the attitude: the collective never asks more.
const ATTITUDE_RESERVE: f32 = 0.1;
/// Ground effect cap (thrust multiplier).
const GROUND_EFFECT_MAX: f32 = 1.3;
/// Beyond this many rotor radii above a surface, no ground effect.
const GROUND_EFFECT_RANGE: f32 = 10.0;
/// Air density, in kg/m³.
const AIR_DENSITY: f32 = 1.225;
/// Electrical power over ideal (momentum theory) power: 1 / (figure of merit ×
/// motor and ESC efficiency).
const POWER_FACTOR: f32 = 2.0;
/// Motor layout, body frame (X right, -Z forward), in arm units: front right, rear
/// left, front left, rear right; then the sign of each one's yaw torque.
const MOTOR_XZ: [(f32, f32); 4] = [(1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0)];
const MOTOR_SPIN: [f32; 4] = [1.0, 1.0, -1.0, -1.0];

/// Geometry of the airframe, derived from its spec.
struct Geometry {
    /// Offset of a motor along X and Z from the center, in m.
    arm: f32,
    /// Roll/pitch inertia per unit mass, in m².
    inertia: f32,
    /// Yaw inertia per unit mass, in m².
    inertia_yaw: f32,
    /// Yaw torque over thrust, in m.
    yaw_arm: f32,
    rotor: f32,
}

impl Geometry {
    fn new(a: &Airframe) -> Self {
        let rotor = a.rotor_radius();
        Self {
            arm: (a.radius - rotor) / std::f32::consts::SQRT_2,
            inertia: INERTIA * a.radius * a.radius,
            inertia_yaw: INERTIA_YAW * a.radius * a.radius,
            yaw_arm: YAW_TORQUE * rotor,
            rotor,
        }
    }

    /// Motor thrusts for a collective `thrust` and body torques (per unit mass),
    /// within `[0, max]`: the thrust first, then roll/pitch, then yaw.
    fn mix(&self, thrust: f32, torque: Vec3, max: [f32; 4]) -> [f32; 4] {
        let ceiling: f32 = max.iter().sum::<f32>() * (1.0 - ATTITUDE_RESERVE);
        let base = [thrust.clamp(0.0, ceiling) / 4.0; 4];
        let k = 1.0 / (4.0 * self.arm);
        let tilt = MOTOR_XZ.map(|(x, z)| k * (-z * torque.x + x * torque.z));
        let yaw = MOTOR_SPIN.map(|s| s * torque.y / (4.0 * self.yaw_arm));
        let room = |base: [f32; 4], d: [f32; 4]| {
            (0..4).fold(1.0_f32, |s, i| {
                let free = if d[i] > 0.0 { max[i] - base[i] } else { base[i] };
                if d[i] == 0.0 { s } else { s.min(free.max(0.0) / d[i].abs()) }
            })
        };
        let s = room(base, tilt);
        let base = std::array::from_fn(|i| base[i] + s * tilt[i]);
        let s = room(base, yaw);
        std::array::from_fn(|i| (base[i] + s * yaw[i]).clamp(0.0, max[i]))
    }

    /// Body torques (per unit mass) of the motor thrusts.
    fn torque(&self, f: [f32; 4]) -> Vec3 {
        (0..4).fold(Vec3::ZERO, |t, i| {
            let (x, z) = MOTOR_XZ[i];
            t + Vec3::new(-z * self.arm, MOTOR_SPIN[i] * self.yaw_arm, x * self.arm) * f[i]
        })
    }
}

/// Thrust multiplier of the ground effect (Cheeseman-Bennett), rotor of radius `r`
/// at `h` above a surface.
pub fn ground_effect(r: f32, h: f32) -> f32 {
    let q = r / (4.0 * h.max(1e-3));
    if q * q >= 1.0 - 1.0 / GROUND_EFFECT_MAX { GROUND_EFFECT_MAX } else { 1.0 / (1.0 - q * q) }
}

impl DroneState {
    /// Drone placed at (x, z), `altitude` m above the ground, hovering.
    pub fn spawn(world: &World, x: f32, z: f32, altitude: f32) -> Self {
        Self {
            position: Vec3::new(x, world.terrain.height_at(x, z) + altitude, z),
            velocity: Vec3::ZERO,
            yaw: 0.0,
            acceleration: Vec3::ZERO,
            yaw_rate: 0.0,
            angular_velocity: Vec3::ZERO,
            flight: Flight::default(),
        }
    }

    /// Heading only (yaw): the frame of the stabilized sensors and of the IMU
    /// once attitude-compensated.
    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw)
    }

    /// Full attitude of the airframe: heading, then tilt of the thrust axis.
    pub fn attitude(&self) -> Quat {
        Quat::from_rotation_arc(Vec3::Y, self.flight.up) * self.rotation()
    }

    /// Tilt in the heading frame, in radians: (pitch, roll). Pitch > 0 = nose
    /// down (accelerating forward), roll > 0 = right side down.
    pub fn tilt(&self) -> (f32, f32) {
        let l = self.rotation().inverse() * self.flight.up;
        ((-l.z).atan2(l.y), l.x.atan2(l.y))
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation() * Vec3::NEG_Z
    }

    /// Height above the ground.
    pub fn altitude(&self, world: &World) -> f32 {
        self.position.y - world.terrain.height_at(self.position.x, self.position.z)
    }

    /// Advances the simulation by `dt` seconds: the flight controller once, the
    /// physics in sub-steps of at most [`PHYSICS_DT`].
    pub fn step(&mut self, cmd: Command, airframe: &Airframe, world: &World, dt: f32) {
        let clamp = |v: f32| v.clamp(-1.0, 1.0);
        let v_before = self.velocity;
        let a = airframe;
        let f = &mut self.flight;
        f.t += dt;

        let yaw_rate_before = self.yaw_rate;
        self.yaw_rate = clamp(cmd.yaw) * a.max_yaw_rate.to_radians();
        self.yaw += self.yaw_rate * dt;
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);

        // Velocity setpoint in the drone frame, clamped to the unit ball so that
        // the diagonal is not faster than an axis.
        let body = Vec3::new(clamp(cmd.right), clamp(cmd.up), -clamp(cmd.forward));
        let target = Quat::from_rotation_y(self.yaw) * body.clamp_length_max(1.0) * a.max_speed;

        // PI velocity loop. Gains scale with the airframe's authority.
        let kp = (a.max_accel / 0.75).clamp(2.0, 8.0);
        let ki = 0.25 * kp;
        let err = target - self.velocity;
        let in_plane = |v: Vec3| v - f.up * v.dot(f.up);
        let drag_still_air = a.drag * in_plane(self.velocity) + a.drag_quad * self.velocity.length() * self.velocity;
        let wanted = |integral: Vec3| {
            let mut acc = kp * err + ki * integral;
            let h = glam::Vec2::new(acc.x, acc.z).clamp_length_max(a.max_accel);
            acc = Vec3::new(h.x, acc.y.clamp(-a.max_accel, a.max_accel), h.y);
            acc + drag_still_air
        };
        // Conditional integration (anti-windup): integrate only near the setpoint
        // (the steady push of the wind, not the transients), while the output is not
        // saturated, and not into the ground.
        let mut grow = err.clamp_length_max(INTEGRAL_ZONE) * dt;
        if f.grounded && grow.y < 0.0 {
            grow.y = 0.0;
        }
        let next = f.integral + grow;
        let raw = kp * err + ki * next;
        let saturated = glam::Vec2::new(raw.x, raw.z).length() > a.max_accel || raw.y.abs() > a.max_accel;
        if !saturated {
            f.integral = next;
        }
        let mut thrust_vec = wanted(f.integral) + Vec3::Y * G;

        // Tilt limit: keep the vertical component, shorten the horizontal one.
        thrust_vec.y = thrust_vec.y.max(0.05 * G);
        let h = glam::Vec2::new(thrust_vec.x, thrust_vec.z);
        let h_max = thrust_vec.y * a.max_tilt_deg.to_radians().tan();
        if h.length() > h_max {
            let h = h.clamp_length_max(h_max);
            thrust_vec = Vec3::new(h.x, thrust_vec.y, h.y);
        }
        let up_wanted = thrust_vec.normalize();
        let thrust_wanted = thrust_vec.length().min(a.thrust_to_weight * G);

        // Held over the sub-steps: wind, ground effect, yaw torque.
        let geo = Geometry::new(a);
        f.wind = world.wind_at(self.position, f.t);
        let reach = GROUND_EFFECT_RANGE * geo.rotor;
        f.ground_effect = world
            .raycast(self.position, Vec3::NEG_Y, reach)
            .map_or(1.0, |hit| ground_effect(geo.rotor, hit.distance));
        let yaw_torque = if dt > 0.0 { geo.inertia_yaw * (self.yaw_rate - yaw_rate_before) / dt } else { 0.0 };

        let n = ((dt / PHYSICS_DT).ceil() as usize).max(1);
        let h = dt / n as f32;
        let k_angle = 1.0 / a.attitude_tau_s;
        let k_rate = RATE_GAIN * k_angle;
        let motor_lag = 1.0 - (-h / a.motor_tau_s).exp();
        let esc_gain = (1.0 - (-h * ESC_SPEEDUP / a.motor_tau_s).exp()) / motor_lag;
        let max_motor = a.thrust_to_weight * G / 4.0;
        let rotor_area = std::f32::consts::PI * geo.rotor * geo.rotor;
        let induced = 1.0 / (2.0 * AIR_DENSITY * rotor_area).sqrt();
        for _ in 0..n {
            // Battery: open-circuit voltage minus the sag of the last current.
            let sag = match a.battery {
                Some(b) => {
                    let v = b.v_empty + (b.v_full - b.v_empty) * f.soc - f.current * b.r_internal;
                    (v.max(0.0) / b.v_full).powi(2)
                }
                None => 1.0,
            };
            let max: [f32; 4] = std::array::from_fn(|i| max_motor * sag * f.motor_health[i].clamp(0.0, 1.0));

            // Attitude: P on the angle → wanted tilt rate, P on the rate → torque.
            let axis = f.up.cross(up_wanted);
            let angle = f.up.angle_between(up_wanted);
            let rate_wanted = if axis.length_squared() > 1e-12 {
                (axis.normalize() * angle * k_angle).clamp_length_max(a.tilt_rate_deg.to_radians())
            } else {
                Vec3::ZERO
            };
            let attitude = Quat::from_rotation_arc(Vec3::Y, f.up) * Quat::from_rotation_y(self.yaw);
            let torque_world = (rate_wanted - f.tilt_rate) * k_rate * geo.inertia;
            let torque_body = attitude.inverse() * torque_world;
            let command = geo.mix(thrust_wanted, Vec3::new(torque_body.x, yaw_torque, torque_body.z), max);

            // Motors: first-order lag on a command the ESC overdrives, within [0, max].
            for i in 0..4 {
                let u = (f.motors[i] + (command[i] - f.motors[i]) * esc_gain).clamp(0.0, max[i]);
                f.motors[i] += (u - f.motors[i]) * motor_lag;
            }
            f.thrust = f.motors.iter().sum();

            // Rotation: the torques turn the thrust axis (yaw is kinematic).
            let t = attitude * geo.torque(f.motors) * f.ground_effect;
            f.tilt_rate += (t - f.up * t.dot(f.up)) / geo.inertia * h;
            f.tilt_rate -= f.up * f.tilt_rate.dot(f.up);
            let turn = f.tilt_rate * h;
            if turn.length_squared() > 0.0 {
                f.up = (Quat::from_scaled_axis(turn) * f.up).normalize();
            }

            // Power: momentum theory on each rotor; the battery delivers it.
            let mass = a.mass_kg;
            let power = POWER_FACTOR * induced * f.motors.iter().map(|&m| (m * mass).max(0.0).powf(1.5)).sum::<f32>();
            let drawn = match a.battery {
                Some(b) => {
                    let ocv = b.v_empty + (b.v_full - b.v_empty) * f.soc;
                    let disc = ocv * ocv - 4.0 * power * b.r_internal;
                    f.voltage = if disc >= 0.0 { 0.5 * (ocv + disc.sqrt()) } else { 0.5 * ocv };
                    f.current = power / f.voltage.max(1e-3);
                    let joules = ocv * f.current * h;
                    f.soc = (f.soc - joules / (b.capacity_wh * 3600.0)).max(0.0);
                    joules
                }
                None => power * h,
            };
            f.energy_wh += drawn / 3600.0;

            // Forces: thrust, gravity, drag on the airspeed (linear in the rotor plane).
            let air = self.velocity - f.wind;
            let air_plane = air - f.up * air.dot(f.up);
            let drag = -a.drag * air_plane - a.drag_quad * air.length() * air;
            let accel = f.up * (f.thrust * f.ground_effect) - Vec3::Y * G + drag;
            self.velocity += accel * h;
            self.position += self.velocity * h;
        }
        self.angular_velocity = f.tilt_rate + Vec3::Y * self.yaw_rate;

        // Obstacles: push back out, absorb the impact.
        f.contact = false;
        for _ in 0..3 {
            let Some(c) = world.contact(self.position, a.radius) else { break };
            f.contact = true;
            self.position += c.normal * c.depth;
            let vn = self.velocity.dot(c.normal);
            if vn < 0.0 {
                let tangent = self.velocity - c.normal * vn;
                self.velocity = tangent * SCRAPE - c.normal * vn * RESTITUTION;
            }
        }

        // Map edges: clamp and cancel the outward velocity.
        let half = world.terrain.half_size() - a.radius;
        for axis in [0, 2] {
            let p = self.position[axis];
            if p.abs() > half {
                self.position[axis] = p.clamp(-half, half);
                self.velocity[axis] = 0.0;
            }
        }

        // Ground: the drone's sphere never goes below it; on the ground, friction.
        let floor = world.terrain.height_at(self.position.x, self.position.z) + a.radius;
        f.grounded = self.position.y <= floor + 1e-3;
        if self.position.y < floor {
            self.position.y = floor;
            self.velocity.y = self.velocity.y.max(0.0);
        }
        if f.grounded && self.velocity.y <= 0.0 {
            let keep = (1.0 - 6.0 * dt).max(0.0);
            self.velocity.x *= keep;
            self.velocity.z *= keep;
        }
        self.acceleration = (self.velocity - v_before) / dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainSpec, WindSpec, WorldSpec};

    fn world_with(obstacles: ObstaclesSpec, wind: WindSpec) -> World {
        World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles,
            wind,
        })
        .unwrap()
    }

    fn flat_world() -> World {
        world_with(ObstaclesSpec::None, WindSpec::default())
    }

    fn airframe() -> Airframe {
        drone_spec::DroneSpec::scout().airframe
    }

    const DT: f32 = 0.01;

    fn run(s: &mut DroneState, cmd: Command, w: &World, seconds: f32) {
        for _ in 0..(seconds / DT).round() as usize {
            s.step(cmd, &airframe(), w, DT);
        }
    }

    #[test]
    fn hovers_in_still_air() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        let start = s.position;
        run(&mut s, Command::default(), &w, 5.0);
        assert!((s.position - start).length() < 1e-3, "{}", s.position);
        assert!(s.flight.up.angle_between(Vec3::Y) < 1e-4);
    }

    #[test]
    fn forward_tilts_nose_down_then_cruises_at_max_speed() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        let cmd = Command { forward: 1.0, ..Default::default() };
        run(&mut s, cmd, &w, 0.3);
        let (pitch, roll) = s.tilt();
        assert!(pitch.to_degrees() > 15.0, "pitch {}", pitch.to_degrees());
        assert!(roll.abs() < 1e-3);
        run(&mut s, cmd, &w, 6.0);
        let v = s.velocity;
        assert!((v.length() - airframe().max_speed).abs() < 0.1, "{v}");
        assert!(v.z < 0.0 && v.x.abs() < 1e-3);
        // At cruise speed the drone stays leaning into its own drag.
        let (pitch, _) = s.tilt();
        assert!(pitch.to_degrees() > 5.0, "cruise pitch {}", pitch.to_degrees());
        // Altitude held.
        assert!((s.position.y - 5.0).abs() < 0.2, "{}", s.position.y);
    }

    #[test]
    fn braking_tilts_nose_up() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        run(&mut s, Command { forward: 1.0, ..Default::default() }, &w, 4.0);
        run(&mut s, Command::default(), &w, 0.3);
        assert!(s.tilt().0.to_degrees() < -15.0, "pitch {}", s.tilt().0.to_degrees());
        run(&mut s, Command::default(), &w, 4.0);
        assert!(s.velocity.length() < 0.05, "{}", s.velocity);
    }

    #[test]
    fn horizontal_acceleration_is_bounded_by_the_tilt() {
        let w = flat_world();
        let a = airframe();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        let cmd = Command { forward: 1.0, right: 1.0, ..Default::default() };
        let tilt_max = a.max_tilt_deg.to_radians();
        for _ in 0..300 {
            s.step(cmd, &a, &w, DT);
            assert!(s.flight.up.angle_between(Vec3::Y) <= tilt_max + 1e-3);
            let h = glam::Vec2::new(s.acceleration.x, s.acceleration.z).length();
            assert!(h <= a.thrust_to_weight * G * tilt_max.sin() + 1e-3, "{h}");
        }
        assert!(s.velocity.length() <= a.max_speed + 0.1);
    }

    #[test]
    fn yaw_rate_is_bounded_and_turns_left() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        run(&mut s, Command { yaw: 1.0, ..Default::default() }, &w, 0.5);
        let expected = airframe().max_yaw_rate.to_radians() * 0.5;
        assert!((s.yaw - expected).abs() < 1e-3);
        // Positive yaw seen from above: the nose turns from -Z towards -X (left).
        assert!(s.forward().x < 0.0);
    }

    #[test]
    fn holds_position_against_the_wind_by_leaning_into_it() {
        // Wind from the east (blows towards -X).
        let wind = WindSpec { speed_ms: 6.0, from_deg: 90.0, ..Default::default() };
        let w = world_with(ObstaclesSpec::None, wind);
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 10.0);
        run(&mut s, Command::default(), &w, 1.0);
        // Pushed downwind at first...
        assert!(s.position.x < -0.1, "{}", s.position);
        run(&mut s, Command::default(), &w, 15.0);
        // ...then the integral holds it still, leaning east (right side down,
        // facing north).
        assert!(s.velocity.length() < 0.05, "{}", s.velocity);
        let (_, roll) = s.tilt();
        assert!(roll.to_degrees() > 5.0, "roll {}", roll.to_degrees());
        assert!((s.flight.wind.x + 6.0).abs() < 0.1);
    }

    #[test]
    fn a_gale_beyond_the_tilt_limit_wins() {
        let wind = WindSpec { speed_ms: 40.0, from_deg: 90.0, ..Default::default() };
        let w = world_with(ObstaclesSpec::None, wind);
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 10.0);
        run(&mut s, Command::default(), &w, 1.0);
        assert!(s.velocity.x < -1.0 && s.position.x < -2.0, "{} {}", s.velocity, s.position);
    }

    #[test]
    fn never_goes_below_ground_and_lands() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 1.0);
        run(&mut s, Command { up: -1.0, ..Default::default() }, &w, 3.0);
        assert!((s.position.y - airframe().radius).abs() < 1e-5);
        assert!(s.velocity.y >= 0.0 && s.flight.grounded);
        // Takes off again without delay.
        run(&mut s, Command { up: 1.0, ..Default::default() }, &w, 1.0);
        assert!(s.position.y > 2.0, "{}", s.position.y);
    }

    #[test]
    fn stays_inside_map() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        run(&mut s, Command { right: 1.0, ..Default::default() }, &w, 20.0);
        assert!(s.position.x <= w.terrain.half_size());
        assert_eq!(s.velocity.x, 0.0);
    }

    #[test]
    fn bounces_off_a_wall_without_going_through() {
        let wall = ObstacleSpec {
            shape: ObstacleShape::Box { size: Vec3::new(20.0, 10.0, 1.0) },
            x: 0.0,
            z: -10.0,
            yaw_deg: 0.0,
            elevation_m: 0.0,
            material: Default::default(),
        };
        let w = world_with(ObstaclesSpec::List(vec![wall]), WindSpec::default());
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 3.0);
        let mut touched = false;
        for _ in 0..500 {
            s.step(Command { forward: 1.0, ..Default::default() }, &airframe(), &w, DT);
            touched |= s.flight.contact;
            // Front face at z = -9.5: the sphere's center never passes it by more than a hair.
            assert!(s.position.z > -9.5 + airframe().radius - 0.01, "{}", s.position);
        }
        assert!(touched);
    }


    fn battery_airframe() -> Airframe {
        Airframe {
            battery: Some(drone_spec::Battery { capacity_wh: 22.0, v_full: 16.8, v_empty: 13.2, r_internal: 0.05 }),
            ..airframe()
        }
    }

    #[test]
    fn hover_under_the_mixer_is_stable_with_balanced_motors() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        run(&mut s, Command::default(), &w, 10.0);
        let m = s.flight.motors;
        assert!(m.iter().all(|&f| (f - G / 4.0).abs() < 1e-3), "{m:?}");
        assert!(s.angular_velocity.length() < 1e-4, "{}", s.angular_velocity);
        assert!(s.velocity.length() < 1e-3);
    }

    #[test]
    fn large_outer_steps_stay_stable() {
        // A slow viewer frame: the physics still runs at 500 Hz.
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        for _ in 0..100 {
            s.step(Command { forward: 1.0, ..Default::default() }, &airframe(), &w, 0.05);
        }
        assert!((s.velocity.length() - airframe().max_speed).abs() < 0.2, "{}", s.velocity);
        assert!((s.position.y - 5.0).abs() < 0.3, "{}", s.position.y);
    }

    #[test]
    fn ground_effect_lowers_the_thrust_needed_near_the_ground() {
        let r = airframe().rotor_radius();
        assert!(ground_effect(r, 0.5 * r) <= GROUND_EFFECT_MAX);
        assert!(ground_effect(r, r) > ground_effect(r, 2.0 * r));
        assert!((ground_effect(r, 50.0 * r) - 1.0).abs() < 1e-3);

        let w = flat_world();
        let hover = |altitude: f32| {
            let mut s = DroneState::spawn(&w, 0.0, 0.0, altitude);
            run(&mut s, Command::default(), &w, 8.0);
            assert!((s.altitude(&w) - altitude).abs() < 0.05, "{}", s.altitude(&w));
            s.flight.thrust
        };
        // Just above the ground (scout: rotors of 0.1 m, center 0.26 m up).
        let low = hover(airframe().radius + 0.01);
        let high = hover(5.0);
        assert!((high - G).abs() < 1e-2, "{high}");
        assert!(low < high * 0.995, "{low} vs {high}");
    }

    #[test]
    fn battery_drains_and_the_climb_weakens() {
        let w = flat_world();
        let af = battery_airframe();
        let climb = |soc: f32| {
            let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
            s.flight.soc = soc;
            for _ in 0..50 {
                s.step(Command::default(), &af, &w, DT);
            }
            let soc_before = s.flight.soc;
            for _ in 0..50 {
                s.step(Command { up: 1.0, ..Default::default() }, &af, &w, DT);
            }
            assert!(s.flight.soc < soc_before);
            assert!(s.flight.energy_wh > 0.0);
            (s.velocity.y, s.flight.voltage)
        };
        let (full, v_full) = climb(1.0);
        let (empty, v_empty) = climb(0.02);
        assert!(v_empty < v_full - 2.0, "{v_empty} vs {v_full}");
        assert!(empty < full * 0.8, "climb {empty} vs {full}");

        // Hovering drains the pack at a plausible rate: tens of minutes.
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        for _ in 0..6000 {
            s.step(Command::default(), &af, &w, DT);
        }
        let used = 1.0 - s.flight.soc;
        assert!(used > 0.01 && used < 0.2, "{used}");
        let watts = s.flight.energy_wh * 60.0;
        assert!((30.0..300.0).contains(&watts), "{watts} W");
    }

    #[test]
    fn without_battery_energy_is_counted_and_soc_stays_full() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        run(&mut s, Command::default(), &w, 10.0);
        assert_eq!(s.flight.soc, 1.0);
        assert!(s.flight.energy_wh > 0.0);
    }

    #[test]
    fn cruise_speed_is_unchanged() {
        // Mission regression guard: every preset still cruises at its max speed.
        let w = flat_world();
        for path in ["scout.ron", "heavy.ron", "pro.ron", "cheap.ron", "mapper.ron"] {
            let text = std::fs::read_to_string(format!("{}/../../drones/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let af = drone_spec::DroneSpec::from_ron(&text).unwrap().airframe;
            // From the south edge northwards: 50 m of straight flight.
            let mut s = DroneState::spawn(&w, -10.0, 45.0, 5.0);
            for _ in 0..600 {
                s.step(Command { forward: 1.0, right: 0.3, ..Default::default() }, &af, &w, DT);
            }
            let h = glam::Vec2::new(s.velocity.x, s.velocity.z).length();
            assert!((h / af.max_speed - 1.0).abs() < 0.05, "{path}: {h} vs {}", af.max_speed);
        }
    }

    #[test]
    fn a_dead_motor_brings_the_drone_down() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 20.0);
        s.flight.motor_health[0] = 0.0;
        run(&mut s, Command::default(), &w, 2.0);
        assert!(s.flight.motors[0] < 1e-6);
        assert!(s.position.y < 19.0 || s.flight.up.angle_between(Vec3::Y) > 0.5, "{} {}", s.position, s.flight.up);
    }
}
