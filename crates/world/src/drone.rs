//! Flight dynamics of the real drone (ground truth).
//!
//! A quadrotor: it only pushes along its thrust axis. To accelerate, brake or
//! hold against the wind, it tilts that axis.
//!
//! - **Flight controller.** The pilot command is a velocity setpoint in the drone
//!   frame. A PI loop turns the velocity error into a wanted acceleration (the
//!   integral learns the steady push of the wind), with a feed-forward of the drag
//!   in still air. The acceleration is bounded by `max_accel`, then by the tilt
//!   limit.
//! - **Attitude and motors.** The thrust axis turns towards the wanted one with
//!   a first-order lag and a bounded tilt rate; the thrust follows with the motor
//!   time constant.
//! - **Forces.** Thrust, gravity, and drag on the airspeed (velocity relative to
//!   the wind): linear (rotors) plus quadratic (body).
//! - **Contacts.** The ground and the map edges stop the drone; an obstacle
//!   pushes it back out and absorbs most of the impact.

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
    /// Thrust per unit mass, in m/s².
    pub thrust: f32,
    /// Integral of the velocity error, in m: what the controller learned about the
    /// steady disturbances (wind).
    pub integral: Vec3,
    /// Wind felt at the last step, in m/s.
    pub wind: Vec3,
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
            integral: Vec3::ZERO,
            wind: Vec3::ZERO,
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
    pub flight: Flight,
}

/// Velocity error beyond which the integral grows no faster, in m/s.
const INTEGRAL_ZONE: f32 = 0.5;
/// Fraction of the normal speed kept after hitting an obstacle.
const RESTITUTION: f32 = 0.2;
/// Fraction of the tangential speed kept after hitting an obstacle.
const SCRAPE: f32 = 0.7;

impl DroneState {
    /// Drone placed at (x, z), `altitude` m above the ground, hovering.
    pub fn spawn(world: &World, x: f32, z: f32, altitude: f32) -> Self {
        Self {
            position: Vec3::new(x, world.terrain.height_at(x, z) + altitude, z),
            velocity: Vec3::ZERO,
            yaw: 0.0,
            acceleration: Vec3::ZERO,
            yaw_rate: 0.0,
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

    /// Advances the simulation by `dt` seconds.
    pub fn step(&mut self, cmd: Command, airframe: &Airframe, world: &World, dt: f32) {
        let clamp = |v: f32| v.clamp(-1.0, 1.0);
        let v_before = self.velocity;
        let a = airframe;
        let f = &mut self.flight;
        f.t += dt;

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
        let drag_still_air = (a.drag + a.drag_quad * self.velocity.length()) * self.velocity;
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

        // Attitude: first-order lag, bounded rate.
        let angle = f.up.angle_between(up_wanted);
        if angle > 1e-6 {
            let lag = angle * (1.0 - (-dt / a.attitude_tau_s).exp());
            let turn = lag.min(a.tilt_rate_deg.to_radians() * dt);
            let axis = f.up.cross(up_wanted);
            if axis.length_squared() > 1e-12 {
                f.up = (Quat::from_axis_angle(axis.normalize(), turn) * f.up).normalize();
            } else {
                f.up = up_wanted;
            }
        }
        // Motors.
        f.thrust += (thrust_wanted - f.thrust) * (1.0 - (-dt / a.motor_tau_s).exp());

        // Forces: thrust, gravity, drag on the airspeed.
        f.wind = world.wind_at(self.position, f.t);
        let air = self.velocity - f.wind;
        let drag = -(a.drag + a.drag_quad * air.length()) * air;
        let accel = f.up * f.thrust - Vec3::Y * G + drag;
        self.velocity += accel * dt;
        self.position += self.velocity * dt;

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
}
