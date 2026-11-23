//! Kinematics of the real drone (ground truth).
//!
//! Deliberately simple model: the drone follows a velocity command in the drone
//! frame, with acceleration, speed and yaw bounded by its airframe. No
//! aerodynamics. The ground and the map edges stop it; collisions
//! with obstacles come with their detection.

use drone_spec::Airframe;
use glam::{Quat, Vec3};

use crate::World;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneState {
    pub position: Vec3,
    pub velocity: Vec3,
    /// Yaw in radians; 0 = facing -Z.
    pub yaw: f32,
    /// Actual acceleration over the last step, in m/s² (what an IMU measures).
    pub acceleration: Vec3,
    /// Yaw rate over the last step, in rad/s.
    pub yaw_rate: f32,
}

impl DroneState {
    /// Drone placed at (x, z), `altitude` m above the ground.
    pub fn spawn(world: &World, x: f32, z: f32, altitude: f32) -> Self {
        Self {
            position: Vec3::new(x, world.terrain.height_at(x, z) + altitude, z),
            velocity: Vec3::ZERO,
            yaw: 0.0,
            acceleration: Vec3::ZERO,
            yaw_rate: 0.0,
        }
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw)
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

        self.yaw_rate = clamp(cmd.yaw) * airframe.max_yaw_rate.to_radians();
        self.yaw += self.yaw_rate * dt;
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);

        // Command in the drone frame, clamped to the unit ball so that
        // the diagonal is not faster than an axis.
        let body = Vec3::new(clamp(cmd.right), clamp(cmd.up), -clamp(cmd.forward));
        let target = self.rotation() * body.clamp_length_max(1.0) * airframe.max_speed;

        let dv = (target - self.velocity).clamp_length_max(airframe.max_accel * dt);
        self.velocity = (self.velocity + dv).clamp_length_max(airframe.max_speed);
        self.position += self.velocity * dt;

        // Map edges: clamp and cancel the outward velocity.
        let half = world.terrain.half_size() - airframe.radius;
        for axis in [0, 2] {
            let p = self.position[axis];
            if p.abs() > half {
                self.position[axis] = p.clamp(-half, half);
                self.velocity[axis] = 0.0;
            }
        }

        // Ground: the drone's sphere never goes below it.
        let floor = world.terrain.height_at(self.position.x, self.position.z) + airframe.radius;
        if self.position.y < floor {
            self.position.y = floor;
            self.velocity.y = self.velocity.y.max(0.0);
        }
        self.acceleration = (self.velocity - v_before) / dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObstaclesSpec, TerrainSpec, WorldSpec};

    fn flat_world() -> World {
        World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::None,
        })
        .unwrap()
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
    fn forward_goes_toward_neg_z_at_max_speed() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        let cmd = Command { forward: 1.0, ..Default::default() };
        run(&mut s, cmd, &w, 3.0);
        assert!((s.velocity.length() - airframe().max_speed).abs() < 1e-3);
        assert!(s.velocity.z < 0.0 && s.velocity.x.abs() < 1e-4);
    }

    #[test]
    fn acceleration_is_bounded() {
        let w = flat_world();
        let a = airframe();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        let cmd = Command { forward: 1.0, right: 1.0, ..Default::default() };
        let mut prev = s.velocity;
        for _ in 0..200 {
            s.step(cmd, &a, &w, DT);
            assert!((s.velocity - prev).length() <= a.max_accel * DT + 1e-5);
            assert!(s.velocity.length() <= a.max_speed + 1e-5);
            prev = s.velocity;
        }
    }

    #[test]
    fn diagonal_is_not_faster() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        let cmd = Command { forward: 1.0, right: 1.0, up: 1.0, ..Default::default() };
        run(&mut s, cmd, &w, 4.0);
        assert!(s.velocity.length() <= airframe().max_speed + 1e-4);
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
    fn never_goes_below_ground() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 1.0);
        run(&mut s, Command { up: -1.0, ..Default::default() }, &w, 3.0);
        assert!((s.position.y - airframe().radius).abs() < 1e-5);
        assert!(s.velocity.y >= 0.0);
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
    fn stops_when_released() {
        let w = flat_world();
        let mut s = DroneState::spawn(&w, 0.0, 0.0, 5.0);
        run(&mut s, Command { forward: 1.0, ..Default::default() }, &w, 2.0);
        run(&mut s, Command::default(), &w, 3.0);
        assert!(s.velocity.length() < 1e-4);
    }
}
