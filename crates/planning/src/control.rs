//! Tracking building blocks shared by missions: safe speed, altitude, command.

use drone_spec::Airframe;
use glam::{Vec2, Vec3};
use mapping::OccupancyGrid;
use world::drone::{Command, DroneState};

use crate::grid;

/// Safe-speed margin in front of an occupied cell, in m.
const OCCUPIED_MARGIN_M: f32 = 0.15;
/// Margin in front of an unknown cell, in m.
const UNKNOWN_MARGIN_M: f32 = 0.05;
/// Response time of the airframe, in s: before braking, a quadrotor must first
/// swing its tilt to the other side. Meanwhile it keeps its speed.
const RESPONSE_S: f32 = 0.2;

/// Vertical command holding the altitude (proportional controller).
pub fn hold_altitude(altitude: f32, state: &DroneState, airframe: &Airframe) -> f32 {
    ((altitude - state.position.y) * 2.0 / airframe.max_speed).clamp(-1.0, 1.0)
}

/// Distance at which `safe_velocity` looks for obstacles: a stop from full speed
/// (reaction, then braking at 70 %), plus margins.
pub fn safe_horizon(airframe: &Airframe) -> f32 {
    let vmax = airframe.max_speed;
    vmax * RESPONSE_S + vmax.powi(2) / (2.0 * 0.7 * airframe.max_accel) + airframe.radius + OCCUPIED_MARGIN_M + 0.5
}

/// Safe speed: the component pointing towards each nearby obstacle (or the unknown,
/// which may hide one) stays low enough to stop before touching it,
/// v·n ≤ √(2·a·(distance − radius)). Following a wall stays possible at full speed.
pub fn safe_velocity(grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe, wanted: Vec2) -> Vec2 {
    let pos = Vec2::new(state.position.x, state.position.z);
    // Unknown inside the body itself is ignored (the drone is there).
    let obstacles = grid::obstacles_near(grid, pos, safe_horizon(airframe), airframe.radius - UNKNOWN_MARGIN_M);
    limit_velocity(&obstacles, state, airframe, wanted, 0.0)
}

/// Caps `wanted` against `obstacles`, keeping `extra_m` more than the default margins.
/// Margins: 70 % of the max deceleration (command delay, direction of the obstacle
/// turning); 0.15 m in front of a known obstacle (distance measured in cells),
/// 0.05 m in front of the unknown (maybe nothing): the drone can skirt closely
/// what it has not seen, without ever entering it.
pub fn limit_velocity(obstacles: &[grid::Near], state: &DroneState, airframe: &Airframe, wanted: Vec2, extra_m: f32) -> Vec2 {
    let a = airframe.max_accel;
    let v_now = Vec2::new(state.velocity.x, state.velocity.z);
    let limit = |o: &grid::Near| {
        let margin = if o.occupied { OCCUPIED_MARGIN_M + extra_m } else { UNKNOWN_MARGIN_M };
        let reaction = v_now.dot(o.dir).max(0.0) * RESPONSE_S;
        (2.0 * 0.7 * a * (o.distance - airframe.radius - margin - reaction).max(0.0)).sqrt()
    };
    let mut v2 = wanted;
    // Already too fast towards an obstacle (momentum from before a heading change):
    // braking takes priority. Start from the current velocity so that all available
    // acceleration goes into braking. Unless the wanted velocity leads away from
    // every such obstacle: it brakes as well, and copying a slow drift inside the
    // margin would hold it forever, the drone never leaving the wall.
    if obstacles.iter().any(|o| v_now.dot(o.dir) > limit(o) && wanted.dot(o.dir) > 0.0) {
        v2 = v_now;
    }
    // Successive projections onto each constraint v·n ≤ limit(d).
    for _ in 0..3 {
        for o in obstacles {
            let excess = v2.dot(o.dir) - limit(o);
            if excess > 0.0 {
                v2 -= o.dir * excess;
            }
        }
    }
    v2
}

/// Heading (yaw) facing the direction `dir` (x, z).
pub fn yaw_towards(dir: Vec2) -> f32 {
    (-dir.x).atan2(-dir.y)
}

/// Normalized command for a desired horizontal velocity, a yaw error and
/// a vertical command.
pub fn command(state: &DroneState, airframe: &Airframe, v2: Vec2, yaw_err: f32, up: f32) -> Command {
    let v = Vec3::new(v2.x, 0.0, v2.y);
    let fwd = state.forward();
    let right = Vec3::new(-fwd.z, 0.0, fwd.x);
    Command {
        forward: v.dot(fwd) / airframe.max_speed,
        right: v.dot(right) / airframe.max_speed,
        up,
        yaw: (yaw_err * 3.0 / airframe.max_yaw_rate.to_radians()).clamp(-1.0, 1.0),
    }
}

/// Angle wrapped into ]-π, π].
pub fn wrap(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let a = a.rem_euclid(TAU);
    if a > PI { a - TAU } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;
    use drone_spec::DroneSpec;
    use world::{ObstaclesSpec, TerrainSpec, World, WorldSpec};

    /// Inside the margin of a wall, drifting slowly towards it: the drone may still
    /// leave along a path that leads away (the command no longer copies the drift).
    #[test]
    fn a_slow_drift_into_a_wall_does_not_hold_the_drone() {
        let w = World::build(&WorldSpec {
            size_m: 50.0,
            cell_m: 2.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::None,
            wind: Default::default(),
        })
        .unwrap();
        let airframe = DroneSpec::scout().airframe;
        let mut state = DroneState::spawn(&w, 0.0, 0.0, 2.0);
        state.velocity = Vec3::new(0.1, 0.0, 0.03);
        // Wall 0.28 m away along +z: within the 0.15 m margin of the 0.25 m body.
        let wall = [grid::Near { distance: 0.28, dir: Vec2::Y, occupied: true }];
        let away = Vec2::new(-0.95, -0.3) * 2.0;
        let v = limit_velocity(&wall, &state, &airframe, away, 0.0);
        assert!((v - away).length() < 1e-4, "{v}");
        // Towards the wall at speed, braking still wins: no velocity towards it.
        state.velocity = Vec3::new(0.0, 0.0, 1.5);
        let along = Vec2::new(-1.0, 0.3) * 2.0;
        let v = limit_velocity(&wall, &state, &airframe, along, 0.0);
        assert!(v.dot(Vec2::Y) <= 1e-4 && v.x.abs() < 1e-4, "{v}");
    }
}
