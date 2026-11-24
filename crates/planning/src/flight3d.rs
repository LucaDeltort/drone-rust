//! 3D flight over the voxel map: the drone climbs over what it can, and only
//! goes around what is too tall.
//!
//! - **Projection.** The voxel map is flattened onto the 2D planning grid: a column
//!   is occupied only if its obstacle stands taller than `max_overfly_m` above the
//!   surrounding ground; lower ones (fences, cars, a house) are free, since the drone
//!   can fly over them. The 2D planners are unchanged.
//! - **Altitude.** The target altitude is the highest surface under and ahead of
//!   the drone (ground, roofs, canopies), plus `clearance_v_m`: the drone follows
//!   the relief and climbs over obstacles, then comes back down, slowly.
//! - **Guard.** Whatever still stands at the drone's current height (an obstacle it
//!   has not climbed over yet, a rising slope) caps its horizontal speed, with
//!   `clearance_h_m` of extra distance: it waits to have climbed before going on.

use drone_spec::Airframe;
use glam::{IVec2, Vec2, Vec3};
use mapping::{Cell, OccupancyGrid, VoxelMap};
use world::drone::{Command, DroneState};

use crate::control;
use crate::grid::Near;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flight3dConfig {
    /// Height kept above the surfaces under and ahead of the drone, in m.
    pub clearance_v_m: f32,
    /// Extra horizontal distance kept from what stands at the drone's height, in m.
    pub clearance_h_m: f32,
    /// Obstacles taller than this above their ground are flown around, in m.
    pub max_overfly_m: f32,
    /// How far ahead the altitude looks, in s of flight.
    pub lookahead_s: f32,
    /// Max descent rate of the target altitude, in m/s.
    pub descent_ms: f32,
}

impl Default for Flight3dConfig {
    fn default() -> Self {
        Self { clearance_v_m: 2.5, clearance_h_m: 0.6, max_overfly_m: 12.0, lookahead_s: 1.5, descent_ms: 1.5 }
    }
}

#[derive(Debug, Clone)]
pub struct Flight3d {
    pub cfg: Flight3dConfig,
    /// Current target altitude (absolute), in m.
    pub target: Option<f32>,
}

/// Extra radius searched for the ground, in m.
const GROUND_SEARCH_M: f32 = 4.0;

/// Vertical room kept above and below the drone's body when checking what stands
/// at its height, in m.
const VERTICAL_MARGIN_M: f32 = 0.4;

impl Flight3d {
    pub fn new(cfg: Flight3dConfig) -> Self {
        Self { cfg, target: None }
    }

    /// Height of a column's obstacle above the ground around it (lowest occupied
    /// voxel of the 3 × 3 neighbourhood).
    fn obstacle_height(voxels: &VoxelMap, c: IVec2) -> Option<f32> {
        let top = voxels.column_top(c)?;
        let floor = (-1..=1)
            .flat_map(|dz| (-1..=1).map(move |dx| c + IVec2::new(dx, dz)))
            .filter_map(|n| voxels.column_floor(n))
            .fold(f32::INFINITY, f32::min);
        Some(top - floor)
    }

    /// Too tall to fly over?
    pub fn is_tall(&self, voxels: &VoxelMap, c: IVec2) -> bool {
        Self::obstacle_height(voxels, c).is_some_and(|h| h > self.cfg.max_overfly_m)
    }

    /// State of a column on the planning grid.
    pub fn column_state(&self, voxels: &VoxelMap, c: IVec2) -> Cell {
        if !voxels.column_known(c) {
            Cell::Unknown
        } else if self.is_tall(voxels, c) {
            Cell::Occupied
        } else {
            Cell::Free
        }
    }

    /// Projects `columns` of the voxel map onto the planning grid. A tall column
    /// also changes the height estimate of its neighbours: pass the changed
    /// columns, their neighbours are refreshed too.
    pub fn project(&self, voxels: &VoxelMap, grid: &mut OccupancyGrid, columns: &[IVec2]) {
        let v = voxels.voxel_m();
        let mut todo: Vec<IVec2> = columns
            .iter()
            .flat_map(|&c| (-1..=1).flat_map(move |dz| (-1..=1).map(move |dx| c + IVec2::new(dx, dz))))
            .collect();
        todo.sort_by_key(|c| (c.x, c.y));
        todo.dedup();
        for c in todo {
            let state = self.column_state(voxels, c);
            if state == Cell::Unknown {
                continue;
            }
            let (x0, z0) = (c.x as f32 * v, c.y as f32 * v);
            let (Some((i0, j0)), Some((i1, j1))) = (grid.index(x0 + 1e-4, z0 + 1e-4), grid.index(x0 + v - 1e-4, z0 + v - 1e-4))
            else {
                continue;
            };
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let p = grid.center(i, j);
                    if (x0..x0 + v).contains(&p.x) && (z0..z0 + v).contains(&p.y) {
                        grid.set_cell(i, j, state);
                    }
                }
            }
        }
    }

    /// Highest surface (top of a column that can be flown over) within `radius`
    /// of `center`, and the lowest ground there.
    fn highest_surface(&self, voxels: &VoxelMap, center: Vec2, radius: f32) -> Option<(f32, f32)> {
        let v = voxels.voxel_m();
        let c0 = voxels.column_key(center.x, center.y);
        // The ground is looked for a bit wider: under a wall seen from above, only
        // its top is known.
        let ground_radius = radius + GROUND_SEARCH_M;
        let n = (ground_radius / v).ceil() as i32 + 1;
        let mut best: Option<f32> = None;
        let mut floor = f32::INFINITY;
        for dz in -n..=n {
            for dx in -n..=n {
                let c = c0 + IVec2::new(dx, dz);
                let d = ((c.as_vec2() + 0.5) * v).distance(center);
                if d > ground_radius + 0.71 * v {
                    continue;
                }
                if let Some(f) = voxels.column_floor(c) {
                    floor = floor.min(f);
                }
                if d > radius + 0.71 * v || self.is_tall(voxels, c) {
                    continue;
                }
                if let Some(top) = voxels.column_top(c) {
                    best = Some(best.map_or(top, |b| b.max(top)));
                }
            }
        }
        best.map(|b| (b, floor))
    }

    /// Updates and returns the target altitude: highest surface under and ahead of
    /// the drone, plus the vertical clearance. Climbs at once, descends slowly.
    pub fn altitude_target(&mut self, voxels: &VoxelMap, state: &DroneState, airframe: &Airframe, dt: f32) -> f32 {
        let pos = Vec2::new(state.position.x, state.position.z);
        let v = Vec2::new(state.velocity.x, state.velocity.z);
        let half = v * self.cfg.lookahead_s / 2.0;
        let radius = half.length() + airframe.radius + self.cfg.clearance_h_m + voxels.voxel_m();
        // Never higher than needed to clear the tallest obstacle it may fly over: a
        // face seen from below keeps growing until it is known too tall.
        let want = self
            .highest_surface(voxels, pos + half, radius)
            .map(|(top, floor)| top.min(floor + self.cfg.max_overfly_m) + self.cfg.clearance_v_m);
        let prev = self.target.unwrap_or(state.position.y);
        let target = match want {
            Some(w) if w >= prev => w,
            Some(w) => w.max(prev - self.cfg.descent_ms * dt),
            None => prev,
        };
        self.target = Some(target);
        target
    }

    /// What stands at the drone's current height, near it, as obstacles for the
    /// safe speed.
    pub fn obstacles_at_height(&self, voxels: &VoxelMap, state: &DroneState, airframe: &Airframe) -> Vec<Near> {
        let v = voxels.voxel_m();
        let pos = Vec2::new(state.position.x, state.position.z);
        let (lo, hi) = (
            state.position.y - airframe.radius - VERTICAL_MARGIN_M,
            state.position.y + airframe.radius + VERTICAL_MARGIN_M,
        );
        let horizon = control::safe_horizon(airframe) + self.cfg.clearance_h_m;
        let c0 = voxels.column_key(pos.x, pos.y);
        let n = (horizon / v).ceil() as i32 + 1;
        let mut out = Vec::new();
        for dz in -n..=n {
            for dx in -n..=n {
                let c = c0 + IVec2::new(dx, dz);
                if !voxels.column_blocks(c, lo, hi) {
                    continue;
                }
                let to = (c.as_vec2() + 0.5) * v - pos;
                let d = (to.length() - 0.5 * v).max(0.0);
                if d < horizon {
                    out.push(Near { distance: d, dir: to.normalize_or_zero(), occupied: true });
                }
            }
        }
        out
    }

    /// Takes the planner's command and makes it 3D: altitude from the surfaces
    /// ahead, horizontal speed capped by what stands at the current height.
    pub fn guard(&mut self, voxels: &VoxelMap, state: &DroneState, airframe: &Airframe, cmd: Command, dt: f32) -> Command {
        let target = self.altitude_target(voxels, state, airframe, dt);
        let up = control::hold_altitude(target, state, airframe);
        let fwd = state.forward();
        let right = Vec3::new(-fwd.z, 0.0, fwd.x);
        let wanted = (fwd * cmd.forward + right * cmd.right) * airframe.max_speed;
        let obstacles = self.obstacles_at_height(voxels, state, airframe);
        let v2 = control::limit_velocity(&obstacles, state, airframe, Vec2::new(wanted.x, wanted.z), self.cfg.clearance_h_m);
        let v = Vec3::new(v2.x, 0.0, v2.y);
        Command { forward: v.dot(fwd) / airframe.max_speed, right: v.dot(right) / airframe.max_speed, up, yaw: cmd.yaw }
    }
}

#[cfg(test)]
mod tests;
