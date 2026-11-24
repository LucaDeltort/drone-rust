//! Autonomous exploration of an unknown area.
//!
//! Loop: pick a frontier (the nearest, or the best trade-off
//! distance / information gain), plan a path to it through known free space,
//! follow it, replan regularly as the map changes. Done when
//! no frontier is reachable.

use std::collections::HashSet;

use drone_spec::Airframe;
use glam::Vec2;
use mapping::OccupancyGrid;
use world::drone::{Command, DroneState};

use crate::control::{self, wrap};
use crate::frontier::{self, Frontier};
use crate::grid::{self, CSpace, DistanceField};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Go to the nearest frontier (in travel distance).
    Nearest,
    /// Minimize λ · distance − information gain.
    InfoGain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplorerConfig {
    pub strategy: Strategy,
    /// Margin around obstacles (drone radius + safety), in m.
    pub inflate_m: f32,
    /// Margin around the unknown, which may hide an obstacle, in m.
    pub unknown_margin_m: f32,
    /// The drone aims at a safe cell at most this far from a frontier, in m.
    pub view_m: f32,
    /// Weight of distance against gain, in m² of gain per m travelled.
    pub lambda: f32,
    /// Radius for computing the information gain (≈ useful sensor range), in m.
    pub gain_radius_m: f32,
    /// Minimum size of a frontier, in cells.
    pub min_frontier_cells: usize,
    /// Replanning period, in s.
    pub replan_s: f32,
    /// Acceptance radius of a path vertex, in m. Small: the drone aims at each
    /// vertex and thus follows the segments without cutting corners.
    pub waypoint_radius_m: f32,
    /// Without progress for this long, the target is abandoned, in s.
    pub stuck_s: f32,
    /// Only explore this area: frontiers outside are ignored, and the
    /// exploration is done when the area is.
    pub zone: Option<crate::Zone>,
}

impl ExplorerConfig {
    pub fn new(strategy: Strategy, airframe: &Airframe) -> Self {
        Self {
            strategy,
            inflate_m: airframe.radius + 0.35,
            unknown_margin_m: airframe.radius + 0.35,
            view_m: 1.5,
            lambda: 10.0,
            gain_radius_m: 10.0,
            min_frontier_cells: 4,
            replan_s: 0.5,
            waypoint_radius_m: 0.3,
            stuck_s: 6.0,
            zone: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Exploring,
    /// No reachable frontier left: area explored.
    Done,
}

pub struct Explorer {
    pub cfg: ExplorerConfig,
    /// Absolute flight height (center of the mapped slab).
    pub altitude: f32,
    pub status: Status,
    /// Current path (x, z), from the drone to the target.
    pub path: Vec<Vec2>,
    pub target: Option<Vec2>,
    /// Frontier centers at the last computation (for display).
    pub frontiers: Vec<Vec2>,
    next: usize,
    since_replan: f32,
    /// Targets abandoned for lack of progress.
    blacklist: Vec<Vec2>,
    /// Frontier cells that a visit did not resolve: no longer targeted.
    exhausted: HashSet<(usize, usize)>,
    /// Frontiers of the last computation, to know what a visit should have resolved.
    last_frontier_cells: Vec<(usize, usize)>,
    cell_m: f32,
    /// Minimum (x, z) corner of the grid, to recover a cell's center.
    grid_origin: Vec2,
    best_remaining: f32,
    since_progress: f32,
    /// Spin in place in progress: remaining yaw angle, in rad.
    spin_left: Option<f32>,
    /// Where the last scan spin took place.
    scanned_at: Option<Vec2>,
    /// Time since start, in s.
    elapsed: f32,
}

/// Distance from the last scan spin beyond which a new spin makes sense, in m.
const RESCAN_M: f32 = 1.0;
/// Wait for the first measurements before concluding anything, in s.
const FIRST_LOOK_S: f32 = 0.5;

/// Radius within which an old target is considered the same.
const SAME_TARGET_M: f32 = 1.5;
/// Exclusion radius around an abandoned target.
const BLACKLIST_M: f32 = 0.5;

impl Explorer {
    pub fn new(cfg: ExplorerConfig, altitude: f32) -> Self {
        Self {
            cfg,
            altitude,
            status: Status::Exploring,
            path: Vec::new(),
            target: None,
            frontiers: Vec::new(),
            next: 0,
            since_replan: f32::INFINITY,
            blacklist: Vec::new(),
            exhausted: HashSet::new(),
            last_frontier_cells: Vec::new(),
            cell_m: 0.0,
            grid_origin: Vec2::ZERO,
            best_remaining: f32::INFINITY,
            since_progress: 0.0,
            spin_left: None,
            scanned_at: None,
            elapsed: 0.0,
        }
    }

    /// One decision step: returns the drone command.
    pub fn update(&mut self, grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe, dt: f32) -> Command {
        self.since_replan += dt;
        self.elapsed += dt;
        if let Some(left) = self.spin_left {
            // Spin in place: sensors that do not see all around (camera,
            // sonars) sweep the surroundings before we conclude.
            let rate = airframe.max_yaw_rate.to_radians();
            let left = left - rate * dt;
            self.spin_left = (left > 0.0).then_some(left);
            if self.spin_left.is_none() {
                self.since_replan = f32::INFINITY;
            }
            let up = control::hold_altitude(self.altitude, state, airframe);
            // Zero velocity (the drone brakes), rotation at max rate.
            let stop = control::command(state, airframe, Vec2::ZERO, 0.0, up);
            return Command { yaw: 1.0, ..stop };
        }
        if self.since_replan >= self.cfg.replan_s || self.path.is_empty() && self.status == Status::Exploring {
            self.replan(grid, state);
        }
        let cmd = self.follow(grid, state, airframe, dt);
        if self.status == Status::Exploring && self.path.is_empty() {
            // Target reached: chain on almost without waiting for the next period.
            self.since_replan = self.since_replan.max(self.cfg.replan_s - 0.05);
        }
        cmd
    }

    fn replan(&mut self, grid: &OccupancyGrid, state: &DroneState) {
        self.since_replan = 0.0;
        let pos = Vec2::new(state.position.x, state.position.z);
        let Some(start) = grid.index(pos.x, pos.y) else {
            self.finish();
            return;
        };
        let cspace = CSpace::build(grid, self.cfg.inflate_m, self.cfg.unknown_margin_m);
        let field = DistanceField::from(&cspace, grid.cell_m(), start);
        let mut frontiers = frontier::detect(grid, self.cfg.min_frontier_cells, &self.exhausted);
        if let Some(zone) = self.cfg.zone {
            // Keep only the frontier cells inside the area.
            frontiers.retain_mut(|f| {
                f.cells.retain(|&(i, j)| zone.contains(grid.center(i, j)));
                if f.cells.len() < self.cfg.min_frontier_cells {
                    return false;
                }
                let sum: Vec2 = f.cells.iter().map(|&(i, j)| grid.center(i, j)).sum();
                f.centroid = sum / f.cells.len() as f32;
                true
            });
        }
        self.last_frontier_cells = frontiers.iter().flat_map(|f| f.cells.iter().copied()).collect();
        self.cell_m = grid.cell_m();
        self.grid_origin = grid.center(0, 0) - grid.cell_m() * 0.5;
        self.frontiers = frontiers.iter().map(|f| f.centroid).collect();

        let Some((goal, _)) = self.choose(grid, &cspace, &field, &frontiers) else {
            self.path.clear();
            // No measurement yet: wait.
            if self.elapsed < FIRST_LOOK_S && grid.cell(start.0, start.1) == mapping::Cell::Unknown {
                return;
            }
            // Before concluding, a spin in place to look everywhere.
            if self.scanned_at.is_none_or(|p| p.distance(pos) > RESCAN_M) {
                self.scanned_at = Some(pos);
                self.spin_left = Some(std::f32::consts::TAU);
                return;
            }
            self.finish();
            return;
        };
        let goal_xz = grid.center(goal.0, goal.1);
        if self.target.is_none_or(|t| t.distance(goal_xz) > SAME_TARGET_M) {
            self.best_remaining = f32::INFINITY;
            self.since_progress = 0.0;
        }
        self.target = Some(goal_xz);
        let cells = field.path_to(goal).expect("target chosen among reachable cells");
        if cells.len() < 2 {
            // Already on the target: its frontier was not resolved on the way here.
            self.exhaust_around(goal_xz);
            self.target = None;
            self.path.clear();
            return;
        }
        let mut path = grid::to_world(grid, &grid::smooth(&cspace, &cells));
        // The path starts from the drone's exact position, not from its cell's center.
        path[0] = pos;
        self.path = path;
        self.next = 1.min(self.path.len() - 1);
    }

    /// Picks the target cell: for each frontier, the safe and reachable cell
    /// nearest to the drone among those within `view_m` of the frontier.
    fn choose(
        &self,
        grid: &OccupancyGrid,
        cspace: &CSpace,
        field: &DistanceField,
        frontiers: &[Frontier],
    ) -> Option<((usize, usize), f32)> {
        let (nx, nz) = grid.size();
        let r = (self.cfg.view_m / grid.cell_m()).ceil() as i64;
        let r2 = (self.cfg.view_m / grid.cell_m()).powi(2);
        let candidates = frontiers.iter().filter_map(|f| {
            let mut best: Option<((usize, usize), f32)> = None;
            for &(fi, fj) in &f.cells {
                for dj in -r..=r {
                    for di in -r..=r {
                        let (i, j) = (fi as i64 + di, fj as i64 + dj);
                        if (di * di + dj * dj) as f32 > r2 || i < 0 || j < 0 || i as usize >= nx || j as usize >= nz {
                            continue;
                        }
                        let (i, j) = (i as usize, j as usize);
                        let d = field.distance(i, j);
                        if cspace.ok(i, j) && d.is_finite() && best.is_none_or(|(_, b)| d < b) {
                            best = Some(((i, j), d));
                        }
                    }
                }
            }
            let (cell, d) = best?;
            let xz = grid.center(cell.0, cell.1);
            if self.blacklist.iter().any(|b| b.distance(xz) < BLACKLIST_M) {
                return None;
            }
            let cost = match self.cfg.strategy {
                Strategy::Nearest => d,
                Strategy::InfoGain => {
                    self.cfg.lambda * d - frontier::information_gain(grid, cell, self.cfg.gain_radius_m)
                }
            };
            // Hysteresis: keep the current target if it is still a frontier.
            let keep = self.target.is_some_and(|t| t.distance(xz) < SAME_TARGET_M);
            Some((cell, if keep { cost - 1e6 } else { cost }))
        });
        candidates.min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The target is reached: the frontier cells that should have been seen from here and that
    /// are still there at the next computation would be targeted again; discard them.
    fn exhaust_around(&mut self, target: Vec2) {
        let reach = self.cfg.view_m + self.cell_m;
        for &c in &self.last_frontier_cells {
            if self.cell_center(c).distance(target) <= reach {
                self.exhausted.insert(c);
            }
        }
    }

    fn cell_center(&self, (i, j): (usize, usize)) -> Vec2 {
        self.grid_origin + (Vec2::new(i as f32, j as f32) + 0.5) * self.cell_m
    }

    fn finish(&mut self) {
        self.status = Status::Done;
        self.path.clear();
        self.target = None;
    }

    /// Path following: aim at the next vertex, speed bounded by the stopping
    /// distance, yaw turned towards the motion (gaze control), altitude held.
    fn follow(&mut self, grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe, dt: f32) -> Command {
        let pos = Vec2::new(state.position.x, state.position.z);
        let up = control::hold_altitude(self.altitude, state, airframe);
        if self.path.len() < 2 {
            self.path.clear();
            return Command { up, ..Default::default() };
        }
        while self.next < self.path.len() - 1 && pos.distance(self.path[self.next]) < self.cfg.waypoint_radius_m {
            self.next += 1;
        }
        let aim = self.path[self.next];
        let remaining = pos.distance(aim)
            + self.path[self.next..].windows(2).map(|w| w[0].distance(w[1])).sum::<f32>();
        if remaining < 0.3 {
            // Target reached. If its frontier does not disappear (unobservable: behind
            // a wall, too high, too low), it will not be targeted again.
            if let Some(t) = self.target.take() {
                self.exhaust_around(t);
            }
            self.path.clear();
            return Command { up, ..Default::default() };
        }

        // Progress towards the target, otherwise give up.
        if remaining < self.best_remaining - 0.5 {
            self.best_remaining = remaining;
            self.since_progress = 0.0;
        } else {
            self.since_progress += dt;
            if self.since_progress > self.cfg.stuck_s {
                if let Some(t) = self.target {
                    self.blacklist.push(t);
                }
                self.path.clear();
                self.target = None;
                self.since_progress = 0.0;
                return Command { up, ..Default::default() };
            }
        }

        // Speed: max, stopping distance before the end, slowing down before a corner.
        let a = airframe.max_accel;
        let mut speed = airframe.max_speed.min((2.0 * a * remaining).sqrt());
        if self.next < self.path.len() - 1 {
            let turn = (aim - pos).normalize_or_zero().dot((self.path[self.next + 1] - aim).normalize_or_zero());
            if turn < 0.85 {
                speed = speed.min((2.0 * a * pos.distance(aim)).sqrt() + 1.0);
            }
        }

        let dir = (aim - pos).normalize_or_zero();
        let err = wrap(control::yaw_towards(dir) - state.yaw);
        // Gaze control: only move towards where we look; too far off,
        // turn first (forward sensors have not seen the side).
        speed *= err.cos().max(0.0);
        let v2 = control::safe_velocity(grid, state, airframe, dir * speed);
        control::command(state, airframe, v2, err, up)
    }
}

#[cfg(test)]
mod tests;
