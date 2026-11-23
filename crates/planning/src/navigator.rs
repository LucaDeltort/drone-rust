//! A-to-B mission on an unknown map.
//!
//! Optimistic planning: the unknown is assumed free, only known occupied
//! cells (inflated by the drone's radius and a margin) block. D* Lite keeps its
//! search from one step to the next and only repairs what the new obstacles
//! change. The path is smoothed, then turned into a timed trajectory (speed
//! profile), followed with the safe speed: in the short term, the drone never rushes
//! towards what it has not seen yet.

use drone_spec::Airframe;
use glam::Vec2;
use mapping::{Cell, OccupancyGrid};
use world::drone::{Command, DroneState};

use crate::control::{self, wrap};
use crate::dstar::DStarLite;
use crate::trajectory::Trajectory;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigatorConfig {
    /// Margin around known obstacles (drone radius + safety), in m.
    pub inflate_m: f32,
    /// Replanning period, in s.
    pub replan_s: f32,
    /// Acceptance radius of a path vertex, in m.
    pub waypoint_radius_m: f32,
    /// B is reached at this distance, in m.
    pub goal_radius_m: f32,
    /// Deviation tolerated in a corner, which sets the cornering speed, in m.
    pub corner_error_m: f32,
    /// Share of the max acceleration used for the speed profile.
    pub accel_share: f32,
}

impl NavigatorConfig {
    pub fn new(airframe: &Airframe) -> Self {
        Self {
            inflate_m: airframe.radius + 0.35,
            replan_s: 0.1,
            waypoint_radius_m: 0.3,
            goal_radius_m: 0.5,
            corner_error_m: 0.3,
            accel_share: 0.8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavStatus {
    Flying,
    /// B is reached.
    Reached,
    /// No known path to B for now (B or the drone off the map, or
    /// enclosed). The drone hovers and replans.
    NoPath,
}

/// Planning cost measurements.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavStats {
    pub replans: usize,
    /// Cells expanded by D* Lite, in total.
    pub expanded: usize,
    /// Cells whose state (blocked or not) changed, in total.
    pub changed_cells: usize,
}

pub struct Navigator {
    pub cfg: NavigatorConfig,
    /// Absolute flight height (center of the mapped slab).
    pub altitude: f32,
    pub goal: Vec2,
    pub status: NavStatus,
    /// Current smoothed path (x, z), from the drone to B.
    pub path: Vec<Vec2>,
    pub trajectory: Option<Trajectory>,
    pub stats: NavStats,
    dstar: Option<DStarLite>,
    inflation: Option<Inflation>,
    next: usize,
    since_replan: f32,
}

impl Navigator {
    pub fn new(cfg: NavigatorConfig, altitude: f32, goal: Vec2) -> Self {
        Self {
            cfg,
            altitude,
            goal,
            status: NavStatus::Flying,
            path: Vec::new(),
            trajectory: None,
            stats: NavStats::default(),
            dstar: None,
            inflation: None,
            next: 0,
            since_replan: f32::INFINITY,
        }
    }

    /// Predicted remaining time to B, in s.
    pub fn eta(&self) -> Option<f32> {
        self.trajectory.as_ref().map(Trajectory::duration)
    }

    /// One decision step: returns the drone command.
    pub fn update(&mut self, grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe, dt: f32) -> Command {
        let pos = Vec2::new(state.position.x, state.position.z);
        let up = control::hold_altitude(self.altitude, state, airframe);
        if self.status == NavStatus::Reached || pos.distance(self.goal) < self.cfg.goal_radius_m {
            self.status = NavStatus::Reached;
            self.path.clear();
            self.trajectory = None;
            return Command { up, ..Default::default() };
        }
        self.since_replan += dt;
        if self.since_replan >= self.cfg.replan_s {
            self.replan(grid, state, airframe);
        }
        if self.status != NavStatus::Flying || self.path.len() < 2 {
            return Command { up, ..Default::default() };
        }
        self.follow(grid, state, airframe, up)
    }

    fn replan(&mut self, grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe) {
        self.since_replan = 0.0;
        self.stats.replans += 1;
        let pos = Vec2::new(state.position.x, state.position.z);
        let (Some(here), Some(goal)) = (grid.index(pos.x, pos.y), grid.index(self.goal.x, self.goal.y)) else {
            self.no_path();
            return;
        };

        // Blocked cells: the map's occupied cells, inflated.
        let inflation = self.inflation.get_or_insert_with(|| Inflation::new(grid, self.cfg.inflate_m));
        let changes = inflation.update(grid);
        let (nx, nz) = grid.size();
        let dstar = self.dstar.get_or_insert_with(|| {
            DStarLite::new(nx, nz, grid.cell_m(), inflation.blocked(), here, goal)
        });
        self.stats.changed_cells += dstar.set_blocked(changes);

        // Drone within an obstacle's margin (too close to a wall): plan from
        // the nearest free cell, joined in a straight line.
        let escape = if dstar.is_blocked(here) { nearest_free(dstar, grid, here) } else { Some(here) };
        let Some(start) = escape else {
            self.no_path();
            return;
        };
        dstar.move_start(start);
        let before = dstar.expanded;
        dstar.compute();
        self.stats.expanded += dstar.expanded - before;
        let Some(cells) = dstar.path() else {
            self.no_path();
            return;
        };

        let cells = smooth(dstar, &cells);
        let mut path = vec![pos];
        path.extend(cells.iter().skip(1).map(|&(i, j)| grid.center(i, j)));
        if start != here {
            path.insert(1, grid.center(start.0, start.1));
        }
        // End exactly on B, not at the center of its cell.
        match path.len() {
            1 => path.push(self.goal),
            _ => *path.last_mut().unwrap() = self.goal,
        }
        path.dedup_by(|a, b| a.distance(*b) < 1e-3);
        if path.len() < 2 {
            path.push(self.goal);
        }

        let dir = (path[1] - path[0]).normalize_or_zero();
        let v0 = Vec2::new(state.velocity.x, state.velocity.z).dot(dir).max(0.0);
        let accel = airframe.max_accel * self.cfg.accel_share;
        self.trajectory = Some(Trajectory::plan(path.clone(), v0, airframe.max_speed, accel, self.cfg.corner_error_m));
        self.path = path;
        self.next = 1;
        self.status = NavStatus::Flying;
    }

    fn no_path(&mut self) {
        self.status = NavStatus::NoPath;
        self.path.clear();
        self.trajectory = None;
    }

    /// Tracking: aim at the next vertex, at the profile speed where the
    /// drone is, bounded by the safe speed; yaw turned towards the motion.
    fn follow(&mut self, grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe, up: f32) -> Command {
        let pos = Vec2::new(state.position.x, state.position.z);
        while self.next < self.path.len() - 1 && pos.distance(self.path[self.next]) < self.cfg.waypoint_radius_m {
            self.next += 1;
        }
        let aim = self.path[self.next];
        let traj = self.trajectory.as_ref().expect("trajectory computed with the path");
        // Drone's arc length: projection onto the current segment.
        let a = self.path[self.next - 1];
        let seg = aim - a;
        let along = if seg.length_squared() > 0.0 { (pos - a).dot(seg) / seg.length_squared() } else { 1.0 };
        let s = traj.s[self.next - 1] + along.clamp(0.0, 1.0) * seg.length();
        // Never quite zero: the end of the profile stops right on B.
        let mut speed = traj.speed_at(s).max(0.4);

        let dir = (aim - pos).normalize_or_zero();
        let err = wrap(control::yaw_towards(dir) - state.yaw);
        speed *= err.cos().max(0.0);
        let v2 = control::safe_velocity(grid, state, airframe, dir * speed);
        control::command(state, airframe, v2, err, up)
    }
}

/// Nearest unblocked cell (as the crow flies), within 3 m.
fn nearest_free(dstar: &DStarLite, grid: &OccupancyGrid, (ci, cj): (usize, usize)) -> Option<(usize, usize)> {
    let (nx, nz) = grid.size();
    let r = (3.0 / grid.cell_m()).ceil() as i64;
    let mut best: Option<((usize, usize), i64)> = None;
    for dj in -r..=r {
        for di in -r..=r {
            let (i, j) = (ci as i64 + di, cj as i64 + dj);
            if i < 0 || j < 0 || i as usize >= nx || j as usize >= nz {
                continue;
            }
            let c = (i as usize, j as usize);
            let d2 = di * di + dj * dj;
            if d2 <= r * r && !dstar.is_blocked(c) && grid.cell(c.0, c.1) != Cell::Occupied && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((c, d2));
            }
        }
    }
    best.map(|(c, _)| c)
}

/// Forward visibility smoothing: from each anchor, advance as long as the
/// straight line touches no blocked cell. Linear, for long paths.
fn smooth(dstar: &DStarLite, cells: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let visible = |a: (usize, usize), b: (usize, usize)| {
        let (ax, az) = (a.0 as f32, a.1 as f32);
        let (bx, bz) = (b.0 as f32, b.1 as f32);
        let n = ((bx - ax).abs().max((bz - az).abs()) * 4.0).ceil().max(1.0) as usize;
        (0..=n).all(|k| {
            let t = k as f32 / n as f32;
            let c = ((ax + (bx - ax) * t).round() as usize, (az + (bz - az) * t).round() as usize);
            !dstar.is_blocked(c)
        })
    };
    let mut out = vec![cells[0]];
    let mut anchor = 0;
    while anchor < cells.len() - 1 {
        let mut next = anchor + 1;
        while next + 1 < cells.len() && visible(cells[anchor], cells[next + 1]) {
            next += 1;
        }
        out.push(cells[next]);
        anchor = next;
    }
    out
}

/// Incremental inflation of occupied cells: each cell counts the occupied cells
/// within a disk around it; it is blocked if this count is non-zero.
struct Inflation {
    nx: usize,
    nz: usize,
    occupied: Vec<bool>,
    count: Vec<u16>,
    disk: Vec<(i32, i32)>,
}

impl Inflation {
    fn new(grid: &OccupancyGrid, inflate_m: f32) -> Self {
        let (nx, nz) = grid.size();
        let r = (inflate_m / grid.cell_m()).ceil() as i32;
        let r2 = (inflate_m / grid.cell_m()).powi(2);
        let disk = (-r..=r)
            .flat_map(|dj| (-r..=r).map(move |di| (di, dj)))
            .filter(|&(di, dj)| (di * di + dj * dj) as f32 <= r2)
            .collect();
        Self { nx, nz, occupied: vec![false; nx * nz], count: vec![0; nx * nz], disk }
    }

    fn blocked(&self) -> Vec<bool> {
        self.count.iter().map(|&c| c > 0).collect()
    }

    /// Re-reads the map. Returns the cells whose blocked state may have changed.
    fn update(&mut self, grid: &OccupancyGrid) -> Vec<((usize, usize), bool)> {
        let mut out = Vec::new();
        // An occupied cell has been observed: the observed rectangle is enough.
        let (ri, rj) = grid.observed_region(0);
        for j in rj {
            for i in ri.clone() {
                let occ = grid.cell(i, j) == Cell::Occupied;
                let k = j * self.nx + i;
                if occ == self.occupied[k] {
                    continue;
                }
                self.occupied[k] = occ;
                for &(di, dj) in &self.disk {
                    let (x, z) = (i as i32 + di, j as i32 + dj);
                    if x < 0 || z < 0 || x as usize >= self.nx || z as usize >= self.nz {
                        continue;
                    }
                    let n = z as usize * self.nx + x as usize;
                    let was = self.count[n] > 0;
                    if occ {
                        self.count[n] += 1;
                    } else {
                        self.count[n] -= 1;
                    }
                    if was != (self.count[n] > 0) {
                        out.push(((x as usize, z as usize), self.count[n] > 0));
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests;
