//! Inspection of one structure: the drone flies to it, finds it in its own 3D
//! map, circles it at several heights facing it, sweeps over its roof, then
//! comes back. Nothing else on the map is visited.
//!
//! - **Detection.** From a point given by the user, the structure is the group of
//!   voxel columns standing out of the local ground, connected to that point. Its
//!   footprint (bounding box), ground and top come from the drone's map, never
//!   from the real world.
//! - **Plan.** Rings around the footprint at `standoff_m`: one low (`low_agl_m`
//!   above the ground), one at mid-height for a tall structure; then two passes
//!   over the roof, `roof_clearance_m` above it.
//! - **Flight.** Transits (to the structure, back home) use the A-to-B navigator.
//!   Along the plan, the drone faces the structure; a waypoint that turns out to be
//!   inside something (a tree on the ring) is skipped, and so is one it cannot
//!   reach for `stuck_s`.

use rustc_hash::FxHashSet as HashSet;
use std::collections::VecDeque;

use drone_spec::Airframe;
use glam::{IVec2, Vec2, Vec3};
use mapping::{OccupancyGrid, VoxelMap};
use world::drone::{Command, DroneState};

use crate::control::{self, wrap};
use crate::{NavStatus, Navigator, NavigatorConfig};

/// A structure found in the voxel map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// Center of the footprint (x, z).
    pub center: Vec2,
    /// Half extents of the footprint (axis-aligned), in m.
    pub half: Vec2,
    /// Ground around it, and its top (absolute heights), in m.
    pub ground: f32,
    pub top: f32,
    /// Number of voxel columns it covers.
    pub columns: usize,
}

/// A column belongs to a structure if it stands this much above the local ground, in m.
const MIN_HEIGHT_M: f32 = 1.0;
/// Radius of the neighbourhood giving a column's local ground, in columns.
const LOCAL_GROUND: i32 = 3;
/// A structure is searched for within this distance of the user's point, in m.
const SEED_M: f32 = 3.0;
/// Rate of change of the velocity asked along the plan, in m/s².
const SETPOINT_ACCEL: f32 = 1.5;
/// An intermediate waypoint is passed once the drone is this close, in m:
/// it heads for the next one without slowing down.
const PASS_M: f32 = 1.2;

/// Period of the re-detection during the survey, in s.
const REDETECT_S: f32 = 1.0;
/// Growth of the footprint or of the top that triggers a new plan, in m.
const GROWTH_M: f32 = 0.8;
/// Bound on the size of a structure, in columns.
const MAX_COLUMNS: usize = 20_000;

fn local_ground(voxels: &VoxelMap, c: IVec2) -> Option<f32> {
    (-LOCAL_GROUND..=LOCAL_GROUND)
        .flat_map(|dz| (-LOCAL_GROUND..=LOCAL_GROUND).map(move |dx| c + IVec2::new(dx, dz)))
        .filter_map(|n| voxels.column_floor(n))
        .reduce(f32::min)
}

fn stands_out(voxels: &VoxelMap, c: IVec2) -> bool {
    match (voxels.column_top(c), local_ground(voxels, c)) {
        (Some(top), Some(ground)) => top - ground > MIN_HEIGHT_M,
        _ => false,
    }
}

/// The structure around `p` (within `search_m`), if the map shows one.
pub fn detect_target(voxels: &VoxelMap, p: Vec2, search_m: f32) -> Option<Target> {
    let v = voxels.voxel_m();
    let mid = |c: IVec2| (c.as_vec2() + 0.5) * v;
    let c0 = voxels.column_key(p.x, p.y);
    let n = (SEED_M / v).ceil() as i32;
    let mut seen: HashSet<IVec2> = HashSet::default();
    let mut queue: VecDeque<IVec2> = (-n..=n)
        .flat_map(|dz| (-n..=n).map(move |dx| c0 + IVec2::new(dx, dz)))
        .filter(|&c| mid(c).distance(p) <= SEED_M && stands_out(voxels, c))
        .collect();
    seen.extend(queue.iter().copied());
    let mut cluster = Vec::new();
    while let Some(c) = queue.pop_front() {
        cluster.push(c);
        if cluster.len() >= MAX_COLUMNS {
            break;
        }
        for dz in -1..=1 {
            for dx in -1..=1 {
                let nb = c + IVec2::new(dx, dz);
                if seen.insert(nb) && mid(nb).distance(p) <= search_m && stands_out(voxels, nb) {
                    queue.push_back(nb);
                }
            }
        }
    }
    if cluster.is_empty() {
        return None;
    }
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    let (mut top, mut ground) = (f32::NEG_INFINITY, f32::INFINITY);
    for &c in &cluster {
        lo = lo.min(c.as_vec2() * v);
        hi = hi.max((c.as_vec2() + 1.0) * v);
        top = top.max(voxels.column_top(c).unwrap_or(top));
        ground = ground.min(local_ground(voxels, c).unwrap_or(ground));
    }
    Some(Target { center: (lo + hi) / 2.0, half: (hi - lo) / 2.0, ground, top, columns: cluster.len() })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InspectConfig {
    /// Distance from the footprint to the rings, in m.
    pub standoff_m: f32,
    /// Spacing of the waypoints along a ring, in m.
    pub spacing_m: f32,
    /// Speed along the plan, in m/s.
    pub speed_ms: f32,
    /// Height of the low ring above the ground, in m.
    pub low_agl_m: f32,
    /// Height of the roof passes above the top, in m.
    pub roof_clearance_m: f32,
    /// The structure is looked for once the drone is this close to the point, in m.
    pub detect_range_m: f32,
    /// How far from the point the structure may extend, in m.
    pub search_m: f32,
    /// Without progress towards a waypoint for this long, it is skipped, in s.
    pub stuck_s: f32,
    /// Come back to the take-off point at the end.
    pub return_home: bool,
}

impl Default for InspectConfig {
    fn default() -> Self {
        Self {
            standoff_m: 4.0,
            spacing_m: 2.0,
            speed_ms: 2.5,
            low_agl_m: 2.0,
            roof_clearance_m: 3.0,
            detect_range_m: 15.0,
            search_m: 25.0,
            stuck_s: 6.0,
            return_home: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub pos: Vec3,
    /// Point the drone faces.
    pub look_at: Vec2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Flying to the point, looking for the structure.
    Approach,
    /// Following the plan around and over it.
    Survey,
    /// Back to the take-off point.
    Return,
    Done,
    /// Reached the point without finding a structure there.
    NotFound,
}

/// A closed loop at `standoff` around the box of half extents `half`: straight
/// sides, quarter circles at the corners, sampled every `spacing`.
fn ring(center: Vec2, half: Vec2, standoff: f32, spacing: f32) -> Vec<Vec2> {
    let mut dense = Vec::new();
    let corners = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
    for (k, &(sx, sz)) in corners.iter().enumerate() {
        let corner = center + Vec2::new(sx * half.x, sz * half.y);
        // Arc from the side before this corner to the side after it.
        let a0 = std::f32::consts::FRAC_PI_2 * k as f32;
        for s in 0..=8 {
            let a = a0 + std::f32::consts::FRAC_PI_2 * s as f32 / 8.0;
            dense.push(corner + Vec2::new(a.cos(), a.sin()) * standoff);
        }
    }
    dense.push(dense[0]);
    // Resample at a constant spacing.
    let mut out = vec![dense[0]];
    let mut carry = 0.0;
    for w in dense.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let mut t = spacing - carry;
        while t <= len {
            out.push(a + (b - a) * (t / len));
            t += spacing;
        }
        carry = len - (t - spacing);
    }
    out
}

/// Waypoints for a target: rings, then roof passes. Each ring starts at the
/// point nearest `from` and closes on itself.
pub fn plan(t: &Target, cfg: &InspectConfig, from: Vec2) -> Vec<Waypoint> {
    let loop_ = ring(t.center, t.half, cfg.standoff_m, cfg.spacing_m);
    let n = loop_.len() - 1;
    let first = (0..n).min_by(|&a, &b| loop_[a].distance(from).total_cmp(&loop_[b].distance(from))).unwrap_or(0);
    let height = t.top - t.ground;
    let mut levels = vec![t.ground + cfg.low_agl_m];
    if height > 4.0 {
        levels.push(t.ground + 0.6 * height);
    }
    let mut out = Vec::new();
    for &y in &levels {
        for k in 0..=n {
            let p = loop_[(first + k) % n];
            out.push(Waypoint { pos: Vec3::new(p.x, y, p.y), look_at: t.center });
        }
    }
    // Roof: two passes along the long side, serpentine.
    let y = t.top + cfg.roof_clearance_m;
    let (long, short) = if t.half.x >= t.half.y { (Vec2::X, Vec2::Y) } else { (Vec2::Y, Vec2::X) };
    let (l, s) = (t.half.dot(long) + 2.0, t.half.dot(short) / 2.0);
    for (k, side) in [-1.0, 1.0].into_iter().enumerate() {
        let ends = [t.center + short * side * s - long * l, t.center + short * side * s + long * l];
        let ends = if k % 2 == 0 { ends } else { [ends[1], ends[0]] };
        for p in ends {
            out.push(Waypoint { pos: Vec3::new(p.x, y, p.y), look_at: t.center });
        }
    }
    out
}

pub struct Inspector {
    pub cfg: InspectConfig,
    /// The point the user asked to inspect (x, z).
    pub point: Vec2,
    /// Take-off point, to come back to.
    pub home: Vec2,
    pub target: Option<Target>,
    pub plan: Vec<Waypoint>,
    /// Next waypoint of the plan.
    pub next: usize,
    pub phase: Phase,
    /// Waypoints skipped (inside something, or out of reach).
    pub skipped: usize,
    nav: Option<Navigator>,
    /// Horizontal velocity asked at the last step: the next one only moves away
    /// from it at `SETPOINT_ACCEL`, so the airframe never snaps its tilt.
    v_cmd: Vec2,
    since_detect: f32,
    best: f32,
    since_progress: f32,
}

impl Inspector {
    pub fn new(cfg: InspectConfig, point: Vec2, home: Vec2) -> Self {
        Self {
            cfg,
            point,
            home,
            target: None,
            plan: Vec::new(),
            next: 0,
            phase: Phase::Approach,
            skipped: 0,
            nav: None,
            v_cmd: Vec2::ZERO,
            since_detect: f32::INFINITY,
            best: f32::INFINITY,
            since_progress: 0.0,
        }
    }

    /// One decision step: the command, and the altitude asked for (for the 3D
    /// flight guard; `None` = follow the surfaces).
    pub fn update(
        &mut self,
        grid: &OccupancyGrid,
        voxels: &VoxelMap,
        state: &DroneState,
        airframe: &Airframe,
        dt: f32,
    ) -> (Command, Option<f32>) {
        let pos = Vec2::new(state.position.x, state.position.z);
        match self.phase {
            Phase::Approach => {
                self.since_detect += dt;
                if self.since_detect >= 0.5 && pos.distance(self.point) < self.cfg.detect_range_m {
                    self.since_detect = 0.0;
                    if let Some(t) = detect_target(voxels, self.point, self.cfg.search_m) {
                        self.target = Some(t);
                        self.plan = plan(&t, &self.cfg, pos);
                        self.next = 0;
                        self.nav = None;
                        self.phase = Phase::Survey;
                        self.v_cmd = Vec2::new(state.velocity.x, state.velocity.z);
                        self.reset_progress();
                        return self.survey(grid, voxels, state, airframe, dt);
                    }
                }
                let (cmd, status) = self.transit(self.point, grid, state, airframe, dt);
                if status == NavStatus::Reached {
                    self.phase = Phase::NotFound;
                }
                (cmd, None)
            }
            Phase::Survey => {
                self.redetect(voxels, pos, dt);
                self.survey(grid, voxels, state, airframe, dt)
            }
            Phase::Return => {
                let (cmd, status) = self.transit(self.home, grid, state, airframe, dt);
                if status == NavStatus::Reached {
                    self.phase = Phase::Done;
                    self.nav = None;
                }
                (cmd, None)
            }
            Phase::Done | Phase::NotFound => (Command::default(), None),
        }
    }

    /// Share of the plan done, in [0, 1].
    pub fn progress(&self) -> f32 {
        match self.phase {
            Phase::Approach => 0.0,
            Phase::Survey => self.next as f32 / self.plan.len().max(1) as f32,
            _ => 1.0,
        }
    }

    /// Seen from one side at first, the structure grows as the drone goes
    /// around it: if its footprint grew, plan again from here.
    fn redetect(&mut self, voxels: &VoxelMap, pos: Vec2, dt: f32) {
        self.since_detect += dt;
        if self.since_detect < REDETECT_S {
            return;
        }
        self.since_detect = 0.0;
        let (Some(old), Some(new)) = (self.target, detect_target(voxels, self.point, self.cfg.search_m)) else {
            return;
        };
        let (old_lo, old_hi) = (old.center - old.half, old.center + old.half);
        let (new_lo, new_hi) = (new.center - new.half, new.center + new.half);
        let grew = (old_lo - new_lo).max_element() > GROWTH_M
            || (new_hi - old_hi).max_element() > GROWTH_M
            || new.top - old.top > GROWTH_M;
        if grew {
            // Keep the levels already done.
            let done_levels = self.plan[..self.next].windows(2).filter(|w| w[1].pos.y - w[0].pos.y > 0.5).count();
            self.target = Some(new);
            let mut plan = plan(&new, &self.cfg, pos);
            let level_starts: Vec<usize> = (0..plan.len())
                .filter(|&k| k == 0 || plan[k].pos.y - plan[k - 1].pos.y > 0.5)
                .collect();
            let skip = level_starts.get(done_levels).copied().unwrap_or(0);
            plan.drain(..skip);
            self.plan = plan;
            self.next = 0;
            self.reset_progress();
        }
    }

    fn reset_progress(&mut self) {
        self.best = f32::INFINITY;
        self.since_progress = 0.0;
    }

    fn transit(&mut self, goal: Vec2, grid: &OccupancyGrid, state: &DroneState, airframe: &Airframe, dt: f32) -> (Command, NavStatus) {
        let nav = self.nav.get_or_insert_with(|| Navigator::new(NavigatorConfig::new(airframe), state.position.y, goal));
        let cmd = nav.update(grid, state, airframe, dt);
        (cmd, nav.status)
    }

    /// Does something stand where the waypoint would put the drone?
    fn blocked(voxels: &VoxelMap, wp: &Waypoint, airframe: &Airframe) -> bool {
        let v = voxels.voxel_m();
        let r = airframe.radius + 0.5;
        let c0 = voxels.column_key(wp.pos.x, wp.pos.z);
        let n = (r / v).ceil() as i32;
        let p = Vec2::new(wp.pos.x, wp.pos.z);
        let (lo, hi) = (wp.pos.y - airframe.radius - 0.3, wp.pos.y + airframe.radius + 0.3);
        (-n..=n).flat_map(|dz| (-n..=n).map(move |dx| c0 + IVec2::new(dx, dz))).any(|c| {
            ((c.as_vec2() + 0.5) * v).distance(p) <= r + 0.71 * v && voxels.column_blocks(c, lo, hi)
        })
    }

    /// The waypoint moved away from the structure, as little as frees it.
    fn push_out(&self, voxels: &VoxelMap, wp: Waypoint, airframe: &Airframe) -> Option<Waypoint> {
        let t = self.target?;
        let p = Vec2::new(wp.pos.x, wp.pos.z);
        let nearest = p.clamp(t.center - t.half, t.center + t.half);
        let out = (p - nearest).normalize_or_zero();
        [1.5, 3.0].into_iter().find_map(|k| {
            let q = p + out * k;
            let moved = Waypoint { pos: Vec3::new(q.x, wp.pos.y, q.y), ..wp };
            (!Self::blocked(voxels, &moved, airframe)).then_some(moved)
        })
    }

    fn advance(&mut self) {
        self.next += 1;
        self.reset_progress();
        if self.next >= self.plan.len() {
            self.phase = if self.cfg.return_home { Phase::Return } else { Phase::Done };
            self.nav = None;
        }
    }

    fn survey(
        &mut self,
        grid: &OccupancyGrid,
        voxels: &VoxelMap,
        state: &DroneState,
        airframe: &Airframe,
        dt: f32,
    ) -> (Command, Option<f32>) {
        // A waypoint inside something (a tree on the ring): moved outwards if that
        // frees it, skipped otherwise.
        while self.phase == Phase::Survey && Self::blocked(voxels, &self.plan[self.next], airframe) {
            match self.push_out(voxels, self.plan[self.next], airframe) {
                Some(wp) => self.plan[self.next] = wp,
                None => {
                    self.skipped += 1;
                    self.advance();
                }
            }
        }
        if self.phase != Phase::Survey {
            return (Command::default(), None);
        }
        let wp = self.plan[self.next];
        let pos = Vec2::new(state.position.x, state.position.z);
        let to = Vec2::new(wp.pos.x, wp.pos.z) - pos;
        let (d, dy) = (to.length(), wp.pos.y - state.position.y);
        // The last waypoint of a level (the next one is higher) or of the plan is a
        // stop; the others are passed through.
        let stop = self.plan.get(self.next + 1).is_none_or(|n| (n.pos.y - wp.pos.y).abs() > 0.5);
        let reached = if stop { d < 0.7 && dy.abs() < 1.0 } else { d < PASS_M && dy.abs() < 1.0 };
        if reached {
            self.advance();
            return self.survey(grid, voxels, state, airframe, dt);
        }
        let remaining = d + dy.abs();
        if remaining < self.best - 0.3 {
            self.best = remaining;
            self.since_progress = 0.0;
        } else {
            self.since_progress += dt;
            if self.since_progress > self.cfg.stuck_s {
                self.skipped += 1;
                self.advance();
                return (Command::default(), None);
            }
        }
        // Cruise speed, braking (gently) only before a stop; climb or descend first
        // between levels.
        let mut speed = self.cfg.speed_ms;
        if stop {
            speed = speed.min((2.0 * SETPOINT_ACCEL * 0.8 * d).sqrt());
        }
        if dy.abs() > 1.5 {
            speed *= 0.3;
        }
        let wanted = to.normalize_or_zero() * speed;
        self.v_cmd += (wanted - self.v_cmd).clamp_length_max(SETPOINT_ACCEL * dt);
        let v2 = control::safe_velocity(grid, state, airframe, self.v_cmd);
        let look = wp.look_at - pos;
        let yaw_err = if look.length() > 0.5 { wrap(control::yaw_towards(look) - state.yaw) } else { 0.0 };
        let up = control::hold_altitude(wp.pos.y, state, airframe);
        (control::command(state, airframe, v2, yaw_err, up), Some(wp.pos.y))
    }
}

#[cfg(test)]
mod tests;
