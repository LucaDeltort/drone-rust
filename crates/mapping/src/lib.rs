//! Online mapping: 2.5D occupancy grid in log-odds.
//!
//! The map is a horizontal slab of the world, between `y_lo` and `y_hi` (absolute
//! heights): a cell is occupied if matter cuts the slab above
//! it. Each sensor feeds it through its inverse model:
//! - lidar and depth: free along the ray within the slab, occupied at the hit
//!   if it falls within the slab;
//! - sonar: a fan of rays within the cone, with a reduced weight, since we do not know
//!   where in the cone the echo comes from.
//!
//! Pure Rust, no Bevy and no access to the ground truth.

use glam::{Quat, Vec2, Vec3};
use sensors::{Data, Measurement};

mod matching;
mod voxel;
pub use matching::{MatchConfig, ScanMatch};
pub use voxel::{CHUNK, Key, VoxelCounts, VoxelMap, chunk_of};

/// Readable state of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Unknown,
    Free,
    Occupied,
}

/// Log-odds fusion settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogOdds {
    /// Increment for a hit (≈ p = 0.7).
    pub hit: f32,
    /// Increment for a free pass (≈ p = 0.4).
    pub miss: f32,
    /// Bounds: the map can always change its mind.
    pub min: f32,
    pub max: f32,
    /// Decision thresholds (|l| beyond = known).
    pub occupied_above: f32,
    pub free_below: f32,
}

impl Default for LogOdds {
    fn default() -> Self {
        Self {
            hit: 0.85,
            miss: -0.4,
            min: -2.0,
            max: 3.5,
            occupied_above: 0.6,
            free_below: -0.6,
        }
    }
}

/// A ray frees a cell if it passes within this fraction of its width from its
/// center.
const CENTER_PASS: f32 = 0.35;

/// Guard around a hit, in m: ≈ 3σ of the range sensors' noise.
const HIT_GUARD_M: f32 = 0.06;

/// Hits and free passes from which an unknown cell is ambiguous.
const AMBIGUOUS_MIN: u16 = 3;

/// A cell that returned at least this many optical echoes (lidar, depth: full-weight
/// rays), and at least one for every `MATTER_RATIO` passes, holds matter even if most
/// rays cross it: glass lets most beams through, and its few echoes would otherwise
/// be outvoted and the pane erased. Passes through it then count for
/// `MATTER_MISS_WEIGHT` only, and prove nothing beyond it: what lies behind an
/// established wall is only reached through glass or by a phantom echo (a lidar beam
/// bounced off metal, reading too far), and must not empty the inside of the building.
/// Sonar echoes do not count: a cone marks a whole arc, mostly empty, that later
/// passes must be able to clear.
const MATTER_HITS: u16 = 3;
const MATTER_RATIO: u16 = 4;
const MATTER_MISS_WEIGHT: f32 = 0.2;
/// A ray passing this much above a cell's highest echo does not cross its matter, in m.
const ABOVE_MATTER_M: f32 = 0.1;
/// An echo less than this beyond established matter comes from the same wall, in m.
const SAME_WALL_M: f32 = 1.0;

/// Weight of a sonar ray relative to a lidar ray.
const SONAR_WEIGHT: f32 = 0.3;
/// Number of rays representing the sonar cone in the map.
const SONAR_FAN: usize = 7;

#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    /// Minimum (x, z) corner of the map.
    origin: Vec2,
    cell_m: f32,
    nx: usize,
    nz: usize,
    pub y_lo: f32,
    pub y_hi: f32,
    pub params: LogOdds,
    logodds: Vec<f32>,
    /// Number of hits, free passes and optical (full-weight) hits received by each cell.
    evidence: Vec<[u16; 3]>,
    /// Mean position (x, z) of the hits in each cell: where the surface really is,
    /// finer than the cell, for scan matching.
    hit_mean: Vec<Vec2>,
    /// Height of the highest optical echo of each cell, in m.
    hit_top: Vec<f32>,
    /// Rectangle (i0, j0, i1, j1) of the cells observed so far: beyond it, everything is
    /// unknown, and computations on the map can stick to this rectangle.
    bounds: Option<[usize; 4]>,
    /// Do no-return rays free the ray up to max range?
    pub miss_as_free: bool,
}

impl OccupancyGrid {
    /// Square map of side `size_m` centered on the origin, slab `[y_lo, y_hi]`.
    pub fn new(size_m: f32, cell_m: f32, y_lo: f32, y_hi: f32) -> Self {
        assert!(cell_m > 0.0 && size_m > 0.0 && y_hi > y_lo);
        let n = (size_m / cell_m).ceil() as usize;
        Self {
            origin: Vec2::splat(-(n as f32) * cell_m / 2.0),
            cell_m,
            nx: n,
            nz: n,
            y_lo,
            y_hi,
            params: LogOdds::default(),
            logodds: vec![0.0; n * n],
            evidence: vec![[0; 3]; n * n],
            hit_mean: vec![Vec2::ZERO; n * n],
            hit_top: vec![f32::NEG_INFINITY; n * n],
            bounds: None,
            miss_as_free: true,
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.nx, self.nz)
    }

    pub fn cell_m(&self) -> f32 {
        self.cell_m
    }

    pub fn clear(&mut self) {
        self.logodds.fill(0.0);
        self.evidence.fill([0; 3]);
        self.hit_mean.fill(Vec2::ZERO);
        self.hit_top.fill(f32::NEG_INFINITY);
        self.bounds = None;
    }

    /// Cells to consider: the observed rectangle, grown by `margin` cells and
    /// clamped to the map, as (i, j) ranges. Empty if nothing is observed.
    pub fn observed_region(&self, margin: usize) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        match self.bounds {
            None => (0..0, 0..0),
            Some([i0, j0, i1, j1]) => (
                i0.saturating_sub(margin)..(i1 + margin + 1).min(self.nx),
                j0.saturating_sub(margin)..(j1 + margin + 1).min(self.nz),
            ),
        }
    }

    /// Indices of the cell containing (x, z), if any.
    pub fn index(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        let i = ((x - self.origin.x) / self.cell_m).floor();
        let j = ((z - self.origin.y) / self.cell_m).floor();
        (i >= 0.0 && j >= 0.0 && (i as usize) < self.nx && (j as usize) < self.nz).then_some((i as usize, j as usize))
    }

    /// Center (x, z) of cell (i, j).
    pub fn center(&self, i: usize, j: usize) -> Vec2 {
        self.origin + (Vec2::new(i as f32, j as f32) + 0.5) * self.cell_m
    }

    pub fn logodds(&self, i: usize, j: usize) -> f32 {
        self.logodds[j * self.nx + i]
    }

    /// Has the cell received at least one measurement?
    pub fn observed(&self, i: usize, j: usize) -> bool {
        self.logodds(i, j) != 0.0
    }

    /// Cell still unknown despite many contradictory measurements
    /// (hits and free passes): typically a hillside that the rays
    /// graze, half inside the slab. Looking at it again will teach nothing.
    pub fn ambiguous(&self, i: usize, j: usize) -> bool {
        let [hits, misses, _] = self.evidence[j * self.nx + i];
        hits >= AMBIGUOUS_MIN && misses >= AMBIGUOUS_MIN && self.cell(i, j) == Cell::Unknown
    }

    /// Where the obstacle of cell (i, j) is, within the cell: the mean of its
    /// hits, or its center if it has none.
    pub fn surface(&self, i: usize, j: usize) -> Vec2 {
        let k = j * self.nx + i;
        if self.evidence[k][0] > 0 { self.hit_mean[k] } else { self.center(i, j) }
    }

    pub fn cell(&self, i: usize, j: usize) -> Cell {
        let l = self.logodds(i, j);
        if l > self.params.occupied_above {
            Cell::Occupied
        } else if l < self.params.free_below {
            Cell::Free
        } else {
            Cell::Unknown
        }
    }

    fn add(&mut self, i: usize, j: usize, delta: f32) {
        let p = self.params;
        let k = j * self.nx + i;
        let l = &mut self.logodds[k];
        *l = (*l + delta).clamp(p.min, p.max);
        let e = &mut self.evidence[k][(delta < 0.0) as usize];
        *e = e.saturating_add(1);
        self.bounds = Some(match self.bounds {
            None => [i, j, i, j],
            Some([i0, j0, i1, j1]) => [i0.min(i), j0.min(j), i1.max(i), j1.max(j)],
        });
    }

    /// Integrates a measurement. Returns the number of cells touched.
    pub fn integrate(&mut self, m: &Measurement) -> usize {
        match &m.data {
            Data::Scan { beams, max_range } | Data::DepthImage { beams, max_range, .. } => beams
                .iter()
                .enumerate()
                // Invalid return (lost, too close): no information.
                .filter(|(_, b)| b.valid)
                .map(|(i, b)| {
                    let (origin, rotation) = m.beam_pose(i);
                    self.integrate_ray(origin, rotation * b.dir, b.range, *max_range, 1.0)
                })
                .sum(),
            Data::Echo { range, cone_half_angle, max_range } => {
                // Horizontal fan within the cone, around its axis.
                (0..SONAR_FAN)
                    .map(|k| {
                        let f = k as f32 / (SONAR_FAN - 1) as f32 * 2.0 - 1.0;
                        let dir = m.rotation * Quat::from_rotation_y(f * cone_half_angle) * Vec3::NEG_Z;
                        self.integrate_ray(m.origin, dir, *range, *max_range, SONAR_WEIGHT)
                    })
                    .sum()
            }
            // Navigation sensors: nothing to put in the map.
            Data::Imu { .. } | Data::Baro { .. } | Data::Gps { .. } => 0,
        }
    }

    /// Inverse model of a ray: free along the part within the slab,
    /// occupied at the hit if the hit is within the slab.
    pub fn integrate_ray(&mut self, origin: Vec3, dir: Vec3, range: Option<f32>, max_range: f32, weight: f32) -> usize {
        let end = match range {
            Some(r) => r,
            None if self.miss_as_free => max_range,
            None => return 0,
        };
        // Part of the ray within the slab [y_lo, y_hi].
        let (t0, t1) = if dir.y.abs() < 1e-6 {
            if (self.y_lo..=self.y_hi).contains(&origin.y) { (0.0, end) } else { return 0 }
        } else {
            let a = (self.y_lo - origin.y) / dir.y;
            let b = (self.y_hi - origin.y) / dir.y;
            (a.min(b).max(0.0), a.max(b).min(end))
        };

        // The hit is on the obstacle's surface: the cell containing the surface
        // contains matter. Push it by `HIT_GUARD_M` (≈ 3σ of the noise) along
        // the ray so that a hit measured slightly short still lands in the matter;
        // no more, otherwise the wall moves back by a cell and the cell containing it is
        // marked free.
        let hit_point = range.map(|r| origin + dir * r);
        let hit_cell = hit_point.and_then(|p| {
            let inside = p + dir * HIT_GUARD_M;
            if (self.y_lo..=self.y_hi).contains(&p.y) { self.index(inside.x, inside.z) } else { None }
        });

        // Just before the hit, the surface may be there (noise): nothing proven free.
        let t1 = match range {
            Some(r) => t1.min(r - HIT_GUARD_M),
            None => t1,
        };
        let mut touched = 0;
        let mut behind_matter = false;
        if t1 > t0 {
            // Step of a third of a cell: no crossed cell is skipped.
            let step = self.cell_m / 3.0;
            let mut last = None;
            let mut t = t0;
            while t <= t1 {
                let p = origin + dir * t;
                if let Some(c) = self.index(p.x, p.z)
                    && Some(c) != last
                    && Some(c) != hit_cell
                {
                    // A no-return ray crossing a known obstacle is very
                    // probably a dropout: it must not empty the inside of walls.
                    if range.is_none() && self.cell(c.0, c.1) == Cell::Occupied {
                        break;
                    }
                    last = Some(c);
                    let [_, misses, echoes] = self.evidence[c.1 * self.nx + c.0];
                    let matter = echoes >= MATTER_HITS && echoes.saturating_mul(MATTER_RATIO) >= misses;
                    // A ray that only clips the cell does not prove it is
                    // empty: it often grazes a wall occupying the rest of it (and the camera
                    // casts thousands of them). Only a pass near the center counts.
                    let to_center = self.center(c.0, c.1) - Vec2::new(origin.x, origin.z);
                    let flat = Vec2::new(dir.x, dir.z).normalize_or_zero();
                    let central = flat == Vec2::ZERO || flat.perp_dot(to_center).abs() <= self.cell_m * CENTER_PASS;
                    if central {
                        let w = if matter { weight * MATTER_MISS_WEIGHT } else { weight };
                        self.add(c.0, c.1, self.params.miss * w);
                        touched += 1;
                    }
                    // Above the highest echo (a slope filling only the bottom of
                    // the slab), the ray does not cross the matter. An echo just
                    // beyond is the same wall, seen from a slightly different pose.
                    let far_behind = range.is_none_or(|r| r - t > SAME_WALL_M);
                    if matter && far_behind && p.y <= self.hit_top[c.1 * self.nx + c.0] + ABOVE_MATTER_M {
                        behind_matter = true;
                        break;
                    }
                }
                t += step;
            }
        }
        // An echo from behind an established wall (through glass, or a phantom) is
        // left out too: a phantom would put matter in the next street.
        if let (Some((i, j)), Some(p), false) = (hit_cell, hit_point, behind_matter) {
            self.add(i, j, self.params.hit * weight);
            if weight >= 1.0 {
                let e = &mut self.evidence[j * self.nx + i][2];
                *e = e.saturating_add(1);
                let top = &mut self.hit_top[j * self.nx + i];
                *top = top.max(p.y);
            }
            // Mean of the hits: an established surface barely moves, so the map
            // does not follow a drifting pose.
            let k = j * self.nx + i;
            let n = self.evidence[k][0] as f32;
            let mean = self.hit_mean[k];
            self.hit_mean[k] = mean + (Vec2::new(p.x, p.z) - mean) / n.max(1.0);
            touched += 1;
        }
        touched
    }

    /// Forces the state of a cell, for a map projected from another one (the 3D
    /// map). The evidence counts are left alone.
    pub fn set_cell(&mut self, i: usize, j: usize, cell: Cell) {
        let target = match cell {
            Cell::Unknown => 0.0,
            Cell::Free => self.params.min,
            Cell::Occupied => self.params.max,
        };
        let k = j * self.nx + i;
        if self.logodds[k] == target {
            return;
        }
        self.logodds[k] = target;
        if cell != Cell::Unknown {
            self.bounds = Some(match self.bounds {
                None => [i, j, i, j],
                Some([i0, j0, i1, j1]) => [i0.min(i), j0.min(j), i1.max(i), j1.max(j)],
            });
        }
    }

    /// The drone's body occupies this disk: there is no obstacle there (otherwise it would
    /// touch it). Proprioceptive measurement, no sensor: mark these cells free.
    /// Also used to set the take-off area, assumed clear.
    pub fn mark_free_disk(&mut self, center: Vec2, radius: f32) {
        let r = (radius / self.cell_m).ceil() as i64;
        let Some((ci, cj)) = self.index(center.x, center.y) else { return };
        for dj in -r..=r {
            for di in -r..=r {
                let (i, j) = (ci as i64 + di, cj as i64 + dj);
                if i < 0 || j < 0 || i as usize >= self.nx || j as usize >= self.nz {
                    continue;
                }
                let (i, j) = (i as usize, j as usize);
                // Only cells entirely inside the disk: a straddling cell may
                // contain the corner of a neighbouring obstacle.
                let far_corner = self.center(i, j).distance(center) + self.cell_m * std::f32::consts::FRAC_1_SQRT_2;
                if far_corner <= radius && self.logodds(i, j) > self.params.min {
                    let delta = self.params.min - self.logodds(i, j);
                    self.add(i, j, delta);
                }
            }
        }
    }

    /// Counts the cells by state.
    pub fn counts(&self) -> Counts {
        let mut c = Counts::default();
        for j in 0..self.nz {
            for i in 0..self.nx {
                match self.cell(i, j) {
                    Cell::Unknown => c.unknown += 1,
                    Cell::Free => c.free += 1,
                    Cell::Occupied => c.occupied += 1,
                }
            }
        }
        c
    }

    /// Consistency with the ground truth, which the map does not know: `truth(x, z)`
    /// returns (matter, empty): does the column contain
    /// matter, empty space, or both (partial cell, correct in both cases)?
    ///
    /// A cell whose outline cuts an obstacle is ambiguous; we therefore tolerate
    /// one cell of error (the truth is sampled over the 3×3 neighbourhood).
    pub fn agreement(&self, truth: impl Fn(f32, f32) -> (bool, bool)) -> Agreement {
        let mut a = Agreement::default();
        let h = self.cell_m;
        for j in 0..self.nz {
            for i in 0..self.nx {
                let state = self.cell(i, j);
                if state == Cell::Unknown {
                    continue;
                }
                let c = self.center(i, j);
                let offsets = [-h, 0.0, h];
                let (mut any_occ, mut any_free) = (false, false);
                for dx in offsets {
                    for dz in offsets {
                        let (matter, space) = truth(c.x + dx, c.y + dz);
                        any_occ |= matter;
                        any_free |= space;
                    }
                }
                match state {
                    Cell::Occupied => {
                        a.occupied += 1;
                        a.occupied_ok += any_occ as usize;
                    }
                    Cell::Free => {
                        a.free += 1;
                        a.free_ok += any_free as usize;
                    }
                    Cell::Unknown => unreachable!(),
                }
            }
        }
        a
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub unknown: usize,
    pub free: usize,
    pub occupied: usize,
}

/// Share of known cells matching the real world.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Agreement {
    pub occupied: usize,
    pub occupied_ok: usize,
    pub free: usize,
    pub free_ok: usize,
}

impl Agreement {
    /// Share of "occupied" cells really occupied (1 if none).
    pub fn occupied_ratio(&self) -> f32 {
        ratio(self.occupied_ok, self.occupied)
    }
    /// Share of "free" cells really free (1 if none).
    pub fn free_ratio(&self) -> f32 {
        ratio(self.free_ok, self.free)
    }
}

fn ratio(ok: usize, total: usize) -> f32 {
    if total == 0 { 1.0 } else { ok as f32 / total as f32 }
}

#[cfg(test)]
mod tests;
