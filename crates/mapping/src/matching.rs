//! Scan-to-map matching: aligns a range scan on the occupancy map to correct the
//! estimated pose (the "localization" half of a SLAM).
//!
//! Point-to-plane ICP, solved by Gauss-Newton:
//! - each scan endpoint is associated with the nearest obstacle surface of the
//!   map (the mean of the hits of an occupied cell, finer than the cell: a scan
//!   matched on the map it just built must not move);
//! - the surfaces around it give the local wall direction (principal axes), and
//!   only the distance across the wall counts: a scan never slides along a wall,
//!   so a corridor constrains the pose across, not along;
//! - around a pole or a corner, there is no dominant direction: point-to-point;
//! - the unknowns are a planar offset and a yaw rotation of the drone's body.
//!
//! Only the hits well within range and near an obstacle already on the map are
//! used. The map knows the obstacles up to the range of the previous scans: the far
//! hits of a scan taken a little further along land past the known end of a wall,
//! and would drag the pose backwards along the motion.
//!
//! The Hessian of the last iteration doubles as the information of the match, so
//! the returned covariance is large along the directions the scan does not see.

use glam::{Mat2, Mat3, Vec2, Vec3};
use sensors::{Data, Measurement};

use crate::{Cell, OccupancyGrid};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchConfig {
    /// At most this many scan endpoints are used (evenly subsampled).
    pub max_points: usize,
    /// Hits beyond this share of the sensor's max range are left out.
    pub max_range_share: f32,
    /// Gauss-Newton iterations (associations recomputed at each one).
    pub iterations: usize,
    /// An endpoint further than this from any obstacle surface is left out, in m.
    pub reach_m: f32,
    /// Reach of the first iteration, in m, shrunk to `reach_m` over the next ones:
    /// set it from the pose uncertainty to recover from a large drift.
    pub search_m: f32,
    /// Endpoints that must be associated for the match to count.
    pub min_support: usize,
    /// Share of the endpoints that must be associated.
    pub min_support_share: f32,
    /// Largest accepted correction: offset in m, rotation in rad.
    pub max_offset_m: f32,
    pub max_yaw: f32,
    /// Huber threshold of the residuals, in m: beyond, an association weighs less.
    pub robust_m: f32,
    /// Inflation of the covariance: neighbouring endpoints share their map errors,
    /// so they are far from independent.
    pub correlation: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            max_points: 400,
            max_range_share: 0.7,
            iterations: 10,
            reach_m: 0.5,
            search_m: 0.5,
            min_support: 25,
            min_support_share: 0.5,
            max_offset_m: 0.5,
            max_yaw: 5f32.to_radians(),
            robust_m: 0.05,
            correlation: 10.0,
        }
    }
}

/// Result of a match: the correction to apply to the body pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanMatch {
    /// Planar offset (x, z) of the body, in m.
    pub offset: Vec2,
    /// Yaw rotation of the body around its position, in rad.
    pub yaw: f32,
    /// Covariance of (x, z, yaw).
    pub covariance: Mat3,
    /// Endpoints used, and those associated with an obstacle.
    pub points: usize,
    pub support: usize,
}

/// Radius of the neighbourhood that gives the local shape of an obstacle, in cells.
const SHAPE_CELLS: i64 = 1;
/// Below this ratio of the principal variances, the obstacle is a line (a wall).
const LINE_RATIO: f32 = 0.03;

/// Association of an endpoint: a point of the obstacle and the directions that count.
struct Anchor {
    point: Vec2,
    /// One normal for a wall, two axes for a pole or a corner.
    normals: [Option<Vec2>; 2],
}

impl OccupancyGrid {
    fn occupied_at(&self, i: i64, j: i64) -> bool {
        i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.nz && self.cell(i as usize, j as usize) == Cell::Occupied
    }

    /// Nearest obstacle surface within `reach` of `p`, and the local shape of the
    /// obstacle there.
    fn anchor(&self, p: Vec2, reach: f32) -> Option<Anchor> {
        let r = (reach / self.cell_m).ceil() as i64;
        let u = (p - self.origin) / self.cell_m;
        let (ci, cj) = (u.x.floor() as i64, u.y.floor() as i64);
        let mut best: Option<(f32, (i64, i64))> = None;
        for j in cj - r..=cj + r {
            for i in ci - r..=ci + r {
                if !self.occupied_at(i, j) {
                    continue;
                }
                let d2 = (p - self.surface(i as usize, j as usize)).length_squared();
                if d2 <= reach * reach && best.is_none_or(|(b, _)| d2 < b) {
                    best = Some((d2, (i, j)));
                }
            }
        }
        let (_, (bi, bj)) = best?;
        let nearest = self.surface(bi as usize, bj as usize);

        // Principal axes of the surfaces around it.
        let mut pts = [Vec2::ZERO; 9];
        let mut n = 0;
        for j in bj - SHAPE_CELLS..=bj + SHAPE_CELLS {
            for i in bi - SHAPE_CELLS..=bi + SHAPE_CELLS {
                if self.occupied_at(i, j) {
                    pts[n] = self.surface(i as usize, j as usize);
                    n += 1;
                }
            }
        }
        if n >= 3 {
            let mean = pts[..n].iter().copied().sum::<Vec2>() / n as f32;
            let mut c = Mat2::ZERO;
            for q in &pts[..n] {
                let d = *q - mean;
                c += Mat2::from_cols(d * d.x, d * d.y);
            }
            let (lo, hi, normal) = smallest_axis(c);
            if lo <= LINE_RATIO * hi {
                // A wall: the line fitted through the surfaces.
                return Some(Anchor { point: mean, normals: [Some(normal), None] });
            }
        }
        Some(Anchor { point: nearest, normals: [Some(Vec2::X), Some(Vec2::Y)] })
    }

    /// Aligns the scan `m` on the map. `m` must be expressed from the estimated
    /// pose (see [`Measurement::reposed`]) and `center` is the estimated body
    /// position (x, z), the center of the rotation. `None` if the scan has too
    /// little overlap with the known obstacles, or if the correction is implausible.
    pub fn match_scan(&self, m: &Measurement, center: Vec2, cfg: &MatchConfig) -> Option<ScanMatch> {
        let points = self.scan_endpoints(m, cfg);
        if points.len() < cfg.min_support {
            return None;
        }
        // Relative to the center, so that the rotation is about the body.
        let rel: Vec<Vec2> = points.iter().map(|p| *p - center).collect();

        let mut xi = Vec3::ZERO;
        let mut reach = cfg.search_m.max(cfg.reach_m);
        for _ in 0..cfg.iterations {
            let eq = self.normal_equations(&rel, center, xi, reach, cfg);
            reach = (reach * 0.6).max(cfg.reach_m);
            if eq.support < cfg.min_support {
                return None;
            }
            // Light damping: a direction the scan does not see stays put.
            let damped = eq.h + Mat3::from_diagonal(Vec3::splat(1e-6 + trace(eq.h) * 1e-5));
            if damped.determinant().abs() < 1e-18 {
                return None;
            }
            let step = damped.inverse() * eq.b;
            xi += step;
            if step.truncate().length() < 1e-4 && step.z.abs() < 1e-5 {
                break;
            }
        }

        let eq = self.normal_equations(&rel, center, xi, cfg.reach_m, cfg);
        let n = rel.len();
        if eq.support < cfg.min_support || (eq.support as f32) < cfg.min_support_share * n as f32 {
            return None;
        }
        if Vec2::new(xi.x, xi.y).length() > cfg.max_offset_m || xi.z.abs() > cfg.max_yaw {
            return None;
        }
        // Covariance ≈ σ² H⁻¹, σ² from the residuals; a floor on H keeps the
        // unseen directions finite but huge.
        let sigma2 = (eq.residual_sq / eq.weight.max(1.0)).max(1e-4) * cfg.correlation;
        let regular = eq.h + Mat3::from_diagonal(Vec3::splat(1e-3));
        let covariance = regular.inverse() * sigma2;
        Some(ScanMatch { offset: Vec2::new(xi.x, xi.y), yaw: xi.z, covariance, points: n, support: eq.support })
    }

    /// Gauss-Newton normal equations of the point-to-plane residuals at `xi`.
    fn normal_equations(&self, rel: &[Vec2], center: Vec2, xi: Vec3, reach: f32, cfg: &MatchConfig) -> Normal {
        let (s, c) = xi.z.sin_cos();
        let mut eq = Normal { h: Mat3::ZERO, b: Vec3::ZERO, residual_sq: 0.0, weight: 0.0, support: 0 };
        for &q in rel {
            // p = center + R(yaw)·q + offset, and ∂p/∂yaw.
            let p = center + Vec2::new(q.x * c + q.y * s, -q.x * s + q.y * c) + Vec2::new(xi.x, xi.y);
            let dp_dyaw = Vec2::new(-q.x * s + q.y * c, -q.x * c - q.y * s);
            let Some(anchor) = self.anchor(p, reach) else { continue };
            eq.support += 1;
            for n in anchor.normals.into_iter().flatten() {
                let r = n.dot(p - anchor.point);
                let j = Vec3::new(n.x, n.y, n.dot(dp_dyaw));
                // Huber weight: a wrong association weighs little.
                let w = if r.abs() <= cfg.robust_m { 1.0 } else { cfg.robust_m / r.abs() };
                eq.h += outer(j, j) * w;
                eq.b -= j * (w * r);
                eq.residual_sq += w * r * r;
                eq.weight += w;
            }
        }
        eq
    }

    /// Hits of the scan within the slab, well within range and near a known
    /// obstacle, as (x, z), subsampled evenly to at most `max_points`.
    fn scan_endpoints(&self, m: &Measurement, cfg: &MatchConfig) -> Vec<Vec2> {
        let (beams, max_range) = match &m.data {
            Data::Scan { beams, max_range } | Data::DepthImage { beams, max_range, .. } => (beams, *max_range),
            _ => return Vec::new(),
        };
        let hits: Vec<Vec2> = beams
            .iter()
            .enumerate()
            .filter(|(_, b)| b.valid)
            .filter_map(|(i, b)| {
                let (origin, rotation) = m.beam_pose(i);
                b.range.filter(|r| *r <= cfg.max_range_share * max_range).map(|r| origin + rotation * b.dir * r)
            })
            .filter(|p| (self.y_lo..=self.y_hi).contains(&p.y))
            .map(|p| Vec2::new(p.x, p.z))
            .filter(|p| self.near_occupied(*p, cfg.search_m.max(cfg.reach_m)))
            .collect();
        if hits.len() <= cfg.max_points {
            return hits;
        }
        let stride = hits.len() as f32 / cfg.max_points as f32;
        (0..cfg.max_points).map(|k| hits[(k as f32 * stride) as usize]).collect()
    }

    /// Is there an occupied cell within `reach` of (x, z)?
    fn near_occupied(&self, p: Vec2, reach: f32) -> bool {
        let Some((ci, cj)) = self.index(p.x, p.y) else { return false };
        let r = (reach / self.cell_m).ceil() as usize;
        (cj.saturating_sub(r)..=(cj + r).min(self.nz - 1))
            .any(|j| (ci.saturating_sub(r)..=(ci + r).min(self.nx - 1)).any(|i| self.cell(i, j) == Cell::Occupied))
    }
}

struct Normal {
    h: Mat3,
    b: Vec3,
    residual_sq: f32,
    weight: f32,
    support: usize,
}

/// Eigenvalues (small, large) of a symmetric 2×2 matrix, and the unit eigenvector
/// of the small one.
fn smallest_axis(c: Mat2) -> (f32, f32, Vec2) {
    let (a, b, d) = (c.x_axis.x, c.x_axis.y, c.y_axis.y);
    let mean = (a + d) / 2.0;
    let spread = (((a - d) / 2.0).powi(2) + b * b).sqrt();
    let (lo, hi) = (mean - spread, mean + spread);
    // (C - lo·I) v = 0.
    let v = if b.abs() > 1e-9 { Vec2::new(b, lo - a) } else if a <= d { Vec2::X } else { Vec2::Y };
    (lo, hi, v.normalize_or(Vec2::X))
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn trace(m: Mat3) -> f32 {
    m.x_axis.x + m.y_axis.y + m.z_axis.z
}
