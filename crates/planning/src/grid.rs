//! Configuration space and shortest paths on the occupancy grid.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::Vec2;
use mapping::{Cell, OccupancyGrid};

/// Cells where the drone's center may go: known free, more than
/// `inflate_m` from any occupied cell and more than `unknown_margin_m` from
/// the unknown (which may hide an obstacle). The unknown is never traversable.
pub struct CSpace {
    pub nx: usize,
    pub nz: usize,
    traversable: Vec<bool>,
    /// Known free, even within a margin: only used to leave the margin.
    free: Vec<bool>,
}

/// Cost factor of a margin cell: leave it by the shortest way.
const MARGIN_COST: f32 = 10.0;

impl CSpace {
    pub fn build(grid: &OccupancyGrid, inflate_m: f32, unknown_margin_m: f32) -> Self {
        let (nx, nz) = grid.size();
        let mut traversable = vec![false; nx * nz];
        let mut occupied = Vec::new();
        // Outside the observed rectangle, everything is unknown: neither free nor occupied.
        let (ri, rj) = grid.observed_region(0);
        for j in rj.clone() {
            for i in ri.clone() {
                match grid.cell(i, j) {
                    Cell::Free => traversable[j * nx + i] = true,
                    Cell::Occupied => occupied.push((i, j)),
                    Cell::Unknown => {}
                }
            }
        }
        let free = traversable.clone();
        // Margin around the never-observed unknown (cells seen only once,
        // still "unknown", do not count). Any never-observed cell close
        // to a free cell has, on the segment joining them, a never-observed cell
        // next to an observed one, at least as close: blocking a disk
        // around this edge of the unknown is enough, and much shorter than the free area.
        let observed = |i: usize, j: usize| grid.observed(i, j);
        let (ei, ej) = grid.observed_region(1);
        for j in ej {
            for i in ei.clone() {
                if observed(i, j) {
                    continue;
                }
                let edge = [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)].iter().any(|&(di, dj)| {
                    let (x, z) = (i as i64 + di, j as i64 + dj);
                    x >= 0 && z >= 0 && (x as usize) < nx && (z as usize) < nz && observed(x as usize, z as usize)
                });
                if edge {
                    block_disk(&mut traversable, (nx, nz), (i, j), unknown_margin_m / grid.cell_m());
                }
            }
        }
        // Obstacle inflation by a disk of radius `inflate_m`.
        for (i, j) in occupied {
            block_disk(&mut traversable, (nx, nz), (i, j), inflate_m / grid.cell_m());
        }
        Self { nx, nz, traversable, free }
    }

    pub fn ok(&self, i: usize, j: usize) -> bool {
        self.traversable[j * self.nx + i]
    }

    pub fn free(&self, i: usize, j: usize) -> bool {
        self.free[j * self.nx + i]
    }

    /// Traversable straight line between two cells (sampled every quarter cell).
    pub fn line_of_sight(&self, a: (usize, usize), b: (usize, usize)) -> bool {
        let (ax, az) = (a.0 as f32, a.1 as f32);
        let (bx, bz) = (b.0 as f32, b.1 as f32);
        let n = (((bx - ax).abs().max((bz - az).abs())) * 4.0).ceil().max(1.0) as usize;
        (0..=n).all(|k| {
            let t = k as f32 / n as f32;
            let (x, z) = (ax + (bx - ax) * t, az + (bz - az) * t);
            self.ok(x.round() as usize, z.round() as usize)
        })
    }
}

/// Blocks the cells within `r` cells of `(i, j)`.
fn block_disk(cells: &mut [bool], (nx, nz): (usize, usize), (i, j): (usize, usize), r: f32) {
    let ri = r.ceil() as i64;
    let r2 = r * r;
    for dj in -ri..=ri {
        for di in -ri..=ri {
            if (di * di + dj * dj) as f32 > r2 {
                continue;
            }
            let (x, z) = (i as i64 + di, j as i64 + dj);
            if x >= 0 && z >= 0 && (x as usize) < nx && (z as usize) < nz {
                cells[z as usize * nx + x as usize] = false;
            }
        }
    }
}

/// Distances from the drone to every reachable cell (Dijkstra, 8-connectivity).
pub struct DistanceField {
    nx: usize,
    cell_m: f32,
    dist: Vec<f32>,
    parent: Vec<u32>,
}

const NONE: u32 = u32::MAX;

#[derive(PartialEq)]
struct Item(f32, u32);
impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap on distance.
        o.0.total_cmp(&self.0)
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl DistanceField {
    /// The start cell is always accepted, even if it is not traversable
    /// (drone in the unknown at take-off, or slightly too close to a wall).
    pub fn from(cspace: &CSpace, cell_m: f32, start: (usize, usize)) -> Self {
        let (nx, nz) = (cspace.nx, cspace.nz);
        let mut dist = vec![f32::INFINITY; nx * nz];
        let mut parent = vec![NONE; nx * nz];
        let s = (start.1 * nx + start.0) as u32;
        dist[s as usize] = 0.0;
        let mut heap = BinaryHeap::from([Item(0.0, s)]);
        const STEPS: [(i64, i64, f32); 8] = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, std::f32::consts::SQRT_2),
            (1, -1, std::f32::consts::SQRT_2),
            (-1, 1, std::f32::consts::SQRT_2),
            (-1, -1, std::f32::consts::SQRT_2),
        ];
        while let Some(Item(d, k)) = heap.pop() {
            if d > dist[k as usize] {
                continue;
            }
            let (i, j) = ((k as usize % nx) as i64, (k as usize / nx) as i64);
            for (di, dj, w) in STEPS {
                let (x, z) = (i + di, j + dj);
                if x < 0 || z < 0 || x as usize >= nx || z as usize >= nz {
                    continue;
                }
                let (x, z) = (x as usize, z as usize);
                // Outside the margin, only traversable cells; inside the margin (drone
                // too close to a wall), it can leave through known free cells.
                let here_ok = cspace.ok(i as usize, j as usize);
                let pass = |a: usize, b: usize| cspace.ok(a, b) || !here_ok && cspace.free(a, b);
                // Diagonally, no corner cutting between two blocked cells.
                if !pass(x, z) || (di != 0 && dj != 0 && (!pass(x, j as usize) || !pass(i as usize, z))) {
                    continue;
                }
                let n = z * nx + x;
                let cost = if cspace.ok(x, z) { 1.0 } else { MARGIN_COST };
                let nd = d + w * cell_m * cost;
                if nd < dist[n] {
                    dist[n] = nd;
                    parent[n] = k;
                    heap.push(Item(nd, n as u32));
                }
            }
        }
        Self { nx, cell_m, dist, parent }
    }

    /// Travel cost to the cell (m, margin cells counting
    /// `MARGIN_COST` times), infinite if unreachable.
    pub fn distance(&self, i: usize, j: usize) -> f32 {
        self.dist[j * self.nx + i]
    }

    pub fn reachable(&self, i: usize, j: usize) -> bool {
        self.distance(i, j).is_finite()
    }

    /// Cells from the start to `goal`, or `None` if unreachable.
    pub fn path_to(&self, goal: (usize, usize)) -> Option<Vec<(usize, usize)>> {
        if !self.reachable(goal.0, goal.1) {
            return None;
        }
        let mut k = (goal.1 * self.nx + goal.0) as u32;
        let mut cells = vec![goal];
        while self.parent[k as usize] != NONE {
            k = self.parent[k as usize];
            cells.push((k as usize % self.nx, k as usize / self.nx));
        }
        cells.reverse();
        Some(cells)
    }

    pub fn cell_m(&self) -> f32 {
        self.cell_m
    }
}

/// Visibility smoothing: skip every intermediate point as long as the
/// straight line stays traversable. The first point (the drone) is not checked.
pub fn smooth(cspace: &CSpace, cells: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if cells.len() <= 2 {
        return cells.to_vec();
    }
    let mut out = vec![cells[0]];
    let mut anchor = 0;
    while anchor < cells.len() - 1 {
        // The farthest visible point from the anchor (at worst the next one).
        let mut next = anchor + 1;
        for k in (anchor + 2..cells.len()).rev() {
            let from = if anchor == 0 { cells[1] } else { cells[anchor] };
            if cspace.line_of_sight(from, cells[k]) {
                next = k;
                break;
            }
        }
        out.push(cells[next]);
        anchor = next;
    }
    out
}

/// Non-free cell near a point.
#[derive(Debug, Clone, Copy)]
pub struct Near {
    /// Distance from the point to the cell's edge, in m.
    pub distance: f32,
    /// Unit direction from the point to the cell.
    pub dir: Vec2,
    /// Occupied (otherwise unknown: maybe an obstacle).
    pub occupied: bool,
}

/// Non-free cells (occupied or unknown: the unknown may hide an obstacle) within
/// `max_m` of the point. Unknown within `body_m` is ignored: it is the
/// drone's own room (if there were matter there, it would already touch it).
pub fn obstacles_near(grid: &OccupancyGrid, p: Vec2, max_m: f32, body_m: f32) -> Vec<Near> {
    let Some((ci, cj)) = grid.index(p.x, p.y) else { return Vec::new() };
    let (nx, nz) = grid.size();
    let r = (max_m / grid.cell_m()).ceil() as i64 + 1;
    let mut out = Vec::new();
    for dj in -r..=r {
        for di in -r..=r {
            let (i, j) = (ci as i64 + di, cj as i64 + dj);
            if i < 0 || j < 0 || i as usize >= nx || j as usize >= nz {
                continue;
            }
            let cell = grid.cell(i as usize, j as usize);
            if cell != Cell::Free {
                let to = grid.center(i as usize, j as usize) - p;
                let d = (to.length() - grid.cell_m() * 0.5).max(0.0);
                let occupied = cell == Cell::Occupied;
                if d < max_m && (occupied || d >= body_m) {
                    out.push(Near { distance: d, dir: to.normalize_or_zero(), occupied });
                }
            }
        }
    }
    out
}

/// Cell centers (x, z).
pub fn to_world(grid: &OccupancyGrid, cells: &[(usize, usize)]) -> Vec<Vec2> {
    cells.iter().map(|&(i, j)| grid.center(i, j)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// 20 m map (0.5 m cells) entirely free, with a wall at x ≈ 2,
    /// from z = -6 to z = 6.
    fn walled() -> OccupancyGrid {
        let mut g = OccupancyGrid::new(20.0, 0.5, 0.0, 2.0);
        for j in 0..g.size().1 {
            for i in 0..g.size().0 {
                let c = g.center(i, j);
                let wall = (c.x - 2.25).abs() < 0.3 && c.y.abs() < 6.0;
                for _ in 0..4 {
                    // Vertical ray from above: free, or a hit within the slab.
                    let range = if wall { Some(1.5) } else { None };
                    g.integrate_ray(Vec3::new(c.x, 2.5, c.y), Vec3::NEG_Y, range, 3.0, 1.0);
                }
            }
        }
        g
    }

    #[test]
    fn inflation_blocks_cells_near_walls() {
        let g = walled();
        let cs = CSpace::build(&g, 0.6, 0.3);
        let (i, j) = g.index(2.25, 0.0).unwrap();
        assert!(!cs.ok(i, j));
        let (i, j) = g.index(1.6, 0.0).unwrap();
        assert!(!cs.ok(i, j), "0.6 m from the wall: blocked");
        let (i, j) = g.index(0.5, 0.0).unwrap();
        assert!(cs.ok(i, j));
    }

    #[test]
    fn path_goes_around_the_wall() {
        let g = walled();
        let cs = CSpace::build(&g, 0.6, 0.3);
        let start = g.index(0.0, 0.0).unwrap();
        let goal = g.index(5.0, 0.0).unwrap();
        let field = DistanceField::from(&cs, g.cell_m(), start);
        let path = field.path_to(goal).unwrap();
        // Going around a 12 m wall from its middle: much longer than 5 m.
        assert!(field.distance(goal.0, goal.1) > 12.0);
        assert!(path.iter().all(|&(i, j)| cs.ok(i, j) || (i, j) == start));
        let smooth = smooth(&cs, &path);
        assert!(smooth.len() < path.len() / 4, "{} -> {}", path.len(), smooth.len());
        for w in smooth.windows(2).skip(1) {
            assert!(cs.line_of_sight(w[0], w[1]));
        }
    }

    #[test]
    fn unknown_is_not_traversable() {
        let g = OccupancyGrid::new(10.0, 0.5, 0.0, 2.0);
        let cs = CSpace::build(&g, 0.5, 0.3);
        let start = g.index(0.0, 0.0).unwrap();
        let field = DistanceField::from(&cs, g.cell_m(), start);
        assert!(field.reachable(start.0, start.1));
        let other = g.index(1.0, 0.0).unwrap();
        assert!(!field.reachable(other.0, other.1));
    }
}
