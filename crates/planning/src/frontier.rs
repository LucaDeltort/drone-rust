//! Frontiers (Yamauchi): free cells next to an unknown cell.

use std::collections::{HashSet, VecDeque};

use glam::Vec2;
use mapping::{Cell, OccupancyGrid};

/// A connected group of frontier cells.
#[derive(Debug, Clone)]
pub struct Frontier {
    pub cells: Vec<(usize, usize)>,
    pub centroid: Vec2,
}

const N4: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const N8: [(i64, i64); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

fn neighbors(
    (i, j): (usize, usize),
    (nx, nz): (usize, usize),
    offsets: &'static [(i64, i64)],
) -> impl Iterator<Item = (usize, usize)> {
    offsets.iter().filter_map(move |&(di, dj)| {
        let (x, z) = (i as i64 + di, j as i64 + dj);
        (x >= 0 && z >= 0 && (x as usize) < nx && (z as usize) < nz).then_some((x as usize, z as usize))
    })
}

/// Free cells touching the unknown, grouped by 8-connectivity. Ambiguous
/// unknown (already well looked at, contradictory measurements) does not count. Groups
/// smaller than `min_cells` cells (noise) are ignored. A frontier can be
/// against a wall: the drone does not go there, it gets close (see the explorer).
///
/// `ignored`: cells already targeted without success (unobservable), excluded.
pub fn detect(grid: &OccupancyGrid, min_cells: usize, ignored: &HashSet<(usize, usize)>) -> Vec<Frontier> {
    let size = grid.size();
    let is_frontier = |c: (usize, usize)| {
        !ignored.contains(&c)
            && grid.cell(c.0, c.1) == Cell::Free
            && neighbors(c, size, &N4).any(|n| grid.cell(n.0, n.1) == Cell::Unknown && !grid.ambiguous(n.0, n.1))
    };
    let mut seen = vec![false; size.0 * size.1];
    let mut out = Vec::new();
    // A frontier cell is free, hence observed: the observed rectangle is enough.
    let (ri, rj) = grid.observed_region(0);
    for j in rj {
        for i in ri.clone() {
            if seen[j * size.0 + i] || !is_frontier((i, j)) {
                continue;
            }
            // Breadth-first walk of the group.
            let mut cells = Vec::new();
            let mut queue = VecDeque::from([(i, j)]);
            seen[j * size.0 + i] = true;
            while let Some(c) = queue.pop_front() {
                cells.push(c);
                for n in neighbors(c, size, &N8) {
                    let k = n.1 * size.0 + n.0;
                    if !seen[k] && is_frontier(n) {
                        seen[k] = true;
                        queue.push_back(n);
                    }
                }
            }
            if cells.len() >= min_cells {
                let sum: Vec2 = cells.iter().map(|&(a, b)| grid.center(a, b)).sum();
                out.push(Frontier { centroid: sum / cells.len() as f32, cells });
            }
        }
    }
    out
}

/// Information gain from a cell: unknown area (m²) visible within a
/// given radius, occupied cells blocking the view.
pub fn information_gain(grid: &OccupancyGrid, from: (usize, usize), radius_m: f32) -> f32 {
    const RAYS: usize = 64;
    let origin = grid.center(from.0, from.1);
    let step = grid.cell_m();
    let mut unknown = HashSet::new();
    for k in 0..RAYS {
        let a = std::f32::consts::TAU * k as f32 / RAYS as f32;
        let dir = Vec2::new(a.cos(), a.sin());
        let mut d = step;
        while d <= radius_m {
            let p = origin + dir * d;
            let Some((i, j)) = grid.index(p.x, p.y) else { break };
            match grid.cell(i, j) {
                Cell::Occupied => break,
                Cell::Unknown if !grid.ambiguous(i, j) => {
                    unknown.insert((i, j));
                }
                Cell::Unknown => {}
                Cell::Free => {}
            }
            d += step;
        }
    }
    unknown.len() as f32 * step * step
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// 20 m map where only a 3 m disk around the origin is known free.
    fn disk() -> OccupancyGrid {
        let mut g = OccupancyGrid::new(20.0, 0.5, 0.0, 2.0);
        for _ in 0..3 {
            for k in 0..360 {
                let a = (k as f32).to_radians();
                g.integrate_ray(Vec3::new(0.01, 1.0, 0.01), Vec3::new(a.cos(), 0.0, a.sin()), None, 3.0, 1.0);
            }
        }
        g
    }

    #[test]
    fn known_disk_has_one_ring_frontier() {
        let g = disk();
        let f = detect(&g, 4, &HashSet::new());
        assert_eq!(f.len(), 1);
        assert!(f[0].centroid.length() < 0.5, "centered ring: {}", f[0].centroid);
        for &(i, j) in &f[0].cells {
            let r = g.center(i, j).length();
            assert!((2.0..=3.3).contains(&r), "frontier at {r} m");
        }
    }

    #[test]
    fn gain_is_larger_at_the_edge_than_at_the_center() {
        let g = disk();
        let center = g.index(0.0, 0.0).unwrap();
        let edge = g.index(2.6, 0.0).unwrap();
        let (gc, ge) = (information_gain(&g, center, 5.0), information_gain(&g, edge, 5.0));
        assert!(ge > gc, "edge {ge} m², center {gc} m²");
    }

    #[test]
    fn fully_known_map_has_no_frontier() {
        let mut g = OccupancyGrid::new(6.0, 0.5, 0.0, 2.0);
        for j in 0..g.size().1 {
            for i in 0..g.size().0 {
                let c = g.center(i, j);
                for _ in 0..3 {
                    g.integrate_ray(Vec3::new(c.x, 1.5, c.y), Vec3::NEG_Y, None, 1.0, 1.0);
                }
            }
        }
        assert!(detect(&g, 1, &HashSet::new()).is_empty());
    }
}
