//! Spatial index of obstacles: a horizontal grid of cells, each cell listing the
//! obstacles whose footprint touches it. A ray only tests the obstacles of the cells
//! it crosses, in order, and stops as soon as a hit is closer than the
//! next cell. Essential for cities (hundreds of buildings).

use glam::Vec3;

use crate::Obstacle;

/// Cell side, in m: about one building.
const CELL_M: f32 = 8.0;

#[derive(Debug, Clone, Default)]
pub(crate) struct ObstacleIndex {
    /// Minimum (x, z) corner covered.
    min: f32,
    n: usize,
    cells: Vec<Vec<u32>>,
    /// Number of indexed obstacles: if the world's list changed without going through
    /// the index, queries fall back to the brute-force walk.
    pub count: usize,
}

impl ObstacleIndex {
    /// Index of the square map of half-side `half`.
    pub fn build(obstacles: &[Obstacle], half: f32) -> Self {
        let n = ((2.0 * half) / CELL_M).ceil().max(1.0) as usize;
        let mut idx = Self { min: -half, n, cells: vec![Vec::new(); n * n], count: 0 };
        for o in obstacles {
            idx.insert(o);
        }
        idx
    }

    pub fn insert(&mut self, o: &Obstacle) {
        let id = self.count as u32;
        self.count += 1;
        let r = o.footprint_radius();
        let (i0, j0) = self.clamped(o.base.x - r, o.base.z - r);
        let (i1, j1) = self.clamped(o.base.x + r, o.base.z + r);
        for j in j0..=j1 {
            for i in i0..=i1 {
                self.cells[j * self.n + i].push(id);
            }
        }
    }

    fn clamped(&self, x: f32, z: f32) -> (usize, usize) {
        let f = |v: f32| (((v - self.min) / CELL_M).floor().max(0.0) as usize).min(self.n - 1);
        (f(x), f(z))
    }

    /// Cell containing (x, z), if it is on the map.
    fn cell(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        let i = ((x - self.min) / CELL_M).floor();
        let j = ((z - self.min) / CELL_M).floor();
        (i >= 0.0 && j >= 0.0 && (i as usize) < self.n && (j as usize) < self.n).then_some((i as usize, j as usize))
    }

    /// Obstacles whose footprint may touch the point (x, z).
    pub fn at(&self, x: f32, z: f32) -> &[u32] {
        match self.cell(x, z) {
            Some((i, j)) => &self.cells[j * self.n + i],
            None => &[],
        }
    }

    /// Obstacles whose footprint may touch the disk (x, z, r); duplicates possible.
    pub fn near(&self, x: f32, z: f32, r: f32) -> impl Iterator<Item = u32> + '_ {
        let (i0, j0) = self.clamped(x - r, z - r);
        let (i1, j1) = self.clamped(x + r, z + r);
        (j0..=j1).flat_map(move |j| (i0..=i1).flat_map(move |i| self.cells[j * self.n + i].iter().copied()))
    }

    /// Walks the cells crossed by the ray, in order (2D DDA). For each
    /// cell, `visit(ids, t_exit)` receives its obstacles and the ray parameter at its
    /// exit; if it returns `true`, the walk stops. Returns `false` if the origin
    /// is off the map (the caller then tests everything).
    pub fn walk(&self, origin: Vec3, dir: Vec3, max: f32, mut visit: impl FnMut(&[u32], f32) -> bool) -> bool {
        let Some((mut i, mut j)) = self.cell(origin.x, origin.z) else { return false };
        let step = |d: f32| if d > 0.0 { 1i64 } else { -1 };
        let (si, sj) = (step(dir.x), step(dir.z));
        // Parameter of the next edge in x and in z, and step between two edges.
        let next_edge = |p: f32, k: usize, s: i64| self.min + (k as f32 + (s > 0) as i32 as f32) * CELL_M - p;
        let mut t_x = if dir.x.abs() < 1e-9 { f32::INFINITY } else { next_edge(origin.x, i, si) / dir.x };
        let mut t_z = if dir.z.abs() < 1e-9 { f32::INFINITY } else { next_edge(origin.z, j, sj) / dir.z };
        let dt_x = if dir.x.abs() < 1e-9 { f32::INFINITY } else { CELL_M / dir.x.abs() };
        let dt_z = if dir.z.abs() < 1e-9 { f32::INFINITY } else { CELL_M / dir.z.abs() };
        loop {
            let t_exit = t_x.min(t_z);
            if visit(&self.cells[j * self.n + i], t_exit) || t_exit > max {
                return true;
            }
            if t_x < t_z {
                let ni = i as i64 + si;
                if ni < 0 || ni as usize >= self.n {
                    return true;
                }
                i = ni as usize;
                t_x += dt_x;
            } else {
                let nj = j as i64 + sj;
                if nj < 0 || nj as usize >= self.n {
                    return true;
                }
                j = nj as usize;
                t_z += dt_z;
            }
        }
    }
}
