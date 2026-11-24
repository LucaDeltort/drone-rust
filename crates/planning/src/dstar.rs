//! D* Lite (Koenig and Likhachev, 2002) on an 8-connected grid.
//!
//! The search starts from the goal and goes back towards the drone: `g(u)` is the cost from `u`
//! to the goal. When the drone moves, only the heuristic changes (`km` term);
//! when cells change (obstacle discovered), only the values that depend
//! on them are recomputed, instead of redoing the whole search.
//!
//! The caller says which cells are blocked; everything else, unknown included,
//! is traversable (optimistic planning).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Priority key, compared lexicographically. In `f64`: on the optimal
/// path, a cell's key equals the start's, and an `f32` rounding was enough to
/// stop the search one cell too early.
type Key = [f64; 2];

/// Tolerance of key comparisons for the search termination.
const EPS: f64 = 1e-6;

/// `a` strictly before `b`, within `EPS` on the first term.
fn before(a: &Key, b: &Key) -> bool {
    a[0] < b[0] - EPS || (a[0] - b[0]).abs() <= EPS && a[1] < b[1] - EPS
}

/// Queue entry (min-heap). An entry is stale if the key recorded for its
/// cell has changed since: it is skipped when popped (lazy deletion).
#[derive(PartialEq)]
struct Entry(Key, u32);
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        less(&o.0, &self.0)
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

fn less(a: &Key, b: &Key) -> Ordering {
    a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1]))
}

const STEPS: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

pub struct DStarLite {
    nx: usize,
    nz: usize,
    cell_m: f32,
    blocked: Vec<bool>,
    g: Vec<f64>,
    rhs: Vec<f64>,
    /// Key of the cell in the queue; `NaN` if it is not in it.
    queued: Vec<Key>,
    heap: BinaryHeap<Entry>,
    start: usize,
    goal: usize,
    /// Sum of the heuristics between successive starts: keeps old
    /// keys comparable after the drone moves.
    km: f64,
    /// Cells expanded since creation (measure of the compute cost).
    pub expanded: usize,
}

impl DStarLite {
    pub fn new(nx: usize, nz: usize, cell_m: f32, blocked: Vec<bool>, start: (usize, usize), goal: (usize, usize)) -> Self {
        assert_eq!(blocked.len(), nx * nz);
        let n = nx * nz;
        let mut d = Self {
            nx,
            nz,
            cell_m,
            blocked,
            g: vec![f64::INFINITY; n],
            rhs: vec![f64::INFINITY; n],
            queued: vec![[f64::NAN; 2]; n],
            heap: BinaryHeap::new(),
            start: start.1 * nx + start.0,
            goal: goal.1 * nx + goal.0,
            km: 0.0,
            expanded: 0,
        };
        d.rhs[d.goal] = 0.0;
        let k = d.key(d.goal);
        d.push(d.goal, k);
        d
    }

    pub fn is_blocked(&self, (i, j): (usize, usize)) -> bool {
        self.blocked[j * self.nx + i]
    }

    pub fn goal(&self) -> (usize, usize) {
        self.xy(self.goal)
    }

    /// Cost of the best path from start to goal (m), infinite if there is none.
    /// Valid after [`Self::compute`]: it is `rhs(start)`, computed from
    /// consistent neighbours (`g(start)` itself may lag behind).
    pub fn cost_to_goal(&self) -> f32 {
        self.rhs[self.start] as f32
    }

    fn xy(&self, k: usize) -> (usize, usize) {
        (k % self.nx, k / self.nx)
    }

    /// Octile distance: exact on an empty grid, hence admissible and consistent.
    fn h(&self, a: usize, b: usize) -> f64 {
        let (ax, az) = self.xy(a);
        let (bx, bz) = self.xy(b);
        let dx = ax.abs_diff(bx) as f64;
        let dz = az.abs_diff(bz) as f64;
        (dx.max(dz) + (std::f64::consts::SQRT_2 - 1.0) * dx.min(dz)) * self.cell_m as f64
    }

    fn key(&self, u: usize) -> Key {
        let m = self.g[u].min(self.rhs[u]);
        [m + self.h(self.start, u) + self.km, m]
    }

    fn push(&mut self, u: usize, k: Key) {
        self.queued[u] = k;
        self.heap.push(Entry(k, u as u32));
    }

    /// Neighbours of `u` and the (symmetric) edge cost, infinite if blocked.
    /// Diagonally, no corner cutting between two cells when one of them is blocked.
    fn edges(&self, u: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (i, j) = self.xy(u);
        STEPS.iter().filter_map(move |&(di, dj)| {
            let (x, z) = (i as i32 + di, j as i32 + dj);
            if x < 0 || z < 0 || x as usize >= self.nx || z as usize >= self.nz {
                return None;
            }
            let v = z as usize * self.nx + x as usize;
            let diag = di != 0 && dj != 0;
            let cut = diag
                && (self.blocked[j * self.nx + x as usize] || self.blocked[z as usize * self.nx + i]);
            let cost = if self.blocked[u] || self.blocked[v] || cut {
                f64::INFINITY
            } else if diag {
                std::f64::consts::SQRT_2 * self.cell_m as f64
            } else {
                self.cell_m as f64
            };
            Some((v, cost))
        })
    }

    fn update_vertex(&mut self, u: usize) {
        if u != self.goal {
            self.rhs[u] = self.edges(u).map(|(v, c)| c + self.g[v]).fold(f64::INFINITY, f64::min);
        }
        if self.g[u] != self.rhs[u] {
            let k = self.key(u);
            self.push(u, k);
        } else {
            self.queued[u] = [f64::NAN; 2];
        }
    }

    /// Smallest valid queue entry, without removing it.
    fn top(&mut self) -> Option<(Key, usize)> {
        while let Some(Entry(k, u)) = self.heap.peek() {
            let u = *u as usize;
            if self.queued[u] == *k {
                return Some((*k, u));
            }
            self.heap.pop();
        }
        None
    }

    /// The drone is now in cell `start`.
    pub fn move_start(&mut self, start: (usize, usize)) {
        let s = start.1 * self.nx + start.0;
        if s != self.start {
            self.km += self.h(self.start, s);
            self.start = s;
        }
    }

    /// Changes the state of cells. Returns the number of cells actually changed.
    pub fn set_blocked(&mut self, cells: impl IntoIterator<Item = ((usize, usize), bool)>) -> usize {
        let mut changed = Vec::new();
        for ((i, j), b) in cells {
            let k = j * self.nx + i;
            if self.blocked[k] != b {
                self.blocked[k] = b;
                changed.push(k);
            }
        }
        // Affected costs: edges of the cell and diagonals grazing it, all
        // between the cell and its neighbours or between two neighbours.
        for &k in &changed {
            self.update_vertex(k);
            let neighbors: Vec<usize> = self.edges(k).map(|(v, _)| v).collect();
            for v in neighbors {
                self.update_vertex(v);
            }
        }
        changed.len()
    }

    /// Updates the costs until the start's is exact.
    pub fn compute(&mut self) {
        while let Some((k_old, u)) = self.top() {
            let k_start = self.key(self.start);
            // Termination: consistent start (g = rhs), and no key left in the band
            // [.., k1(start) + EPS]. The heap compares exactly: two keys equal up to
            // rounding may come out in the wrong order of the second term; so we process
            // the whole band (a few more cells, with no effect on the result).
            if k_old[0] > k_start[0] + EPS && self.rhs[self.start] == self.g[self.start] {
                break;
            }
            self.expanded += 1;
            let k_new = self.key(u);
            if before(&k_old, &k_new) {
                // Key made stale by a drone move: update it.
                self.push(u, k_new);
                continue;
            }
            self.queued[u] = [f64::NAN; 2];
            self.heap.pop();
            let preds: Vec<usize> = self.edges(u).map(|(v, _)| v).collect();
            if self.g[u] > self.rhs[u] {
                // Improved cost: it becomes final, propagate to the neighbours.
                self.g[u] = self.rhs[u];
            } else {
                // Worsened cost: restart from infinity and recompute.
                self.g[u] = f64::INFINITY;
                self.update_vertex(u);
            }
            for v in preds {
                self.update_vertex(v);
            }
        }
    }

    /// Path from start to goal, descending the costs. `None` if there is none.
    ///
    /// Only descend through consistent cells (g = rhs): a stale cell
    /// outside the queue may exactly equal the cost of the true successor (the heuristic
    /// is exact between neighbours) and lead the descent into a loop.
    pub fn path(&self) -> Option<Vec<(usize, usize)>> {
        if !self.cost_to_goal().is_finite() {
            return None;
        }
        let mut u = self.start;
        let mut out = vec![self.xy(u)];
        let mut seen = rustc_hash::FxHashSet::from_iter([u]);
        while u != self.goal {
            let (v, c) = self
                .edges(u)
                .filter(|&(v, _)| self.g[v] == self.rhs[v])
                .map(|(v, c)| (v, c + self.g[v]))
                .min_by(|a, b| a.1.total_cmp(&b.1))?;
            if !c.is_finite() || !seen.insert(v) {
                return None;
            }
            u = v;
            out.push(self.xy(u));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use world::rng::SplitMix64;

    /// Exact cost from start to goal with Dijkstra, for comparison.
    fn dijkstra(nx: usize, nz: usize, blocked: &[bool], start: (usize, usize), goal: (usize, usize)) -> f32 {
        let d = DStarLite::new(nx, nz, 1.0, blocked.to_vec(), start, goal);
        let mut dist = vec![f64::INFINITY; nx * nz];
        let s = start.1 * nx + start.0;
        dist[s] = 0.0;
        let mut heap = BinaryHeap::from([Entry([0.0, 0.0], s as u32)]);
        while let Some(Entry([c, _], u)) = heap.pop() {
            let u = u as usize;
            if c > dist[u] {
                continue;
            }
            for (v, w) in d.edges(u) {
                if c + w < dist[v] {
                    dist[v] = c + w;
                    heap.push(Entry([c + w, 0.0], v as u32));
                }
            }
        }
        dist[goal.1 * nx + goal.0] as f32
    }

    fn random_blocked(rng: &mut SplitMix64, n: usize, density: f32, keep: &[(usize, usize)]) -> Vec<bool> {
        let mut b: Vec<bool> = (0..n * n).map(|_| rng.next_f32() < density).collect();
        for &(i, j) in keep {
            b[j * n + i] = false;
        }
        b
    }

    #[test]
    fn empty_grid_gives_the_octile_distance() {
        let mut d = DStarLite::new(20, 20, 0.5, vec![false; 400], (1, 2), (15, 6));
        d.compute();
        let expected = (10.0 + 4.0 * std::f32::consts::SQRT_2) * 0.5;
        assert!((d.cost_to_goal() - expected).abs() < 1e-4);
        let p = d.path().unwrap();
        assert_eq!((p[0], *p.last().unwrap()), ((1, 2), (15, 6)));
    }

    #[test]
    fn matches_dijkstra_on_random_grids() {
        let mut rng = SplitMix64::new(7);
        let n = 30;
        for _ in 0..20 {
            let (s, g) = ((0, 0), (n - 1, n - 1));
            let b = random_blocked(&mut rng, n, 0.3, &[s, g]);
            let mut d = DStarLite::new(n, n, 1.0, b.clone(), s, g);
            d.compute();
            let want = dijkstra(n, n, &b, s, g);
            assert_eq!(d.cost_to_goal().is_finite(), want.is_finite());
            if want.is_finite() {
                assert!((d.cost_to_goal() - want).abs() < 1e-3, "{} != {want}", d.cost_to_goal());
                assert_eq!(d.path().unwrap().last(), Some(&g));
            }
        }
    }

    /// The drone moves along the path while cells change: the
    /// repaired cost is always that of a full search on the current map.
    #[test]
    fn incremental_repair_matches_a_fresh_search() {
        let mut rng = SplitMix64::new(11);
        let n = 40;
        let goal = (n - 2, n - 3);
        let mut b = random_blocked(&mut rng, n, 0.2, &[(1, 1), goal]);
        let mut d = DStarLite::new(n, n, 1.0, b.clone(), (1, 1), goal);
        d.compute();
        let mut fresh_expanded = 0;
        let mut repair_expanded = 0;
        for _ in 0..15 {
            let Some(path) = d.path() else { break };
            if path.len() < 3 {
                break;
            }
            let here = path[2];
            d.move_start(here);
            // A few cells change, except the position and the goal.
            let mut changes = Vec::new();
            for _ in 0..12 {
                let c = ((rng.next_u64() % n as u64) as usize, (rng.next_u64() % n as u64) as usize);
                if c != here && c != goal {
                    let v = rng.next_f32() < 0.7;
                    b[c.1 * n + c.0] = v;
                    changes.push((c, v));
                }
            }
            let before = d.expanded;
            d.set_blocked(changes);
            d.compute();
            repair_expanded += d.expanded - before;

            let mut f = DStarLite::new(n, n, 1.0, b.clone(), here, goal);
            f.compute();
            fresh_expanded += f.expanded;
            let (got, want) = (d.cost_to_goal(), f.cost_to_goal());
            assert_eq!(got.is_finite(), want.is_finite());
            if want.is_finite() {
                assert!((got - want).abs() < 1e-3, "{got} != {want}");
            }
        }
        assert!(repair_expanded < fresh_expanded, "{repair_expanded} >= {fresh_expanded}");
    }

    /// Large grid, 0.25 m cells, long path, moving start: the costs
    /// reach hundreds of meters, where an `f32` rounding stopped the search
    /// too early and made the path unextractable.
    #[test]
    fn long_paths_on_large_grids_are_always_extracted() {
        let mut rng = SplitMix64::new(3);
        let n = 600;
        let goal = (n - 10, 20);
        let b = random_blocked(&mut rng, n, 0.15, &[(10, n - 10), goal]);
        let mut d = DStarLite::new(n, n, 0.25, b, (10, n - 10), goal);
        for _ in 0..20 {
            d.compute();
            let path = d.path().expect("path extracted");
            assert_eq!(*path.last().unwrap(), goal);
            if path.len() < 30 {
                break;
            }
            d.move_start(path[25]);
        }
    }

    #[test]
    fn no_corner_cutting() {
        // Two diagonally blocked cells: the passage between them is closed.
        let mut b = vec![false; 9];
        b[1] = true; // (1, 0)
        b[3] = true; // (0, 1)
        let mut d = DStarLite::new(3, 3, 1.0, b, (0, 0), (1, 1));
        d.compute();
        assert!(d.path().is_none());
    }
}
