//! 3D map: sparse voxels in log-odds.
//!
//! Where the 2D grid only sees a horizontal slab, the voxel map keeps the whole
//! volume: facades, roofs, overhangs, the ground relief. Each ray frees the voxels
//! it crosses (3D DDA, Amanatides & Woo) and marks the voxel of its hit.
//! Voxels live in a hash map, so the map costs memory only where something was
//! seen. Pure Rust, no access to the ground truth.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use glam::{IVec2, IVec3, Vec3};
use sensors::{Data, Measurement};

use crate::LogOdds;

/// Voxel key: integer coordinates, voxel (0, 0, 0) spans [0, size)³.
pub type Key = IVec3;

/// Side of a chunk, in voxels: the unit of change tracking for the viewer's meshes.
pub const CHUNK: i32 = 16;

/// Chunk holding a voxel.
pub fn chunk_of(k: Key) -> IVec3 {
    k.div_euclid(IVec3::splat(CHUNK))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoxelCounts {
    pub occupied: usize,
    pub free: usize,
}

#[derive(Debug, Clone)]
pub struct VoxelMap {
    voxel_m: f32,
    pub params: LogOdds,
    cells: HashMap<Key, f32>,
    /// Occupied voxel heights (y keys) of each column (x, z), sorted.
    columns: HashMap<IVec2, Vec<i32>>,
    /// Columns with at least one observed voxel.
    seen: HashSet<IVec2>,
    /// Columns whose content changed since the last [`VoxelMap::take_dirty`].
    dirty: HashSet<IVec2>,
    /// Chunks whose occupied voxels, or the faces they show, changed since the
    /// last [`VoxelMap::take_dirty_chunks`].
    dirty_chunks: HashSet<IVec3>,
    /// Kept up to date by [`VoxelMap::add`]: reading them costs nothing.
    counts: VoxelCounts,
    /// Bumped at each change of an occupied voxel's state: the viewer redraws
    /// only when it moves.
    revision: u64,
    /// Rays longer than this only free their first `max_free_m` meters (far
    /// misses at grazing angles would erase thin structures).
    pub max_free_m: f32,
}

impl VoxelMap {
    pub fn new(voxel_m: f32) -> Self {
        assert!(voxel_m > 0.0);
        Self {
            voxel_m,
            params: LogOdds::default(),
            cells: HashMap::default(),
            columns: HashMap::default(),
            dirty: HashSet::default(),
            dirty_chunks: HashSet::default(),
            counts: VoxelCounts::default(),
            seen: HashSet::default(),
            revision: 0,
            max_free_m: 15.0,
        }
    }

    pub fn voxel_m(&self) -> f32 {
        self.voxel_m
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn clear(&mut self) {
        self.dirty.extend(self.columns.keys().copied());
        self.dirty.extend(self.cells.keys().map(|k| IVec2::new(k.x, k.z)));
        let t = self.params.occupied_above;
        self.dirty_chunks.extend(self.cells.iter().filter(|(_, l)| **l > t).map(|(k, _)| chunk_of(*k)));
        self.cells.clear();
        self.columns.clear();
        self.seen.clear();
        self.counts = VoxelCounts::default();
        self.revision += 1;
    }

    pub fn key(&self, p: Vec3) -> Key {
        (p / self.voxel_m).floor().as_ivec3()
    }

    pub fn center(&self, k: Key) -> Vec3 {
        (k.as_vec3() + 0.5) * self.voxel_m
    }

    pub fn logodds(&self, k: Key) -> f32 {
        self.cells.get(&k).copied().unwrap_or(0.0)
    }

    pub fn is_occupied(&self, k: Key) -> bool {
        self.logodds(k) > self.params.occupied_above
    }

    pub fn is_free(&self, k: Key) -> bool {
        self.logodds(k) < self.params.free_below
    }

    /// Occupied voxels, in no particular order.
    pub fn occupied(&self) -> impl Iterator<Item = Key> + '_ {
        let t = self.params.occupied_above;
        self.cells.iter().filter(move |(_, l)| **l > t).map(|(k, _)| *k)
    }

    pub fn counts(&self) -> VoxelCounts {
        self.counts
    }

    fn add(&mut self, k: Key, delta: f32) {
        let p = self.params;
        let col = IVec2::new(k.x, k.z);
        if self.seen.insert(col) {
            // A newly observed column.
            self.dirty.insert(col);
        }
        let l = self.cells.entry(k).or_insert(0.0);
        let was = *l > p.occupied_above;
        let was_free = *l < p.free_below;
        *l = (*l + delta).clamp(p.min, p.max);
        let now = *l > p.occupied_above;
        let now_free = *l < p.free_below;
        self.counts.occupied = self.counts.occupied + now as usize - was as usize;
        self.counts.free = self.counts.free + now_free as usize - was_free as usize;
        if was != now {
            self.revision += 1;
            self.dirty.insert(col);
            self.mark_chunks(k);
            let ys = self.columns.entry(col).or_default();
            match ys.binary_search(&k.y) {
                Err(at) if now => ys.insert(at, k.y),
                Ok(at) if !now => {
                    ys.remove(at);
                }
                _ => {}
            }
        }
    }

    /// Marks the chunk of `k` dirty, and the neighbor chunks whose faces it hides or
    /// uncovers (when `k` lies on a chunk border).
    fn mark_chunks(&mut self, k: Key) {
        let c = chunk_of(k);
        self.dirty_chunks.insert(c);
        let local = k - c * CHUNK;
        for axis in 0..3 {
            let mut step = IVec3::ZERO;
            if local[axis] == 0 {
                step[axis] = -1;
            } else if local[axis] == CHUNK - 1 {
                step[axis] = 1;
            } else {
                continue;
            }
            self.dirty_chunks.insert(c + step);
        }
    }

    /// Occupied voxels of a chunk, column by column.
    pub fn chunk_occupied(&self, chunk: IVec3) -> impl Iterator<Item = Key> + '_ {
        let o = chunk * CHUNK;
        (0..CHUNK).flat_map(move |dx| (0..CHUNK).map(move |dz| IVec2::new(o.x + dx, o.z + dz))).flat_map(move |c| {
            let ys = self.column(c);
            let lo = ys.partition_point(|&y| y < o.y);
            let hi = ys.partition_point(|&y| y < o.y + CHUNK);
            ys[lo..hi].iter().map(move |&y| IVec3::new(c.x, y, c.y))
        })
    }

    /// Chunks changed since the last call.
    pub fn take_dirty_chunks(&mut self) -> Vec<IVec3> {
        self.dirty_chunks.drain().collect()
    }

    /// Column key of the point (x, z).
    pub fn column_key(&self, x: f32, z: f32) -> IVec2 {
        IVec2::new((x / self.voxel_m).floor() as i32, (z / self.voxel_m).floor() as i32)
    }

    /// Occupied voxel heights of a column (y keys, sorted).
    pub fn column(&self, c: IVec2) -> &[i32] {
        self.columns.get(&c).map_or(&[], Vec::as_slice)
    }

    /// Has any voxel of the column been observed (free or occupied)?
    pub fn column_known(&self, c: IVec2) -> bool {
        self.seen.contains(&c)
    }

    /// Top of the highest occupied voxel of a column, in m.
    pub fn column_top(&self, c: IVec2) -> Option<f32> {
        self.column(c).last().map(|&y| (y + 1) as f32 * self.voxel_m)
    }

    /// Bottom of the lowest occupied voxel of a column, in m.
    pub fn column_floor(&self, c: IVec2) -> Option<f32> {
        self.column(c).first().map(|&y| y as f32 * self.voxel_m)
    }

    /// Is there an occupied voxel of the column overlapping heights [lo, hi]?
    pub fn column_blocks(&self, c: IVec2, lo: f32, hi: f32) -> bool {
        let (a, b) = ((lo / self.voxel_m).floor() as i32, (hi / self.voxel_m).floor() as i32);
        let ys = self.column(c);
        let at = ys.partition_point(|&y| y < a);
        ys.get(at).is_some_and(|&y| y <= b)
    }

    /// Every column with at least one observed voxel.
    pub fn known_columns(&self) -> impl Iterator<Item = IVec2> + '_ {
        self.seen.iter().copied()
    }

    /// Columns changed since the last call.
    pub fn take_dirty(&mut self) -> Vec<IVec2> {
        self.dirty.drain().collect()
    }

    /// Integrates a range measurement (lidar, depth image). Other measurements
    /// (sonar, navigation) are ignored: a sonar does not say where in its cone
    /// the echo comes from.
    ///
    /// The voxels each ray crosses are computed in parallel, then applied in the
    /// order of the rays: the same map as one ray after the other.
    pub fn integrate(&mut self, m: &Measurement) {
        let (beams, max_range) = match &m.data {
            Data::Scan { beams, max_range } | Data::DepthImage { beams, max_range, .. } => (beams, *max_range),
            _ => return,
        };
        let reach = max_range.min(self.max_free_m);
        let ray = |(i, b): (usize, &sensors::Beam)| {
            let (origin, rotation) = m.beam_pose(i);
            let dir = rotation * b.dir;
            let (end, hit) = match b.range {
                Some(d) => (origin + dir * d, true),
                None => (origin + dir * reach, false),
            };
            let mut keys = Vec::new();
            let end_key = self.ray_keys(origin, end, hit, &mut keys);
            (keys, end_key)
        };
        #[cfg(feature = "parallel")]
        let rays: Vec<_> = {
            use rayon::prelude::*;
            beams.par_iter().enumerate().with_min_len(64).filter(|(_, b)| b.valid).map(ray).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let rays: Vec<_> = beams.iter().enumerate().filter(|(_, b)| b.valid).map(ray).collect();
        let (miss, hit) = (self.params.miss, self.params.hit);
        for (keys, end_key) in rays {
            for k in keys {
                self.add(k, miss);
            }
            if let Some(k) = end_key {
                self.add(k, hit);
            }
        }
    }

    /// One ray from `origin` to `end`: frees the voxels crossed before `end`,
    /// and marks `end`'s voxel occupied if `hit`.
    pub fn insert_ray(&mut self, origin: Vec3, end: Vec3, hit: bool) {
        let mut keys = Vec::new();
        let end_key = self.ray_keys(origin, end, hit, &mut keys);
        for k in keys {
            self.add(k, self.params.miss);
        }
        if let Some(k) = end_key {
            self.add(k, self.params.hit);
        }
    }

    /// Voxels a ray frees (pushed to `free`), and the one it marks occupied.
    fn ray_keys(&self, origin: Vec3, end: Vec3, hit: bool, free: &mut Vec<Key>) -> Option<Key> {
        let end_key = self.key(end);
        let length = origin.distance(end);
        // Free space stops a little before the hit: noise must not erase the
        // surface it lands on.
        let free_len = if hit { (length - 0.5 * self.voxel_m).max(0.0) } else { length.min(self.max_free_m) };
        if free_len > 0.0 {
            let dir = (end - origin) / length.max(1e-6);
            self.traverse(origin, dir, free_len, |k| {
                if k != end_key || !hit {
                    free.push(k);
                }
            });
        }
        hit.then_some(end_key)
    }

    /// Visits the voxels crossed by the segment `origin + t·dir`, `0 <= t <= len`.
    fn traverse(&self, origin: Vec3, dir: Vec3, len: f32, mut visit: impl FnMut(Key)) {
        let s = self.voxel_m;
        let mut k = self.key(origin);
        let step = IVec3::new(sign(dir.x), sign(dir.y), sign(dir.z));
        // Distance along the ray to the next boundary on each axis, and between boundaries.
        let mut t_max = Vec3::ZERO;
        let mut t_delta = Vec3::ZERO;
        for i in 0..3 {
            if dir[i].abs() < 1e-9 {
                t_max[i] = f32::INFINITY;
                t_delta[i] = f32::INFINITY;
            } else {
                let next = if dir[i] > 0.0 { (k[i] + 1) as f32 * s } else { k[i] as f32 * s };
                t_max[i] = (next - origin[i]) / dir[i];
                t_delta[i] = s / dir[i].abs();
            }
        }
        visit(k);
        loop {
            let axis = if t_max.x <= t_max.y && t_max.x <= t_max.z {
                0
            } else if t_max.y <= t_max.z {
                1
            } else {
                2
            };
            if t_max[axis] > len {
                return;
            }
            k[axis] += step[axis];
            t_max[axis] += t_delta[axis];
            visit(k);
        }
    }
}

fn sign(v: f32) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use drone_spec::{Mount, SensorKind, SensorSpec};
    use glam::Quat;
    use sensors::{Pose, SensorSuite};
    use world::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainSpec, World, WorldSpec};

    #[test]
    fn traverse_visits_a_connected_line() {
        let m = VoxelMap::new(0.5);
        let mut keys = Vec::new();
        m.traverse(Vec3::new(0.1, 0.1, 0.1), Vec3::new(1.0, 0.5, 0.25).normalize(), 5.0, |k| keys.push(k));
        assert_eq!(keys[0], IVec3::ZERO);
        for w in keys.windows(2) {
            let d = (w[1] - w[0]).abs();
            assert_eq!(d.x + d.y + d.z, 1, "{w:?}");
        }
        let last = *keys.last().unwrap();
        let end = m.key(Vec3::new(0.1, 0.1, 0.1) + Vec3::new(1.0, 0.5, 0.25).normalize() * 5.0);
        assert_eq!(last, end);
    }

    #[test]
    fn columns_track_tops_and_changes() {
        let mut m = VoxelMap::new(0.5);
        for _ in 0..3 {
            m.insert_ray(Vec3::new(0.2, 10.2, 0.2), Vec3::new(0.2, 3.2, 0.2), true);
            m.insert_ray(Vec3::new(5.2, 1.2, 0.2), Vec3::new(5.2, 0.1, 0.2), true);
        }
        let c = m.column_key(0.2, 0.2);
        assert!(m.column_known(c));
        assert_eq!(m.column_top(c), Some(3.5));
        assert!(m.column_blocks(c, 2.0, 3.2));
        assert!(!m.column_blocks(c, 3.6, 6.0));
        let dirty = m.take_dirty();
        assert!(dirty.contains(&c) && dirty.contains(&m.column_key(5.2, 0.2)));
        assert!(m.take_dirty().is_empty());
        assert!(!m.column_known(m.column_key(30.0, 30.0)));
    }

    #[test]
    fn a_ray_frees_its_path_and_marks_its_end() {
        let mut m = VoxelMap::new(0.25);
        for _ in 0..3 {
            m.insert_ray(Vec3::new(0.1, 1.1, 0.1), Vec3::new(5.1, 1.1, 0.1), true);
        }
        assert!(m.is_occupied(m.key(Vec3::new(5.1, 1.1, 0.1))));
        assert!(m.is_free(m.key(Vec3::new(2.5, 1.1, 0.1))));
        assert!(!m.is_occupied(m.key(Vec3::new(2.5, 1.1, 0.1))));
    }

    /// A 3D lidar circling a building maps its walls and roof, and the voxels it
    /// marks occupied really hold matter.
    #[test]
    fn lidar_maps_a_building_in_3d() {
        let world = World::build(&WorldSpec {
            size_m: 60.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::List(vec![
                ObstacleSpec {
                    shape: ObstacleShape::Box { size: Vec3::new(8.0, 4.0, 6.0) },
                    x: 0.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                    elevation_m: 0.0,
                    material: Default::default(),
                },
                ObstacleSpec {
                    shape: ObstacleShape::Roof { size: Vec3::new(9.0, 2.5, 7.0) },
                    x: 0.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                    elevation_m: 4.0,
                    material: Default::default(),
                },
            ]),
            wind: Default::default(),
        })
        .unwrap();
        let lidar = SensorSpec {
            kind: SensorKind::Lidar3D {
                channels: 16,
                rays: 180,
                fov_up_deg: 15.0,
                fov_down_deg: -45.0,
                range: 20.0,
                noise_std: 0.0,
                dropout: 0.0,
            },
            mount: Mount::default(),
            rate_hz: 10.0,
            latency_s: 0.0,
        };
        let mut suite = SensorSuite::new(&[lidar], 1);
        let mut map = VoxelMap::new(0.4);
        for k in 0..24 {
            let a = std::f32::consts::TAU * k as f32 / 24.0;
            let p = Vec3::new(12.0 * a.cos(), 9.0, 12.0 * a.sin());
            suite.update(&world, &Pose::new(p, Quat::IDENTITY), k as f64);
            map.integrate(suite.latest[0].as_ref().unwrap());
        }
        let occupied: Vec<Key> = map.occupied().collect();
        // Every occupied voxel touches matter (ground, walls, roof).
        let near_matter = |c: Vec3| {
            let h = map.voxel_m();
            [-0.5, 0.5].iter().any(|dx| {
                [-0.5, 0.5].iter().any(|dz| {
                    let col = world.column(c.x + dx * h, c.z + dz * h, c.y - h, c.y + h);
                    col.matter
                })
            })
        };
        let wrong = occupied.iter().filter(|&&k| !near_matter(map.center(k))).count();
        assert!(wrong * 100 < occupied.len(), "{wrong}/{} voxels in the air", occupied.len());
        // The roof is seen: voxels above the walls (4 m), under the ridge (6.5 m).
        let roof = occupied.iter().filter(|&&k| {
            let c = map.center(k);
            c.y > 4.5 && c.x.abs() < 4.0 && c.z.abs() < 3.0
        });
        assert!(roof.count() > 50);
        // The walls are seen, at mid-height.
        let walls = occupied.iter().filter(|&&k| {
            let c = map.center(k);
            (1.0..3.5).contains(&c.y) && (c.x.abs() - 4.0).abs() < 0.5 && c.z.abs() < 3.0
        });
        assert!(walls.count() > 20);
    }

    #[test]
    fn chunks_track_occupied_changes() {
        let mut map = VoxelMap::new(1.0);
        map.take_dirty_chunks();
        // Inside a chunk: only that chunk is dirty.
        map.insert_ray(Vec3::new(5.5, 5.5, 0.5), Vec3::new(5.5, 5.5, 5.5), true);
        map.insert_ray(Vec3::new(5.5, 5.5, 0.5), Vec3::new(5.5, 5.5, 5.5), true);
        assert!(map.is_occupied(IVec3::new(5, 5, 5)));
        assert_eq!(map.take_dirty_chunks(), vec![IVec3::ZERO]);
        assert_eq!(map.chunk_occupied(IVec3::ZERO).collect::<Vec<_>>(), vec![IVec3::new(5, 5, 5)]);
        // Repeated hits that do not change the state dirty nothing.
        map.insert_ray(Vec3::new(5.5, 5.5, 0.5), Vec3::new(5.5, 5.5, 5.5), true);
        assert!(map.take_dirty_chunks().is_empty());
        // On a chunk border: the neighbor's faces change too.
        for _ in 0..2 {
            map.insert_ray(Vec3::new(5.5, 5.5, 20.5), Vec3::new(5.5, 5.5, 16.5), true);
        }
        let mut d = map.take_dirty_chunks();
        d.sort_by_key(|c| c.z);
        assert_eq!(d, vec![IVec3::ZERO, IVec3::Z]);
        assert!(map.chunk_occupied(IVec3::NEG_ONE).next().is_none());
        // Clearing dirties every chunk that had something.
        map.clear();
        let mut d = map.take_dirty_chunks();
        d.sort_by_key(|c| c.z);
        assert_eq!(d, vec![IVec3::ZERO, IVec3::Z]);
    }
}
