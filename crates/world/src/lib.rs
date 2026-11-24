//! Simulated world: ground truth (terrain + obstacles).
//!
//! Pure Rust, no Bevy. A world is described by a [`WorldSpec`]:
//! - generated from a seed ([`WorldSpec::procedural`]);
//! - or hand-drawn in a RON file ([`WorldSpec::from_ron`]).
//!
//! Frame: Y up, origin at the center of the map, units in meters.

mod city;
mod collision;
mod index;
pub mod drone;
mod raycast;
pub mod rng;
mod spec;
mod terrain;
mod wind;

pub use city::CitySpec;
pub use collision::Contact;
pub use raycast::Hit;
pub use spec::{Material, ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainFeature, TerrainSpec, WorldSpec};
pub use wind::{Wind, WindSpec};
pub use terrain::Heightmap;


use glam::Vec3;
use rng::SplitMix64;

/// Obstacle placed in the world.
#[derive(Debug, Clone, PartialEq)]
pub struct Obstacle {
    pub shape: ObstacleShape,
    /// Center of the base, resting on the terrain.
    pub base: Vec3,
    /// Rotation around Y, in radians.
    pub yaw: f32,
    pub material: Material,
}

impl Obstacle {
    /// Height of the obstacle above its base.
    pub fn height(&self) -> f32 {
        match self.shape {
            ObstacleShape::Box { size } | ObstacleShape::Roof { size } => size.y,
            ObstacleShape::Cylinder { height, .. } => height,
        }
    }

    /// Point (x, z) in the obstacle frame (inverse yaw rotation), relative to its base.
    fn local_xz(&self, x: f32, z: f32) -> (f32, f32) {
        let (dx, dz) = (x - self.base.x, z - self.base.z);
        let (s, c) = self.yaw.sin_cos();
        (c * dx - s * dz, s * dx + c * dz)
    }

    /// Vertical extent (bottom, top) of the matter above (x, z), if any.
    pub fn span_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        if !self.covers(x, z) {
            return None;
        }
        let top = match self.shape {
            ObstacleShape::Roof { size } => {
                let (_, lz) = self.local_xz(x, z);
                size.y * (1.0 - lz.abs() / (size.z / 2.0)).max(0.0)
            }
            _ => self.height(),
        };
        Some((self.base.y, self.base.y + top))
    }

    /// Is the point (x, z) inside the ground footprint?
    pub fn covers(&self, x: f32, z: f32) -> bool {
        let (dx, dz) = (x - self.base.x, z - self.base.z);
        match self.shape {
            ObstacleShape::Box { size } | ObstacleShape::Roof { size } => {
                let (lx, lz) = self.local_xz(x, z);
                lx.abs() <= size.x / 2.0 && lz.abs() <= size.z / 2.0
            }
            ObstacleShape::Cylinder { radius, .. } => dx * dx + dz * dz <= radius * radius,
        }
    }

    /// Radius of the circle containing the ground footprint.
    pub fn footprint_radius(&self) -> f32 {
        match self.shape {
            ObstacleShape::Box { size } | ObstacleShape::Roof { size } => 0.5 * (size.x * size.x + size.z * size.z).sqrt(),
            ObstacleShape::Cylinder { radius, .. } => radius,
        }
    }
}

/// Built world: the ground truth that only the sensors can read.
#[derive(Debug, Clone)]
pub struct World {
    pub terrain: Heightmap,
    /// Obstacles. To add one, go through [`World::add_obstacle`], which keeps
    /// the spatial index up to date (otherwise queries fall back to a brute-force walk).
    pub obstacles: Vec<Obstacle>,
    pub(crate) index: index::ObstacleIndex,
    /// Wind field; can be changed in flight.
    pub wind: Wind,
}

impl World {
    pub fn build(spec: &WorldSpec) -> Result<Self, BuildError> {
        spec.validate()?;
        let terrain = Heightmap::build(spec.size_m, spec.cell_m, &spec.terrain)?;
        let obstacles = match &spec.obstacles {
            ObstaclesSpec::None => Vec::new(),
            ObstaclesSpec::List(list) => list
                .iter()
                .map(|o| {
                    let mut p = place(&terrain, o.shape, o.x, o.z, o.yaw_deg.to_radians());
                    p.material = o.material;
                    p.base.y += o.elevation_m;
                    p
                })
                .collect(),
            ObstaclesSpec::Random {
                seed,
                count,
                clear_radius_m,
            } => scatter(&terrain, *seed, *count, *clear_radius_m),
            ObstaclesSpec::City(city) => city::generate(&terrain, city),
        };
        let index = index::ObstacleIndex::build(&obstacles, terrain.half_size());
        Ok(Self { terrain, obstacles, index, wind: Wind::new(spec.wind) })
    }

    /// Adds an obstacle resting on the terrain at (x, z), for example an unexpected
    /// obstacle appearing during a mission. Returns the placed obstacle.
    pub fn add_obstacle(&mut self, shape: ObstacleShape, x: f32, z: f32, yaw: f32) -> Obstacle {
        let o = place(&self.terrain, shape, x, z, yaw);
        if self.index.count == self.obstacles.len() {
            self.index.insert(&o);
        }
        self.obstacles.push(o.clone());
        o
    }

    /// Does the index cover every obstacle?
    pub(crate) fn indexed(&self) -> bool {
        self.index.count == self.obstacles.len()
    }

    /// Obstacles that may cover the point (x, z).
    fn obstacles_at(&self, x: f32, z: f32) -> Box<dyn Iterator<Item = &Obstacle> + '_> {
        if self.indexed() {
            Box::new(self.index.at(x, z).iter().map(|&k| &self.obstacles[k as usize]))
        } else {
            Box::new(self.obstacles.iter())
        }
    }

    /// Ground truth for a 2.5D map: what does the vertical column at (x, z) contain
    /// between heights `y_lo` and `y_hi`? A column can contain both
    /// matter and empty space (a hillside cutting through the slab).
    pub fn column(&self, x: f32, z: f32, y_lo: f32, y_hi: f32) -> Column {
        let half = self.terrain.half_size();
        if x.abs() > half || z.abs() > half {
            return Column { matter: false, space: true };
        }
        // Intervals of matter within the slab.
        let mut spans: Vec<(f32, f32)> = Vec::new();
        let ground = self.terrain.height_at(x, z);
        if ground >= y_lo {
            spans.push((y_lo, ground.min(y_hi)));
        }
        for o in self.obstacles_at(x, z) {
            if let Some((b, t)) = o.span_at(x, z) {
                let (b, t) = (b.max(y_lo), t.min(y_hi));
                if b <= t {
                    spans.push((b, t));
                }
            }
        }
        if spans.is_empty() {
            return Column { matter: false, space: true };
        }
        // Does matter fill the whole slab?
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut reach = y_lo;
        for (b, t) in spans {
            if b > reach + 1e-4 {
                break;
            }
            reach = reach.max(t);
        }
        Column {
            matter: true,
            space: reach < y_hi - 1e-4,
        }
    }
}

/// Content of a world column within a height slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Column {
    /// There is matter somewhere in the slab.
    pub matter: bool,
    /// There is empty space somewhere in the slab.
    pub space: bool,
}

/// Places an obstacle: its base follows the lowest terrain point under its footprint,
/// so that it never floats.
fn place(terrain: &Heightmap, shape: ObstacleShape, x: f32, z: f32, yaw: f32) -> Obstacle {
    let mut o = Obstacle {
        shape,
        base: Vec3::new(x, 0.0, z),
        yaw,
        material: Material::Concrete,
    };
    let r = o.footprint_radius();
    let ground = [(0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]
        .iter()
        .map(|(dx, dz)| terrain.height_at(x + dx, z + dz))
        .fold(f32::INFINITY, f32::min);
    o.base.y = ground;
    o
}

/// Scatters `count` random obstacles, outside the start area around the origin.
fn scatter(terrain: &Heightmap, seed: u64, count: usize, clear_radius_m: f32) -> Vec<Obstacle> {
    let mut rng = SplitMix64::new(seed);
    let half = terrain.half_size() * 0.95;
    let mut out = Vec::with_capacity(count);
    // Cap on attempts: a too-large cleared area does not block generation.
    let mut attempts = 0;
    while out.len() < count && attempts < count * 20 {
        attempts += 1;
        let x = rng.range(-half, half);
        let z = rng.range(-half, half);
        let shape = if rng.next_f32() < 0.5 {
            ObstacleShape::Box {
                size: Vec3::new(rng.range(1.0, 6.0), rng.range(2.0, 12.0), rng.range(1.0, 6.0)),
            }
        } else {
            ObstacleShape::Cylinder {
                radius: rng.range(0.3, 2.0),
                height: rng.range(3.0, 15.0),
            }
        };
        let yaw = rng.range(0.0, std::f32::consts::TAU);
        let o = place(terrain, shape, x, z, yaw);
        if (x * x + z * z).sqrt() - o.footprint_radius() < clear_radius_m {
            continue;
        }
        out.push(o);
    }
    out
}

#[derive(Debug)]
pub enum BuildError {
    Invalid(String),
    Ron(ron::error::SpannedError),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Invalid(msg) => write!(f, "invalid map: {msg}"),
            BuildError::Ron(e) => write!(f, "invalid RON: {e}"),
        }
    }
}

impl std::error::Error for BuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_world() {
        let a = World::build(&WorldSpec::procedural(42)).unwrap();
        let b = World::build(&WorldSpec::procedural(42)).unwrap();
        assert_eq!(a.terrain.heights(), b.terrain.heights());
        assert_eq!(a.obstacles, b.obstacles);
    }

    #[test]
    fn different_seed_different_world() {
        let a = World::build(&WorldSpec::procedural(1)).unwrap();
        let b = World::build(&WorldSpec::procedural(2)).unwrap();
        assert_ne!(a.terrain.heights(), b.terrain.heights());
        assert_ne!(a.obstacles, b.obstacles);
    }

    #[test]
    fn spawn_zone_is_clear() {
        let spec = WorldSpec::procedural(7);
        let ObstaclesSpec::Random { clear_radius_m, .. } = spec.obstacles else {
            panic!("procedural must draw obstacles");
        };
        let w = World::build(&spec).unwrap();
        assert!(!w.obstacles.is_empty());
        for o in &w.obstacles {
            let d = (o.base.x * o.base.x + o.base.z * o.base.z).sqrt();
            assert!(d - o.footprint_radius() >= clear_radius_m);
        }
    }

    #[test]
    fn obstacles_never_float() {
        let w = World::build(&WorldSpec::procedural(3)).unwrap();
        for o in &w.obstacles {
            let ground = w.terrain.height_at(o.base.x, o.base.z);
            assert!(o.base.y <= ground + 1e-4, "base {} above ground {}", o.base.y, ground);
        }
    }

    #[test]
    fn hand_drawn_map_loads() {
        let src = include_str!("../../../maps/corridor.ron");
        let spec = WorldSpec::from_ron(src).unwrap();
        let w = World::build(&spec).unwrap();
        assert!(matches!(spec.terrain, TerrainSpec::Flat { .. }));
        assert!(w.obstacles.len() >= 3);
        // Flat terrain: all bases at the same height.
        assert!(w.obstacles.iter().all(|o| o.base.y == w.obstacles[0].base.y));
    }

    #[test]
    fn house_map_has_relief_roofs_and_wind() {
        let spec = WorldSpec::from_ron(include_str!("../../../maps/house.ron")).unwrap();
        let w = World::build(&spec).unwrap();
        assert!(w.obstacles.iter().any(|o| matches!(o.shape, ObstacleShape::Roof { .. })));
        let (lo, hi) = w.terrain.range();
        assert!(hi - lo > 10.0, "{lo}..{hi}");
        // Take-off pad and house plot are flat.
        assert!(w.terrain.height_at(2.0, -2.0).abs() < 1e-4);
        assert!((w.terrain.height_at(-22.0, -18.0) - 0.5).abs() < 1e-4);
        // Clear around the take-off point.
        assert!(!w.collides(Vec3::new(0.0, 4.0, 0.0), 3.5));
        // The door, the porch and the building's passage are open at 1.5 m.
        assert!(!w.collides(Vec3::new(-22.0, 2.0, -13.0), 0.4));
        assert!(!w.collides(Vec3::new(-22.0, 2.0, -11.5), 0.4));
        assert!(!w.collides(Vec3::new(26.0, 2.5, 8.0), 0.8));
        // Above the door, the lintel; inside, under the upper floor, room to fly.
        assert!(w.collides(Vec3::new(-22.0, 3.5, -13.0), 0.2));
        assert!(!w.collides(Vec3::new(-20.0, 2.0, -18.0), 0.4));
        // Wind blows over open ground.
        let open = w.wind_at(Vec3::new(0.0, 10.0, -50.0), 0.0);
        assert!(open.length() > 2.0, "{open}");
    }

    #[test]
    fn shipped_maps_build() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../maps");
        let mut n = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "ron") {
                let spec = WorldSpec::from_ron(&std::fs::read_to_string(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                World::build(&spec).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                n += 1;
            }
        }
        assert!(n >= 6);
    }

    #[test]
    fn spec_round_trips_through_ron() {
        let spec = WorldSpec::procedural(9);
        let text = spec.to_ron();
        assert_eq!(WorldSpec::from_ron(&text).unwrap(), spec);
    }

    #[test]
    fn column_matches_obstacles_and_ground() {
        let w = World::build(&WorldSpec::from_ron(include_str!("../../../maps/corridor.ron")).unwrap()).unwrap();
        let full = Column { matter: true, space: false };
        let empty = Column { matter: false, space: true };
        let partial = Column { matter: true, space: true };
        // Left wall of the corridor (x = -4, thickness 0.5, z from -35 to -5, height 4).
        assert_eq!(w.column(-4.0, -20.0, 1.0, 3.0), full);
        assert_eq!(w.column(-4.0, -20.0, 3.0, 5.0), partial); // top of the wall in the slab
        assert_eq!(w.column(-4.0, -20.0, 4.5, 6.0), empty); // above the wall
        assert_eq!(w.column(0.0, -20.0, 1.0, 3.0), empty); // middle of the corridor
        assert_eq!(w.column(0.0, -20.0, -1.0, 0.5), partial); // slab cutting the ground
        assert_eq!(w.column(0.0, -20.0, -2.0, -1.0), full); // below the ground
        // Block rotated by 30° at (-15, 12), 5 × 3 m: its corner along its long axis.
        let (s, c) = 30f32.to_radians().sin_cos();
        let along = |d: f32| (-15.0 + c * d, 12.0 - s * d);
        let (x, z) = along(2.3);
        assert_eq!(w.column(x, z, 1.0, 3.0), full);
        let (x, z) = along(2.7);
        assert_eq!(w.column(x, z, 1.0, 3.0), empty);
    }

    #[test]
    fn invalid_spec_is_rejected() {
        let mut spec = WorldSpec::procedural(1);
        spec.cell_m = 0.0;
        assert!(World::build(&spec).is_err());
    }
}
