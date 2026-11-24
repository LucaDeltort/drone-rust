//! Procedural cities: a street grid, city blocks split into lots,
//! one building per built lot, parks planted with trees and a few towers.
//!
//! The origin is on an intersection, cleared within a given radius: the drone takes off there.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{Heightmap, Obstacle, ObstacleShape, place, rng::SplitMix64};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitySpec {
    pub seed: u64,
    /// Side of a city block, sidewalks included, in m.
    pub block_m: f32,
    /// Street width, in m.
    pub street_m: f32,
    /// Lots per block side (1 to 6).
    pub lots: u32,
    /// Width of the alleys between lots, in m.
    pub alley_m: f32,
    /// Building height (min, max), in m.
    pub height_m: (f32, f32),
    /// Probability that a lot is built.
    pub density: f32,
    /// Probability that a block is a park (trees, no buildings).
    #[serde(default)]
    pub park_prob: f32,
    /// Probability that a building is a tower (max height × 2.5, smaller footprint).
    #[serde(default)]
    pub tower_prob: f32,
    /// Rotation of the whole grid, in degrees.
    #[serde(default)]
    pub rotation_deg: f32,
    /// Old-town disorder: rotation (degrees) and offset (m) of each
    /// building within its lot.
    #[serde(default)]
    pub jitter_deg: f32,
    #[serde(default)]
    pub jitter_m: f32,
    /// Cleared square around the origin, in m.
    pub clear_radius_m: f32,
}

impl CitySpec {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let positive = [("block_m", self.block_m), ("street_m", self.street_m)];
        if let Some((name, _)) = positive.iter().find(|(_, v)| !v.is_finite() || *v <= 0.0) {
            return Err(format!("{name} must be > 0"));
        }
        if !(1..=6).contains(&self.lots) {
            return Err("lots out of 1..=6".into());
        }
        let inner = self.block_m - (self.lots - 1) as f32 * self.alley_m;
        if self.alley_m < 0.0 || inner / self.lots as f32 <= 1.0 {
            return Err("lots too small for this block and these alleys".into());
        }
        let (lo, hi) = self.height_m;
        if !(lo > 0.0 && hi >= lo) {
            return Err("height_m: 0 < min <= max expected".into());
        }
        for (name, p) in [("density", self.density), ("park_prob", self.park_prob), ("tower_prob", self.tower_prob)] {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("{name} out of 0..=1"));
            }
        }
        if self.jitter_m < 0.0 || self.clear_radius_m < 0.0 {
            return Err("jitter_m and clear_radius_m must be >= 0".into());
        }
        Ok(())
    }
}

pub(crate) fn generate(terrain: &Heightmap, spec: &CitySpec) -> Vec<Obstacle> {
    let mut rng = SplitMix64::new(spec.seed);
    let half = terrain.half_size();
    let pitch = spec.block_m + spec.street_m;
    let rot = spec.rotation_deg.to_radians();
    let (s, c) = rot.sin_cos();
    // Grid frame to world frame (same yaw convention as obstacles).
    let to_world = |p: Vec2| Vec2::new(p.x * c + p.y * s, -p.x * s + p.y * c);
    let lot = (spec.block_m - (spec.lots - 1) as f32 * spec.alley_m) / spec.lots as f32;
    // Enough blocks to cover the map whatever the rotation.
    let k_max = (half * std::f32::consts::SQRT_2 / pitch).ceil() as i32 + 1;
    let mut out = Vec::new();

    for bj in -k_max..k_max {
        for bi in -k_max..k_max {
            // Block between two streets; the origin is in the middle of an intersection.
            let corner = Vec2::new(bi as f32, bj as f32) * pitch + Vec2::splat(spec.street_m / 2.0);
            let center = to_world(corner + Vec2::splat(spec.block_m / 2.0));
            if center.x.abs() > half - spec.block_m / 2.0 || center.y.abs() > half - spec.block_m / 2.0 {
                continue;
            }
            if rng.next_f32() < spec.park_prob {
                park(terrain, spec, &mut rng, corner, &to_world, rot, &mut out);
                continue;
            }
            for lj in 0..spec.lots {
                for li in 0..spec.lots {
                    let lot_corner = corner + Vec2::new(li as f32, lj as f32) * (lot + spec.alley_m);
                    let lot_center = to_world(lot_corner + Vec2::splat(lot / 2.0));
                    if lot_center.length() < spec.clear_radius_m + lot || rng.next_f32() >= spec.density {
                        continue;
                    }
                    out.push(building(terrain, spec, &mut rng, lot, lot_center, rot));
                }
            }
        }
    }
    out
}

fn building(terrain: &Heightmap, spec: &CitySpec, rng: &mut SplitMix64, lot: f32, center: Vec2, rot: f32) -> Obstacle {
    let (lo, hi) = spec.height_m;
    let tower = rng.next_f32() < spec.tower_prob;
    // Footprint: the lot minus a random setback; a tower is thinner.
    let shrink = if tower { 0.55 } else { 1.0 };
    let wx = lot * rng.range(0.7, 0.95) * shrink;
    let wz = lot * rng.range(0.7, 0.95) * shrink;
    let height = if tower { hi * rng.range(1.8, 2.5) } else { rng.range(lo, hi) };
    // The offset never pushes the building out of its lot.
    let room = ((lot - wx.max(wz)) / 2.0).max(0.0);
    let jitter = spec.jitter_m.min(room);
    let offset = Vec2::new(rng.range(-jitter, jitter.max(1e-6)), rng.range(-jitter, jitter.max(1e-6)));
    let yaw = rot + rng.range(-spec.jitter_deg, spec.jitter_deg.max(1e-6)).to_radians();
    let p = center + offset;
    place(terrain, ObstacleShape::Box { size: Vec3::new(wx, height, wz) }, p.x, p.y, yaw)
}

fn park(
    terrain: &Heightmap,
    spec: &CitySpec,
    rng: &mut SplitMix64,
    corner: Vec2,
    to_world: &impl Fn(Vec2) -> Vec2,
    rot: f32,
    out: &mut Vec<Obstacle>,
) {
    // Trees in a loose staggered pattern, one every ~6 m, away from the streets.
    let spacing = 6.0;
    let n = (spec.block_m / spacing).floor().max(1.0) as i32;
    for j in 0..n {
        for i in 0..n {
            if rng.next_f32() < 0.35 {
                continue;
            }
            let local = corner
                + (Vec2::new(i as f32, j as f32) + 0.5) * (spec.block_m / n as f32)
                + Vec2::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
            let p = to_world(local);
            if p.length() < spec.clear_radius_m + 1.0 {
                continue;
            }
            let shape = ObstacleShape::Cylinder { radius: rng.range(0.25, 0.6), height: rng.range(4.0, 9.0) };
            out.push(place(terrain, shape, p.x, p.y, rot));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObstaclesSpec, TerrainSpec, World, WorldSpec};

    pub(crate) fn downtown() -> CitySpec {
        CitySpec {
            seed: 3,
            block_m: 24.0,
            street_m: 8.0,
            lots: 3,
            alley_m: 1.5,
            height_m: (8.0, 30.0),
            density: 0.9,
            park_prob: 0.1,
            tower_prob: 0.1,
            rotation_deg: 20.0,
            jitter_deg: 0.0,
            jitter_m: 0.0,
            clear_radius_m: 10.0,
        }
    }

    fn city(spec: CitySpec) -> World {
        World::build(&WorldSpec {
            size_m: 200.0,
            cell_m: 2.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::City(spec),
            wind: Default::default(),
        })
        .unwrap()
    }

    #[test]
    fn a_dense_city_with_a_clear_start() {
        let w = city(downtown());
        assert!(w.obstacles.len() > 200, "{} obstacles", w.obstacles.len());
        assert!(!w.collides(Vec3::new(0.0, 2.0, 0.0), 2.0));
        // The middle of a street (between two blocks) is clear at flight height, far from the start.
        let spec = downtown();
        let pitch = spec.block_m + spec.street_m;
        let rot = spec.rotation_deg.to_radians();
        let (s, c) = rot.sin_cos();
        let local = Vec2::new(2.0 * pitch, 1.4 * pitch);
        let p = Vec2::new(local.x * c + local.y * s, -local.x * s + local.y * c);
        assert!(!w.collides(Vec3::new(p.x, 5.0, p.y), spec.street_m / 2.0 - 0.5), "street blocked at {p}");
    }

    #[test]
    fn same_seed_same_city() {
        assert_eq!(city(downtown()).obstacles, city(downtown()).obstacles);
        let other = CitySpec { seed: 4, ..downtown() };
        assert_ne!(city(downtown()).obstacles, city(other).obstacles);
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let too_many = CitySpec { lots: 6, alley_m: 5.0, ..downtown() };
        assert!(too_many.validate().is_err());
        assert!(CitySpec { density: 1.5, ..downtown() }.validate().is_err());
        assert!(downtown().validate().is_ok());
    }

    #[test]
    fn shipped_city_maps_build() {
        for text in [
            include_str!("../../../maps/downtown.ron"),
            include_str!("../../../maps/suburbs.ron"),
            include_str!("../../../maps/old_town.ron"),
            include_str!("../../../maps/dense_city.ron"),
        ] {
            let w = World::build(&WorldSpec::from_ron(text).unwrap()).unwrap();
            let y = w.terrain.height_at(0.0, 0.0) + 2.0;
            assert!(w.obstacles.len() > 50, "{} obstacles", w.obstacles.len());
            assert!(!w.collides(Vec3::new(0.0, y, 0.0), 1.5));
        }
    }

    /// The indexed raycast finds exactly what the brute-force walk finds.
    #[test]
    fn indexed_raycast_matches_brute_force() {
        let w = city(downtown());
        let mut brute = w.clone();
        brute.index.count = usize::MAX; // forces the brute-force walk
        let mut rng = SplitMix64::new(9);
        for _ in 0..3000 {
            let origin = Vec3::new(rng.range(-90.0, 90.0), rng.range(0.5, 40.0), rng.range(-90.0, 90.0));
            let a = rng.range(0.0, std::f32::consts::TAU);
            let e = rng.range(-1.2, 0.6);
            let dir = Vec3::new(a.cos() * e.cos(), e.sin(), a.sin() * e.cos());
            let (h1, h2) = (w.raycast(origin, dir, 60.0), brute.raycast(origin, dir, 60.0));
            assert_eq!(h1.map(|h| h.distance), h2.map(|h| h.distance), "{origin} {dir}");
            assert_eq!(w.collides(origin, 1.0), brute.collides(origin, 1.0));
            assert_eq!(w.column(origin.x, origin.z, 1.5, 2.5), brute.column(origin.x, origin.z, 1.5, 2.5));
        }
    }
}
