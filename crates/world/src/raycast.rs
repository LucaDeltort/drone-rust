//! Ray casting against the world: the only way sensors read it.
//!
//! Analytic tests; only the obstacles of the cells crossed by the ray are
//! tested (spatial index), from nearest to farthest.

use glam::{Quat, Vec3};

use crate::{Material, Obstacle, ObstacleShape, World};

/// Hit point of a ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Distance from the ray origin, in m.
    pub distance: f32,
    pub point: Vec3,
    /// Unit normal of the hit surface, facing the ray.
    pub normal: Vec3,
    pub material: Material,
}

impl World {
    /// First hit of the ray `origin + t·dir`, `0 <= t <= max_distance`.
    /// `dir` must be a unit vector.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_distance: f32) -> Option<Hit> {
        debug_assert!((dir.length() - 1.0).abs() < 1e-3, "direction not unit");
        let mut best = self.raycast_terrain(origin, dir, max_distance);
        let test = |o: &Obstacle, best: &mut Option<Hit>| {
            let limit = best.map_or(max_distance, |h| h.distance);
            if let Some(h) = ray_obstacle(o, origin, dir, limit) {
                *best = Some(h);
            }
        };
        if self.indexed() {
            // An obstacle can cover several cells: remember those already tested.
            let mut tested: Vec<u32> = Vec::new();
            let walked = self.index.walk(origin, dir, max_distance, |ids, t_exit| {
                for &k in ids {
                    if !tested.contains(&k) {
                        tested.push(k);
                        test(&self.obstacles[k as usize], &mut best);
                    }
                }
                // Hit before the cell exit: no farther cell can do better.
                best.is_some_and(|h| h.distance <= t_exit)
            });
            if walked {
                return best;
            }
        }
        for o in &self.obstacles {
            test(o, &mut best);
        }
        best
    }

    /// Marches along the ray by half cells, then refines by bisection.
    /// The terrain only exists on the map: beyond the edges, nothing is hit.
    fn raycast_terrain(&self, origin: Vec3, dir: Vec3, max_distance: f32) -> Option<Hit> {
        let t = &self.terrain;
        let half = t.half_size();
        let inside = |p: Vec3| p.x.abs() <= half && p.z.abs() <= half;
        let below = |p: Vec3| p.y <= t.height_at(p.x, p.z);

        if inside(origin) && below(origin) {
            return Some(Hit {
                distance: 0.0,
                point: origin,
                normal: -dir,
                material: Material::Concrete,
            });
        }
        let step = t.cell_m() * 0.5;
        // Above the highest point, the ray cannot touch the ground: start the march
        // where it comes down to that height, if it ever does.
        let mut prev = 0.0_f32;
        let top = t.max_height();
        if origin.y > top {
            if dir.y >= 0.0 {
                return None;
            }
            prev = (origin.y - top) / -dir.y;
            if prev > max_distance {
                return None;
            }
        }
        let mut s = (prev + step).min(max_distance);
        loop {
            let p = origin + dir * s;
            if inside(p) && below(p) {
                // The crossing below the ground is between prev and s.
                let (mut lo, mut hi) = (prev, s);
                for _ in 0..20 {
                    let mid = 0.5 * (lo + hi);
                    let q = origin + dir * mid;
                    if inside(q) && below(q) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let point = origin + dir * hi;
                return Some(Hit {
                    distance: hi,
                    point,
                    normal: self.terrain_normal(point.x, point.z),
                    material: Material::Concrete,
                });
            }
            if s >= max_distance {
                return None;
            }
            prev = s;
            s = (s + step).min(max_distance);
        }
    }

    /// Terrain normal by finite differences.
    pub fn terrain_normal(&self, x: f32, z: f32) -> Vec3 {
        let e = self.terrain.cell_m() * 0.5;
        let h = |x, z| self.terrain.height_at(x, z);
        let dx = h(x + e, z) - h(x - e, z);
        let dz = h(x, z + e) - h(x, z - e);
        Vec3::new(-dx, 2.0 * e, -dz).normalize()
    }
}

fn ray_obstacle(o: &Obstacle, origin: Vec3, dir: Vec3, max: f32) -> Option<Hit> {
    ray_shape(o, origin, dir, max).map(|h| Hit { material: o.material, ..h })
}

fn ray_shape(o: &Obstacle, origin: Vec3, dir: Vec3, max: f32) -> Option<Hit> {
    match o.shape {
        ObstacleShape::Box { size } => {
            let center = o.base + Vec3::Y * (size.y / 2.0);
            ray_box(center, Quat::from_rotation_y(o.yaw), size / 2.0, origin, dir, max)
        }
        ObstacleShape::Cylinder { radius, height } => ray_cylinder(o.base, radius, height, origin, dir, max),
        ObstacleShape::Roof { size } => {
            let rot = Quat::from_rotation_y(o.yaw);
            let inv = rot.inverse();
            let (lo, ld) = (inv * (origin - o.base), inv * dir);
            ray_convex(&roof_planes(size), lo, ld, max).map(|(t, n)| Hit {
                distance: t,
                point: origin + dir * t,
                normal: rot * n,
                material: Material::Concrete,
            })
        }
    }
}

/// Faces of a gable roof in its frame (base center at the origin), as half-spaces
/// `n·p <= d` with unit `n`: floor, two gable ends, two slopes.
pub(crate) fn roof_planes(size: Vec3) -> [(Vec3, f32); 5] {
    let (hx, h, hz) = (size.x / 2.0, size.y, size.z / 2.0);
    let slope = |s: f32| {
        let n = Vec3::new(0.0, 1.0, s * h / hz);
        let len = n.length();
        (n / len, h / len)
    };
    [(Vec3::NEG_Y, 0.0), (Vec3::X, hx), (Vec3::NEG_X, hx), slope(1.0), slope(-1.0)]
}

/// Ray against a convex polyhedron (Cyrus-Beck clipping). Returns the distance and
/// the outward normal of the entry face; distance 0 if the origin is inside.
fn ray_convex(planes: &[(Vec3, f32)], origin: Vec3, dir: Vec3, max: f32) -> Option<(f32, Vec3)> {
    let (mut t_in, mut t_out) = (0.0_f32, max);
    let mut normal = -dir;
    for &(n, d) in planes {
        let dist = d - n.dot(origin);
        let rate = n.dot(dir);
        if rate.abs() < 1e-9 {
            if dist < 0.0 {
                return None;
            }
            continue;
        }
        let t = dist / rate;
        if rate < 0.0 {
            // Entering through this face.
            if t > t_in {
                t_in = t;
                normal = n;
            }
        } else {
            t_out = t_out.min(t);
        }
        if t_in > t_out {
            return None;
        }
    }
    Some((t_in, normal))
}

/// Oriented box: move into its frame, then slab method.
fn ray_box(center: Vec3, rot: Quat, half: Vec3, origin: Vec3, dir: Vec3, max: f32) -> Option<Hit> {
    let inv = rot.inverse();
    let o = inv * (origin - center);
    let d = inv * dir;
    let (mut t_near, mut t_far) = (f32::NEG_INFINITY, f32::INFINITY);
    let mut axis_near = 0;
    for i in 0..3 {
        if d[i].abs() < 1e-9 {
            if o[i].abs() > half[i] {
                return None;
            }
            continue;
        }
        let a = (-half[i] - o[i]) / d[i];
        let b = (half[i] - o[i]) / d[i];
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if a > t_near {
            t_near = a;
            axis_near = i;
        }
        t_far = t_far.min(b);
        if t_near > t_far {
            return None;
        }
    }
    if t_far < 0.0 {
        return None;
    }
    // Origin inside the box: immediate hit.
    let (t, normal) = if t_near < 0.0 {
        (0.0, -dir)
    } else {
        let mut n = Vec3::ZERO;
        n[axis_near] = -d[axis_near].signum();
        (t_near, rot * n)
    };
    (t <= max).then(|| Hit {
        distance: t,
        point: origin + dir * t,
        normal,
        material: Material::Concrete,
    })
}

/// Closed vertical cylinder: side wall, then top and bottom disks.
fn ray_cylinder(base: Vec3, r: f32, h: f32, origin: Vec3, dir: Vec3, max: f32) -> Option<Hit> {
    let rel = origin - base;
    let (y0, y1) = (0.0, h);
    let inside_xz = |p: Vec3| p.x * p.x + p.z * p.z <= r * r;
    if inside_xz(rel) && (y0..=y1).contains(&rel.y) {
        return Some(Hit {
            distance: 0.0,
            point: origin,
            normal: -dir,
            material: Material::Concrete,
        });
    }
    let mut best: Option<(f32, Vec3)> = None;
    let mut keep = |t: f32, n: Vec3| {
        if t >= 0.0 && t <= max && best.is_none_or(|(b, _)| t < b) {
            best = Some((t, n));
        }
    };

    // Side wall: |rel_xz + t·d_xz|² = r².
    let a = dir.x * dir.x + dir.z * dir.z;
    if a > 1e-12 {
        let b = rel.x * dir.x + rel.z * dir.z;
        let c = rel.x * rel.x + rel.z * rel.z - r * r;
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let t = (-b - disc.sqrt()) / a;
            let p = rel + dir * t;
            if (y0..=y1).contains(&p.y) {
                keep(t, Vec3::new(p.x, 0.0, p.z).normalize());
            }
        }
    }
    // Disks.
    if dir.y.abs() > 1e-9 {
        for (y, n) in [(y1, Vec3::Y), (y0, Vec3::NEG_Y)] {
            let t = (y - rel.y) / dir.y;
            if inside_xz(rel + dir * t) && n.dot(dir) < 0.0 {
                keep(t, n);
            }
        }
    }
    best.map(|(t, normal)| Hit {
        distance: t,
        point: origin + dir * t,
        normal,
        material: Material::Concrete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObstacleSpec, ObstaclesSpec, TerrainSpec, WorldSpec};

    fn world(obstacles: Vec<ObstacleSpec>) -> World {
        World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::List(obstacles),
            wind: Default::default(),
        })
        .unwrap()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn hits_flat_ground_straight_down() {
        let w = world(vec![]);
        let h = w.raycast(Vec3::new(3.0, 5.0, 2.0), Vec3::NEG_Y, 10.0).unwrap();
        assert!(close(h.distance, 5.0));
        assert!(close(h.normal.y, 1.0));
    }

    #[test]
    fn slanted_ray_hits_ground_at_expected_distance() {
        let w = world(vec![]);
        let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
        let h = w.raycast(Vec3::new(0.0, 2.0, 0.0), dir, 10.0).unwrap();
        assert!(close(h.distance, 2.0 * 2f32.sqrt()), "{}", h.distance);
    }

    #[test]
    fn misses_beyond_max_distance_and_outside_map() {
        let w = world(vec![]);
        assert!(w.raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 4.0).is_none());
        // Horizontal: never the ground, and nothing beyond the edge.
        assert!(w.raycast(Vec3::new(0.0, 1.0, 0.0), Vec3::X, 500.0).is_none());
    }

    #[test]
    fn hits_box_face_with_normal() {
        let w = world(vec![ObstacleSpec {
            shape: ObstacleShape::Box { size: Vec3::new(2.0, 4.0, 2.0) },
            x: 10.0,
            z: 0.0,
            yaw_deg: 0.0,
            elevation_m: 0.0,
            material: Default::default(),
        }]);
        let h = w.raycast(Vec3::new(0.0, 1.0, 0.0), Vec3::X, 50.0).unwrap();
        assert!(close(h.distance, 9.0));
        assert!(close(h.normal.x, -1.0));
        // Above the box (4 m): passes.
        assert!(w.raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::X, 50.0).is_none());
    }

    #[test]
    fn rotated_box_is_wider_on_its_diagonal() {
        let w = world(vec![ObstacleSpec {
            shape: ObstacleShape::Box { size: Vec3::new(2.0, 4.0, 2.0) },
            x: 10.0,
            z: 0.0,
            yaw_deg: 45.0,
            elevation_m: 0.0,
            material: Default::default(),
        }]);
        let h = w.raycast(Vec3::new(0.0, 1.0, 0.0), Vec3::X, 50.0).unwrap();
        assert!(close(h.distance, 10.0 - 2f32.sqrt()), "{}", h.distance);
    }

    #[test]
    fn hits_cylinder_side_and_top() {
        let w = world(vec![ObstacleSpec {
            shape: ObstacleShape::Cylinder { radius: 1.0, height: 3.0 },
            x: 0.0,
            z: -10.0,
            yaw_deg: 0.0,
            elevation_m: 0.0,
            material: Default::default(),
        }]);
        let side = w.raycast(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z, 50.0).unwrap();
        assert!(close(side.distance, 9.0));
        assert!(close(side.normal.z, 1.0));
        let top = w.raycast(Vec3::new(0.0, 10.0, -10.0), Vec3::NEG_Y, 50.0).unwrap();
        assert!(close(top.distance, 7.0));
        assert!(close(top.normal.y, 1.0));
    }

    #[test]
    fn hits_roof_slope_and_passes_beside_the_ridge() {
        let w = world(vec![ObstacleSpec {
            shape: ObstacleShape::Roof { size: Vec3::new(10.0, 3.0, 6.0) },
            x: 0.0,
            z: 0.0,
            yaw_deg: 0.0,
            elevation_m: 4.0,
            material: Default::default(),
        }]);
        // Straight down onto the ridge: top at 4 + 3 = 7 m.
        let h = w.raycast(Vec3::new(0.0, 20.0, 0.0), Vec3::NEG_Y, 50.0).unwrap();
        assert!(close(h.distance, 13.0), "{}", h.distance);
        // Halfway down the slope: 4 + 1.5 m, normal leaning towards +Z.
        let h = w.raycast(Vec3::new(1.0, 20.0, 1.5), Vec3::NEG_Y, 50.0).unwrap();
        assert!(close(h.point.y, 5.5), "{}", h.point.y);
        assert!(h.normal.z > 0.3 && h.normal.y > 0.5);
        // Under the roof (elevated): the ray reaches the ground.
        let h = w.raycast(Vec3::new(-10.0, 2.0, 0.0), Vec3::X, 50.0);
        assert!(h.is_none());
        // Horizontal ray into the gable end.
        let h = w.raycast(Vec3::new(-10.0, 5.0, 0.0), Vec3::X, 50.0).unwrap();
        assert!(close(h.distance, 5.0) && close(h.normal.x, -1.0));
        // Column: halfway down the slope, the roof tops out at 5.5 m.
        let c = w.column(0.0, 1.5, 5.0, 6.0);
        assert!(c.matter && c.space);
    }

    #[test]
    fn nearest_obstacle_wins() {
        let cyl = |z| ObstacleSpec {
            shape: ObstacleShape::Cylinder { radius: 0.5, height: 3.0 },
            x: 0.0,
            z,
            yaw_deg: 0.0,
            elevation_m: 0.0,
            material: Default::default(),
        };
        let w = world(vec![cyl(-20.0), cyl(-8.0)]);
        let h = w.raycast(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z, 50.0).unwrap();
        assert!(close(h.distance, 7.5));
    }

    #[test]
    fn noisy_terrain_hit_lies_on_surface() {
        let w = World::build(&WorldSpec::procedural(11)).unwrap();
        let dir = Vec3::new(0.3, -1.0, 0.2).normalize();
        let h = w.raycast(Vec3::new(0.0, 30.0, 0.0), dir, 100.0).unwrap();
        let ground = w.terrain.height_at(h.point.x, h.point.z);
        // An obstacle may be hit before the ground; otherwise the point is on the ground.
        if (h.point.y - ground).abs() > 1e-2 {
            assert!(h.point.y > ground);
        }
        assert!(h.normal.y > 0.0 || h.normal.length() > 0.99);
    }

    /// From above the highest point of the relief: rays going up touch nothing, rays
    /// coming down land on the ground (the march starts at the top height).
    #[test]
    fn rays_from_above_the_relief() {
        let w = World::build(&crate::WorldSpec::procedural(3)).unwrap();
        let top = w.terrain.max_height();
        let mut rng = crate::rng::SplitMix64::new(4);
        for _ in 0..500 {
            let o = Vec3::new(rng.range(-60.0, 60.0), top + rng.range(0.5, 20.0), rng.range(-60.0, 60.0));
            let a = rng.range(0.0, std::f32::consts::TAU);
            let e = rng.range(0.05, 1.4);
            let up = Vec3::new(a.cos() * e.cos(), e.sin(), a.sin() * e.cos());
            assert!(w.raycast_terrain(o, up, 200.0).is_none());
            let down = Vec3::new(up.x, -up.y, up.z);
            if let Some(h) = w.raycast_terrain(o, down, 200.0) {
                assert!((h.point.y - w.terrain.height_at(h.point.x, h.point.z)).abs() < 0.01, "{o} {down} {h:?}");
            }
        }
    }

    /// Raycast cost on a dense city, split between terrain and obstacles
    /// (`cargo test --release -p world raycast_cost -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn raycast_cost() {
        use std::time::Instant;
        let spec = crate::WorldSpec::from_ron(include_str!("../../../maps/dense_city.ron")).unwrap();
        let w = World::build(&spec).unwrap();
        // A 16 × 360 lidar, 25 m, from 20 points along the streets, 2 m up.
        let mut rays = Vec::new();
        for k in 0..20 {
            let a = k as f32 * 0.7;
            let o = Vec3::new(30.0 * a.cos(), 0.0, 30.0 * a.sin());
            let o = Vec3::new(o.x, w.terrain.height_at(o.x, o.z) + 2.0, o.z);
            for c in 0..16 {
                let e = (20.0 - 60.0 * c as f32 / 15.0).to_radians();
                for r in 0..360 {
                    let az = (r as f32).to_radians();
                    rays.push((o, Vec3::new(-az.sin() * e.cos(), e.sin(), -az.cos() * e.cos())));
                }
            }
        }
        // Best of 5: the first passes warm the caches.
        let time = |f: &dyn Fn(Vec3, Vec3) -> Option<Hit>| {
            let mut best = f64::INFINITY;
            let mut hits = 0;
            for _ in 0..5 {
                let t = Instant::now();
                hits = 0;
                for &(o, d) in &rays {
                    hits += f(o, d).is_some() as usize;
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e9 / rays.len() as f64);
            }
            (best, hits)
        };
        let (all, h) = time(&|o, d| w.raycast(o, d, 25.0));
        let (terrain, _) = time(&|o, d| w.raycast_terrain(o, d, 25.0));
        let (obstacles, _) = time(&|o, d| {
            let mut best = None;
            for ob in &w.obstacles {
                let lim = best.map_or(25.0, |h: Hit| h.distance);
                if let Some(h) = ray_obstacle(ob, o, d, lim) {
                    best = Some(h);
                }
            }
            best
        });
        println!("{} rays, {} hits, {} obstacles: raycast {all:.0} ns, terrain alone {terrain:.0} ns, all obstacles brute {obstacles:.0} ns", rays.len(), h, w.obstacles.len());
    }
}
