//! Ray casting against the world: the only way sensors read it.
//!
//! Analytic tests; only the obstacles of the cells crossed by the ray are
//! tested (spatial index), from nearest to farthest.

use glam::{Quat, Vec3};

use crate::{Obstacle, ObstacleShape, World};

/// Hit point of a ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Distance from the ray origin, in m.
    pub distance: f32,
    pub point: Vec3,
    /// Unit normal of the hit surface, facing the ray.
    pub normal: Vec3,
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
            });
        }
        let step = t.cell_m() * 0.5;
        let mut prev = 0.0_f32;
        let mut s = step.min(max_distance);
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
    match o.shape {
        ObstacleShape::Box { size } => {
            let center = o.base + Vec3::Y * (size.y / 2.0);
            ray_box(center, Quat::from_rotation_y(o.yaw), size / 2.0, origin, dir, max)
        }
        ObstacleShape::Cylinder { radius, height } => ray_cylinder(o.base, radius, height, origin, dir, max),
    }
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
        }]);
        let side = w.raycast(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z, 50.0).unwrap();
        assert!(close(side.distance, 9.0));
        assert!(close(side.normal.z, 1.0));
        let top = w.raycast(Vec3::new(0.0, 10.0, -10.0), Vec3::NEG_Y, 50.0).unwrap();
        assert!(close(top.distance, 7.0));
        assert!(close(top.normal.y, 1.0));
    }

    #[test]
    fn nearest_obstacle_wins() {
        let cyl = |z| ObstacleSpec {
            shape: ObstacleShape::Cylinder { radius: 0.5, height: 3.0 },
            x: 0.0,
            z,
            yaw_deg: 0.0,
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
}
