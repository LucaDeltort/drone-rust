//! Collision detection (ground truth): does the drone's sphere touch the world,
//! and how to push it back out.

use glam::{Quat, Vec3};

use crate::{Obstacle, ObstacleShape, World};

impl World {
    /// Does the sphere (`center`, `radius`) touch the ground or an obstacle?
    pub fn collides(&self, center: Vec3, radius: f32) -> bool {
        // Ground: compare with the highest point under the footprint (center and 4 edges).
        let ground = [(0.0, 0.0), (radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)]
            .iter()
            .map(|(dx, dz)| self.terrain.height_at(center.x + dx, center.z + dz))
            .fold(f32::NEG_INFINITY, f32::max);
        if center.y - radius < ground - 1e-3 {
            return true;
        }
        if self.indexed() {
            self.index.near(center.x, center.z, radius).any(|k| distance_to(&self.obstacles[k as usize], center) < radius)
        } else {
            self.obstacles.iter().any(|o| distance_to(o, center) < radius)
        }
    }
}

/// Contact of a sphere with an obstacle: outward unit normal, penetration depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    pub normal: Vec3,
    pub depth: f32,
}

impl World {
    /// Deepest contact of the sphere (`center`, `radius`) with an obstacle, if any.
    /// The ground is left to the caller (it clamps on the height map).
    pub fn contact(&self, center: Vec3, radius: f32) -> Option<Contact> {
        let mut best: Option<Contact> = None;
        let mut test = |o: &Obstacle| {
            if let Some(c) = contact_with(o, center, radius)
                && best.is_none_or(|b| c.depth > b.depth)
            {
                best = Some(c);
            }
        };
        if self.indexed() {
            self.index.near(center.x, center.z, radius).for_each(|k| test(&self.obstacles[k as usize]));
        } else {
            self.obstacles.iter().for_each(test);
        }
        best
    }
}

fn contact_with(o: &Obstacle, p: Vec3, r: f32) -> Option<Contact> {
    let rot = Quat::from_rotation_y(o.yaw);
    match o.shape {
        ObstacleShape::Box { size } => {
            let center = o.base + Vec3::Y * (size.y / 2.0);
            let local = rot.inverse() * (p - center);
            let half = size / 2.0;
            let closest = local.clamp(-half, half);
            let d = (local - closest).length();
            if d >= r {
                return None;
            }
            if d > 1e-6 {
                return Some(Contact { normal: rot * ((local - closest) / d), depth: r - d });
            }
            // Center inside: out through the nearest face.
            let gap = half - local.abs();
            let axis = if gap.x <= gap.y && gap.x <= gap.z { 0 } else if gap.y <= gap.z { 1 } else { 2 };
            let mut n = Vec3::ZERO;
            n[axis] = if local[axis] >= 0.0 { 1.0 } else { -1.0 };
            Some(Contact { normal: rot * n, depth: r + gap[axis] })
        }
        ObstacleShape::Cylinder { radius, height } => {
            let rel = p - o.base;
            let radial = Vec3::new(rel.x, 0.0, rel.z);
            let rl = radial.length();
            let out_r = if rl > 1e-6 { radial / rl } else { Vec3::X };
            let dr = rl - radius;
            let dy = if rel.y < 0.0 { -rel.y } else if rel.y > height { rel.y - height } else { 0.0 };
            if dr > 0.0 || dy > 0.0 {
                let d = (dr.max(0.0).powi(2) + dy * dy).sqrt();
                if d >= r {
                    return None;
                }
                let vy = if rel.y < 0.0 { -dy } else { dy };
                let n = (out_r * dr.max(0.0) + Vec3::Y * vy) / d.max(1e-6);
                return Some(Contact { normal: n, depth: r - d });
            }
            // Inside: nearest of side, top, bottom.
            let (side, top, bottom) = (-dr, height - rel.y, rel.y);
            let (n, gap) = if side <= top && side <= bottom {
                (out_r, side)
            } else if top <= bottom {
                (Vec3::Y, top)
            } else {
                (Vec3::NEG_Y, bottom)
            };
            Some(Contact { normal: n, depth: r + gap })
        }
        ObstacleShape::Roof { size } => {
            let local = rot.inverse() * (p - o.base);
            let closest = roof_closest(size, local);
            let d = local.distance(closest);
            if d >= r {
                return None;
            }
            if d > 1e-6 {
                return Some(Contact { normal: rot * ((local - closest) / d), depth: r - d });
            }
            // Center inside: out through the nearest face.
            let (n, s) = crate::raycast::roof_planes(size)
                .iter()
                .map(|&(n, d)| (n, n.dot(local) - d))
                .max_by(|a, b| a.1.total_cmp(&b.1))?;
            Some(Contact { normal: rot * n, depth: r - s })
        }
    }
}

/// Closest point of a gable roof (its frame, base center at the origin) to `p`:
/// a triangular cross-section (z, y) extruded along x. `p` itself if inside.
fn roof_closest(size: Vec3, p: Vec3) -> Vec3 {
    let (hx, h, hz) = (size.x / 2.0, size.y, size.z / 2.0);
    let x = p.x.clamp(-hx, hx);
    let q = glam::Vec2::new(p.z, p.y);
    let (a, b, c) = (glam::Vec2::new(-hz, 0.0), glam::Vec2::new(hz, 0.0), glam::Vec2::new(0.0, h));
    // Inside the triangle: above the base, under both slopes.
    let inside = q.y >= 0.0 && q.y <= h * (1.0 - q.x.abs() / hz);
    let on = if inside {
        q
    } else {
        let seg = |u: glam::Vec2, v: glam::Vec2| {
            let t = ((q - u).dot(v - u) / (v - u).length_squared()).clamp(0.0, 1.0);
            u + (v - u) * t
        };
        [seg(a, b), seg(b, c), seg(c, a)].into_iter().min_by(|m, n| m.distance(q).total_cmp(&n.distance(q))).unwrap()
    };
    Vec3::new(x, on.y, on.x)
}

/// Distance from a point to an obstacle (0 inside).
fn distance_to(o: &Obstacle, p: Vec3) -> f32 {
    match o.shape {
        ObstacleShape::Roof { size } => {
            let local = Quat::from_rotation_y(o.yaw).inverse() * (p - o.base);
            local.distance(roof_closest(size, local))
        }
        ObstacleShape::Box { size } => {
            let center = o.base + Vec3::Y * (size.y / 2.0);
            let local = Quat::from_rotation_y(o.yaw).inverse() * (p - center);
            let half = size / 2.0;
            let closest = local.clamp(-half, half);
            (local - closest).length()
        }
        ObstacleShape::Cylinder { radius, height } => {
            let rel = p - o.base;
            let radial = (rel.x * rel.x + rel.z * rel.z).sqrt();
            let dr = (radial - radius).max(0.0);
            let dy = if rel.y < 0.0 {
                -rel.y
            } else if rel.y > height {
                rel.y - height
            } else {
                0.0
            };
            (dr * dr + dy * dy).sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainSpec, World, WorldSpec};
    use glam::Vec3;

    fn world() -> World {
        World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::List(vec![
                ObstacleSpec {
                    shape: ObstacleShape::Box { size: Vec3::new(2.0, 4.0, 2.0) },
                    x: 10.0,
                    z: 0.0,
                    yaw_deg: 45.0,
                    elevation_m: 0.0,
                    material: Default::default(),
                },
                ObstacleSpec {
                    shape: ObstacleShape::Cylinder { radius: 1.0, height: 3.0 },
                    x: -10.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                    elevation_m: 0.0,
                    material: Default::default(),
                },
            ]),
            wind: Default::default(),
        })
        .unwrap()
    }

    #[test]
    fn free_air_does_not_collide() {
        assert!(!world().collides(Vec3::new(0.0, 2.0, 0.0), 0.25));
    }

    #[test]
    fn ground_contact_collides() {
        let w = world();
        assert!(w.collides(Vec3::new(0.0, 0.2, 0.0), 0.25));
        assert!(!w.collides(Vec3::new(0.0, 0.26, 0.0), 0.25));
    }

    #[test]
    fn rotated_box_uses_its_diagonal() {
        let w = world();
        // 2 m box rotated by 45°: its corner is √2 from the center, along X.
        let corner = 10.0 - 2f32.sqrt();
        assert!(w.collides(Vec3::new(corner - 0.2, 2.0, 0.0), 0.25));
        assert!(!w.collides(Vec3::new(corner - 0.3, 2.0, 0.0), 0.25));
    }

    #[test]
    fn cylinder_side_and_top() {
        let w = world();
        assert!(w.collides(Vec3::new(-8.8, 1.5, 0.0), 0.25));
        assert!(!w.collides(Vec3::new(-8.7, 1.5, 0.0), 0.25));
        assert!(w.collides(Vec3::new(-10.0, 3.2, 0.0), 0.25));
        assert!(!w.collides(Vec3::new(-10.0, 3.3, 0.0), 0.25));
    }

    #[test]
    fn roof_eave_is_exact() {
        let w = World::build(&WorldSpec {
            size_m: 100.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Flat { height: 0.0 },
            obstacles: ObstaclesSpec::List(vec![ObstacleSpec {
                shape: ObstacleShape::Roof { size: Vec3::new(10.0, 3.0, 6.0) },
                x: 0.0,
                z: 0.0,
                yaw_deg: 0.0,
                elevation_m: 5.0,
                material: Default::default(),
            }]),
            wind: Default::default(),
        })
        .unwrap();
        // Eave at z = 3, y = 5: 0.6 m outside it, just below, nothing touches.
        assert!(!w.collides(Vec3::new(0.0, 4.9, 3.6), 0.3));
        assert!(w.contact(Vec3::new(0.0, 4.9, 3.6), 0.3).is_none());
        // On the slope and under the eave, it does.
        assert!(w.collides(Vec3::new(0.0, 6.6, 1.0), 0.3));
        assert!(w.collides(Vec3::new(0.0, 4.8, 2.0), 0.3));
    }
}
