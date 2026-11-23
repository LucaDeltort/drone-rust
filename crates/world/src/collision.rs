//! Collision detection (ground truth): does the drone's sphere touch the world?
//! Used to count collisions, not to prevent them: physics will come later.

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

/// Distance from a point to an obstacle (0 inside).
fn distance_to(o: &Obstacle, p: Vec3) -> f32 {
    match o.shape {
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
                },
                ObstacleSpec {
                    shape: ObstacleShape::Cylinder { radius: 1.0, height: 3.0 },
                    x: -10.0,
                    z: 0.0,
                    yaw_deg: 0.0,
                },
            ]),
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
}
