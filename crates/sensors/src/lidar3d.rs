//! Spinning multi-channel lidar: a 3D point cloud, delivered as a depth image.

use drone_spec::Mount;
use glam::{Quat, Vec3};
use world::{World, rng::SplitMix64};

use crate::{Beam, Data, Noise, Pose, Sensor};

/// Distance below which the lidar sees nothing, in m.
const MIN_RANGE: f32 = 0.3;

/// `channels` rings from `fov_up` (first row) down to `fov_down`, each of `rays`
/// rays over 360°, the first one forward (-Z), then counter-clockwise seen from
/// above. Mounted on the stabilized frame (heading only), like the 2D lidar.
pub struct Lidar3D {
    mount: Mount,
    rate_hz: f32,
    width: u32,
    height: u32,
    dirs: Vec<Vec3>,
    range: f32,
    noise: Noise,
    rng: SplitMix64,
    label: String,
}

impl Lidar3D {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mount: Mount,
        rate_hz: f32,
        channels: u32,
        rays: u32,
        fov_up_deg: f32,
        fov_down_deg: f32,
        range: f32,
        noise: Noise,
        rng: SplitMix64,
    ) -> Self {
        let mut dirs = Vec::with_capacity((channels * rays) as usize);
        for c in 0..channels {
            let k = if channels == 1 { 0.5 } else { c as f32 / (channels - 1) as f32 };
            let e = (fov_up_deg + (fov_down_deg - fov_up_deg) * k).to_radians();
            for r in 0..rays {
                let a = std::f32::consts::TAU * r as f32 / rays as f32;
                dirs.push(Vec3::new(-a.sin() * e.cos(), e.sin(), -a.cos() * e.cos()));
            }
        }
        Self {
            mount,
            rate_hz,
            width: rays,
            height: channels,
            dirs,
            range,
            noise,
            rng,
            label: format!("3D lidar ({channels}×{rays}, {range} m)"),
        }
    }
}

impl Sensor for Lidar3D {
    fn label(&self) -> &str {
        &self.label
    }
    fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    fn mount(&self) -> &Mount {
        &self.mount
    }
    fn noise(&self) -> &Noise {
        &self.noise
    }
    fn noise_mut(&mut self) -> &mut Noise {
        &mut self.noise
    }

    fn measure(&mut self, world: &World, _body: &Pose, origin: Vec3, rotation: Quat) -> Data {
        let beams = self
            .dirs
            .iter()
            .map(|&dir| {
                let hit = world.raycast(origin, rotation * dir, self.range).map(|h| h.distance);
                Beam::measured(dir, hit, MIN_RANGE, self.range, &self.noise, &mut self.rng)
            })
            .collect();
        Data::DepthImage { width: self.width, height: self.height, beams, max_range: self.range }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::wall_world;

    #[test]
    fn rings_see_wall_ahead_and_ground_below() {
        let w = wall_world();
        let exact = Noise { std_m: 0.0, growth_per_m2: 0.0, dropout: 0.0 };
        let mut l = Lidar3D::new(Mount::default(), 10.0, 5, 36, 10.0, -30.0, 20.0, exact, SplitMix64::new(1));
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let Data::DepthImage { width, height, beams, .. } = l.measure(&w, &Pose::new(origin, Quat::IDENTITY), origin, Quat::IDENTITY)
        else {
            panic!()
        };
        assert_eq!((width, height, beams.len()), (36, 5, 180));
        // Top ring (+10°), forward: the wall at 10 m horizontally.
        let d = beams[0].range.unwrap();
        assert!((d - 10.0 / 10f32.to_radians().cos()).abs() < 1e-3, "{d}");
        // Bottom ring (-30°), backward: the ground at 2 / sin(30°) = 4 m.
        let d = beams[4 * 36 + 18].range.unwrap();
        assert!((d - 4.0).abs() < 1e-2, "{d}");
    }
}
