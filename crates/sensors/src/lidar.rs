use drone_spec::Mount;
use glam::{Quat, Vec3};
use world::{World, rng::SplitMix64};

use crate::{Data, Noise, Pose, Sensor};

/// Distance below which the lidar sees nothing, in m.
const MIN_RANGE: f32 = 0.1;

/// Horizontal fan of `rays` rays over 360°, the first one forward (-Z),
/// then counter-clockwise seen from above.
pub struct Lidar2D {
    mount: Mount,
    rate_hz: f32,
    dirs: Vec<Vec3>,
    range: f32,
    noise: Noise,
    rng: SplitMix64,
    label: String,
}

impl Lidar2D {
    pub fn new(mount: Mount, rate_hz: f32, rays: u32, range: f32, noise: Noise, rng: SplitMix64) -> Self {
        let dirs = (0..rays)
            .map(|k| {
                let a = std::f32::consts::TAU * k as f32 / rays as f32;
                Vec3::new(-a.sin(), 0.0, -a.cos())
            })
            .collect();
        Self {
            mount,
            rate_hz,
            dirs,
            range,
            noise,
            rng,
            label: format!("2D lidar ({rays} rays, {range} m)"),
        }
    }
}

impl Sensor for Lidar2D {
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

    fn measure(&mut self, world: &World, body: &Pose, origin: Vec3, rotation: Quat) -> Data {
        self.measure_sweep(world, body, (origin, rotation), (origin, rotation))
    }

    fn spinning(&self) -> bool {
        true
    }

    fn measure_sweep(&mut self, world: &World, _body: &Pose, start: (Vec3, Quat), end: (Vec3, Quat)) -> Data {
        let n = self.dirs.len() as f32;
        let pose = |i: usize| crate::sweep_pose(start, end, (i as f32 + 0.5) / n);
        let beams = crate::cast_beams(world, pose, &self.dirs, (MIN_RANGE, self.range), &self.noise, &mut self.rng);
        Data::Scan { beams, max_range: self.range }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::wall_world;

    fn lidar(noise: Noise) -> Lidar2D {
        Lidar2D::new(Mount::default(), 10.0, 360, 12.0, noise, SplitMix64::new(1))
    }

    const EXACT: Noise = Noise { std_m: 0.0, growth_per_m2: 0.0, dropout: 0.0 };

    #[test]
    fn front_beam_sees_wall_back_beam_sees_nothing() {
        let w = wall_world();
        let Data::Scan { beams, .. } = lidar(EXACT).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY) else {
            panic!()
        };
        assert_eq!(beams.len(), 360);
        assert!((beams[0].range.unwrap() - 10.0).abs() < 1e-3);
        assert!(beams[180].range.is_none());
        // 30° to the left: the wall is at 10 / cos(30°).
        let expected = 10.0 / 30f32.to_radians().cos();
        assert!((beams[30].range.unwrap() - expected).abs() < 1e-3);
    }

    #[test]
    fn yaw_rotates_the_scan() {
        let w = wall_world();
        // Drone turned 90° to the left: the wall is now on the right (ray 270).
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let Data::Scan { beams, .. } = lidar(EXACT).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), Vec3::new(0.0, 2.0, 0.0), rot) else { panic!() };
        assert!((beams[270].range.unwrap() - 10.0).abs() < 1e-3);
        assert!(beams[0].range.is_none());
    }

    #[test]
    fn full_dropout_loses_everything() {
        let w = wall_world();
        let mut l = lidar(Noise { dropout: 1.0, ..EXACT });
        let Data::Scan { beams, .. } = l.measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY) else { panic!() };
        assert!(beams.iter().all(|b| b.range.is_none()));
    }
}
