use drone_spec::Mount;
use glam::{Quat, Vec3};
use world::{World, rng::SplitMix64};

use crate::{Data, Noise, Pose, Sensor};

const MIN_RANGE: f32 = 0.2;

/// Depth camera: a grid of rays within the field of view, without
/// going through the GPU. Distance along each ray, noise growing with it.
pub struct DepthCamera {
    mount: Mount,
    rate_hz: f32,
    width: u32,
    height: u32,
    /// Pixel directions, sensor frame, row by row from the top left.
    dirs: Vec<Vec3>,
    range: f32,
    noise: Noise,
    rng: SplitMix64,
    label: String,
}

impl DepthCamera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mount: Mount,
        rate_hz: f32,
        width: u32,
        height: u32,
        fov_deg: f32,
        range: f32,
        noise: Noise,
        rng: SplitMix64,
    ) -> Self {
        let tan_h = (fov_deg.to_radians() / 2.0).tan();
        let tan_v = tan_h * height as f32 / width as f32;
        let mut dirs = Vec::with_capacity((width * height) as usize);
        for j in 0..height {
            for i in 0..width {
                let u = ((i as f32 + 0.5) / width as f32 * 2.0 - 1.0) * tan_h;
                let v = (1.0 - (j as f32 + 0.5) / height as f32 * 2.0) * tan_v;
                dirs.push(Vec3::new(u, v, -1.0).normalize());
            }
        }
        Self {
            mount,
            rate_hz,
            width,
            height,
            dirs,
            range,
            noise,
            rng,
            label: format!("Depth ({width}x{height}, {fov_deg} deg, {range} m)"),
        }
    }
}

impl Sensor for DepthCamera {
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
        let beams = crate::cast_beams(world, |_| (origin, rotation), &self.dirs, (MIN_RANGE, self.range), &self.noise, &mut self.rng);
        Data::DepthImage {
            width: self.width,
            height: self.height,
            beams,
            max_range: self.range,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::wall_world;

    const EXACT: Noise = Noise { std_m: 0.0, growth_per_m2: 0.0, dropout: 0.0 };

    fn cam(noise: Noise) -> DepthCamera {
        DepthCamera::new(Mount::default(), 10.0, 9, 5, 90.0, 20.0, noise, SplitMix64::new(1))
    }

    #[test]
    fn center_pixel_sees_wall_straight_ahead() {
        let w = wall_world();
        let Data::DepthImage { width, height, beams, .. } = cam(EXACT).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), Vec3::new(0.0, 5.0, 0.0), Quat::IDENTITY)
        else {
            panic!()
        };
        assert_eq!(beams.len(), (width * height) as usize);
        let center = beams[(2 * width + 4) as usize];
        assert!(center.dir.abs_diff_eq(Vec3::NEG_Z, 1e-5));
        assert!((center.range.unwrap() - 10.0).abs() < 1e-3);
    }

    #[test]
    fn top_row_looks_up_bottom_row_down() {
        let c = cam(EXACT);
        assert!(c.dirs[0].y > 0.0 && c.dirs[0].x < 0.0);
        assert!(c.dirs.last().unwrap().y < 0.0 && c.dirs.last().unwrap().x > 0.0);
    }

    #[test]
    fn noise_grows_with_distance() {
        let w = wall_world();
        let noise = Noise { std_m: 0.0, growth_per_m2: 0.01, dropout: 0.0 };
        let spread = |pos: Vec3| {
            let mut c = cam(noise);
            let xs: Vec<f32> = (0..400)
                .filter_map(|_| match c.measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), pos, Quat::IDENTITY) {
                    Data::DepthImage { beams, width, .. } => beams[(2 * width + 4) as usize].range,
                    _ => None,
                })
                .collect();
            let m = xs.iter().sum::<f32>() / xs.len() as f32;
            (xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32).sqrt()
        };
        let near = spread(Vec3::new(0.0, 5.0, -8.0)); // wall at 2 m
        let far = spread(Vec3::new(0.0, 5.0, 0.0)); // wall at 10 m
        assert!(far > near * 10.0, "near {near}, far {far}");
    }
}
