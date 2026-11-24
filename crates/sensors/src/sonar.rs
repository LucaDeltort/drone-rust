use drone_spec::Mount;
use glam::{Quat, Vec3};
use world::{World, rng::SplitMix64};

use crate::{Data, Noise, Pose, Sensor};

const MIN_RANGE: f32 = 0.15;

/// Sonar: cast a few rays within the cone and keep the nearest echo.
/// A too oblique surface (incidence > `max_incidence`) returns nothing; a fairly
/// oblique one may send the sound on to a second surface first (multipath), and the
/// echo reads too far.
pub struct Sonar {
    mount: Mount,
    rate_hz: f32,
    /// Sampling directions of the cone, sensor frame (-Z axis).
    dirs: Vec<Vec3>,
    half_angle: f32,
    range: f32,
    cos_max_incidence: f32,
    multipath: f32,
    noise: Noise,
    rng: SplitMix64,
    label: String,
}

impl Sonar {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mount: Mount,
        rate_hz: f32,
        cone_deg: f32,
        range: f32,
        max_incidence_deg: f32,
        noise: Noise,
        rng: SplitMix64,
    ) -> Self {
        let half_angle = (cone_deg / 2.0).to_radians();
        // Axis, then two rings at half cone and at the cone edge.
        let mut dirs = vec![Vec3::NEG_Z];
        for (frac, count) in [(0.5, 8), (1.0, 12)] {
            let tilt = half_angle * frac;
            for k in 0..count {
                let a = std::f32::consts::TAU * k as f32 / count as f32;
                dirs.push(Quat::from_rotation_z(a) * Quat::from_rotation_x(tilt) * Vec3::NEG_Z);
            }
        }
        Self {
            mount,
            rate_hz,
            dirs,
            half_angle,
            range,
            cos_max_incidence: max_incidence_deg.to_radians().cos(),
            multipath: 0.0,
            noise,
            rng,
            label: format!("Sonar ({cone_deg} deg, {range} m)"),
        }
    }

    /// Chance of a multipath echo off an oblique surface (scaled by the obliquity).
    pub fn with_multipath(mut self, multipath: f32) -> Self {
        self.multipath = multipath;
        self
    }
}

impl Sensor for Sonar {
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
        let nearest = self
            .dirs
            .iter()
            .filter_map(|&d| {
                let dir = rotation * d;
                let hit = world.raycast(origin, dir, self.range)?;
                let facing = -dir.dot(hit.normal);
                (hit.distance >= MIN_RANGE && facing >= self.cos_max_incidence).then_some((hit, dir, facing))
            })
            .min_by(|a, b| a.0.distance.total_cmp(&b.0.distance));
        let nearest = nearest.map(|(hit, dir, facing)| {
            // The more oblique, the more of the sound goes on, mirrored.
            if self.multipath > 0.0 && self.rng.next_f32() < self.multipath * (1.0 - facing) {
                let mirrored = dir - 2.0 * dir.dot(hit.normal) * hit.normal;
                let from = hit.point + hit.normal * 0.01;
                if let Some(second) = world.raycast(from, mirrored, self.range) {
                    // Time of flight: out, bounce, back to the sensor; read as half.
                    return (hit.distance + second.distance + second.point.distance(origin)) / 2.0;
                }
            }
            hit.distance
        });
        let range = nearest
            .and_then(|d| self.noise.apply(&mut self.rng, d))
            .filter(|&d| d <= self.range);
        Data::Echo {
            range,
            cone_half_angle: self.half_angle,
            max_range: self.range,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::wall_world;

    fn sonar(cone_deg: f32) -> Sonar {
        let exact = Noise { std_m: 0.0, growth_per_m2: 0.0, dropout: 0.0 };
        Sonar::new(Mount::default(), 20.0, cone_deg, 5.0, 60.0, exact, SplitMix64::new(1))
    }

    fn echo(d: Data) -> Option<f32> {
        match d {
            Data::Echo { range, .. } => range,
            _ => panic!(),
        }
    }

    #[test]
    fn looking_down_reads_height() {
        let w = wall_world();
        let down = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        let r = echo(sonar(20.0).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), Vec3::new(0.0, 2.0, 0.0), down)).unwrap();
        assert!((r - 2.0).abs() < 1e-3, "{r}");
    }

    #[test]
    fn nearest_echo_in_cone_wins() {
        let w = wall_world();
        // Wall 3 m ahead, ground 2 m below a 120° cone: the cone edge hits the ground at
        // 2 / cos(30°) ≈ 2.31 m, closer than the wall. With 20°, only the wall answers.
        let pos = Vec3::new(0.0, 2.0, -7.0);
        let wide = Sonar::new(
            Mount::default(),
            20.0,
            120.0,
            5.0,
            80.0,
            Noise { std_m: 0.0, growth_per_m2: 0.0, dropout: 0.0 },
            SplitMix64::new(1),
        );
        let mut wide = wide;
        let r_wide = echo(wide.measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), pos, Quat::IDENTITY)).unwrap();
        let r_narrow = echo(sonar(20.0).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), pos, Quat::IDENTITY)).unwrap();
        assert!((r_narrow - 3.0).abs() < 1e-3, "{r_narrow}");
        assert!(r_wide < 2.5, "{r_wide}");
    }

    #[test]
    fn oblique_surface_gives_no_echo() {
        let w = wall_world();
        // Aimed 70° from vertical towards the ground: incidence 70° > 60°.
        let rot = Quat::from_rotation_x(-20f32.to_radians());
        let pos = Vec3::new(0.0, 1.0, 0.0);
        assert!(echo(sonar(4.0).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), pos, rot)).is_none());
        // Same aim, steeper (incidence 40°): echo.
        let steep = Quat::from_rotation_x(-50f32.to_radians());
        assert!(echo(sonar(4.0).measure(&w, &Pose::new(Vec3::ZERO, Quat::IDENTITY), pos, steep)).is_some());
    }

    #[test]
    fn an_oblique_echo_can_bounce_and_read_too_far() {
        let w = wall_world();
        // Aimed 45° down at the ground, 2 m up, 3 m from the wall: the sound can
        // bounce off the ground onto the wall and come back late.
        let rot = Quat::from_rotation_x(-45f32.to_radians());
        let pos = Vec3::new(0.0, 2.0, -7.0);
        let body = Pose::new(Vec3::ZERO, Quat::IDENTITY);
        let mut clean = sonar(4.0);
        let mut bouncy = sonar(4.0).with_multipath(1.0);
        // The nearest ray of the cone: a little short of the axis's 2√2 m.
        let direct = echo(clean.measure(&w, &body, pos, rot)).unwrap();
        assert!((direct - 2.0 * 2f32.sqrt()).abs() < 0.15, "{direct}");
        let far = (0..200).filter(|_| echo(bouncy.measure(&w, &body, pos, rot)).is_some_and(|r| r > direct + 0.5)).count();
        // Incidence 45°: about 1 - cos 45° ≈ 29 % of the echoes bounce.
        assert!((40..=80).contains(&far), "{far}/200 bounced");
    }
}
