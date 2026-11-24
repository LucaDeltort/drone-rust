//! Wind: a deterministic field of position and time.
//!
//! - mean wind, stronger with height (power-law shear, 1/7 exponent);
//! - gusts: fronts travelling downwind at the mean speed;
//! - turbulence: 4D Perlin noise carried along by the mean wind (frozen
//!   turbulence), slowly evolving;
//! - shelter: downwind of an obstacle, the mean wind drops and the turbulence of
//!   the wake rises.
//!
//! Being a pure function of (position, time), the wind needs no state: the same
//! run gives the same gusts.

use glam::Vec3;
use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

/// Wind description. Compass convention: `from_deg` = where the wind comes
/// from, 0 = north (-Z), 90 = east (+X).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindSpec {
    /// Mean speed at 10 m above the ground, in m/s.
    pub speed_ms: f32,
    pub from_deg: f32,
    /// Gust amplitude, in m/s, added on top of the mean.
    pub gust_ms: f32,
    /// Turbulence standard deviation, in m/s.
    pub turbulence_ms: f32,
    pub seed: u32,
}

impl Default for WindSpec {
    fn default() -> Self {
        Self { speed_ms: 0.0, from_deg: 0.0, gust_ms: 0.0, turbulence_ms: 0.0, seed: 1 }
    }
}

impl WindSpec {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let ok = |v: f32| v.is_finite() && v >= 0.0;
        if !(ok(self.speed_ms) && ok(self.gust_ms) && ok(self.turbulence_ms) && self.from_deg.is_finite()) {
            return Err("wind speeds must be >= 0".into());
        }
        Ok(())
    }

    pub fn is_calm(&self) -> bool {
        self.speed_ms == 0.0 && self.gust_ms == 0.0 && self.turbulence_ms == 0.0
    }
}

/// Reference height of the mean speed, in m.
const REF_HEIGHT_M: f32 = 10.0;
/// Shear exponent (open terrain).
const SHEAR: f32 = 1.0 / 7.0;
/// Distance upwind within which an obstacle shelters, in m.
pub const SHELTER_M: f32 = 15.0;
/// Size of the turbulent eddies, in m.
const EDDY_M: f32 = 6.0;

#[derive(Clone)]
pub struct Wind {
    spec: WindSpec,
    noise: [Perlin; 3],
    /// Gust phases, drawn from the seed.
    phases: [f32; 3],
}

impl std::fmt::Debug for Wind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wind").field("spec", &self.spec).finish()
    }
}

impl Wind {
    pub fn new(spec: WindSpec) -> Self {
        let s = spec.seed;
        let mut rng = crate::rng::SplitMix64::new(s as u64);
        let tau = std::f32::consts::TAU;
        Self {
            spec,
            noise: [Perlin::new(s), Perlin::new(s.wrapping_add(1)), Perlin::new(s.wrapping_add(2))],
            phases: [rng.range(0.0, tau), rng.range(0.0, tau), rng.range(0.0, tau)],
        }
    }

    pub fn spec(&self) -> &WindSpec {
        &self.spec
    }

    /// Changes the wind (from the viewer, say). Same seed: same eddies.
    pub fn set(&mut self, spec: WindSpec) {
        if spec.seed != self.spec.seed {
            *self = Self::new(spec);
        } else {
            self.spec = spec;
        }
    }

    /// Horizontal unit vector the wind blows towards.
    pub fn heading(&self) -> Vec3 {
        let (s, c) = self.spec.from_deg.to_radians().sin_cos();
        // Coming from (sin, -cos): blowing the other way.
        Vec3::new(-s, 0.0, c)
    }

    /// Mean wind speed at `height` m above the ground.
    pub fn mean_speed(&self, height: f32) -> f32 {
        self.spec.speed_ms * (height.max(0.5) / REF_HEIGHT_M).powf(SHEAR)
    }

    /// Gust intensity in [0, 1] at `p`, time `t`: fronts travel downwind.
    pub fn gust(&self, p: Vec3, t: f32) -> f32 {
        let along = p.dot(self.heading());
        let s = t - along / self.spec.speed_ms.max(1.0);
        let tau = std::f32::consts::TAU;
        let g = 0.6 * (tau * s / 7.0 + self.phases[0]).sin()
            + 0.3 * (tau * s / 3.1 + self.phases[1]).sin()
            + 0.1 * (tau * s / 1.3 + self.phases[2]).sin();
        // Gusts are bursts: mostly calm, sometimes strong.
        (0.5 + 0.5 * g).powi(2)
    }

    /// Turbulence at `p`, time `t`, in m/s (zero mean).
    pub fn turbulence(&self, p: Vec3, t: f32) -> Vec3 {
        if self.spec.turbulence_ms == 0.0 {
            return Vec3::ZERO;
        }
        let drift = self.heading() * self.spec.speed_ms * t;
        let q = (p - drift) / EDDY_M;
        let at = |n: &Perlin| n.get([q.x as f64, q.y as f64, q.z as f64, t as f64 * 0.3]) as f32;
        // Perlin noise spans about ±0.5 in std: ×2 for a unit std; less vertically.
        Vec3::new(at(&self.noise[0]), 0.5 * at(&self.noise[1]), at(&self.noise[2])) * 2.0 * self.spec.turbulence_ms
    }

    /// Wind at `p` (`height` m above the ground), time `t`, with a `shelter`
    /// factor in [0, 1] (1 = open air, 0 = right behind a wall).
    pub fn at(&self, p: Vec3, height: f32, t: f32, shelter: f32) -> Vec3 {
        if self.spec.is_calm() {
            return Vec3::ZERO;
        }
        let shear = (height.max(0.5) / REF_HEIGHT_M).powf(SHEAR);
        let mean = (self.spec.speed_ms + self.spec.gust_ms * self.gust(p, t)) * shear;
        let wake = 1.0 + 1.5 * (1.0 - shelter);
        self.heading() * mean * (0.15 + 0.85 * shelter) + self.turbulence(p, t) * wake
    }
}

impl crate::World {
    /// Wind felt at `p`, time `t`, in m/s.
    pub fn wind_at(&self, p: Vec3, t: f32) -> Vec3 {
        let spec = self.wind.spec();
        if spec.is_calm() {
            return Vec3::ZERO;
        }
        let height = p.y - self.terrain.height_at(p.x, p.z);
        // Shelter: what is upwind, within SHELTER_M?
        let shelter = if spec.speed_ms + spec.gust_ms > 0.1 {
            self.raycast(p, -self.wind.heading(), SHELTER_M)
                .map_or(1.0, |h| (h.distance / SHELTER_M).clamp(0.0, 1.0))
        } else {
            1.0
        };
        self.wind.at(p, height, t, shelter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> WindSpec {
        WindSpec { speed_ms: 6.0, from_deg: 90.0, gust_ms: 0.0, turbulence_ms: 0.0, seed: 3 }
    }

    #[test]
    fn east_wind_blows_west() {
        let w = Wind::new(spec());
        assert!((w.heading() - Vec3::NEG_X).length() < 1e-5);
        let v = w.at(Vec3::new(0.0, 10.0, 0.0), 10.0, 0.0, 1.0);
        assert!((v - Vec3::new(-6.0, 0.0, 0.0)).length() < 1e-4, "{v}");
    }

    #[test]
    fn stronger_higher() {
        let w = Wind::new(spec());
        assert!(w.mean_speed(2.0) < w.mean_speed(10.0));
        assert!(w.mean_speed(30.0) > w.mean_speed(10.0));
    }

    #[test]
    fn calm_is_zero_and_shelter_weakens() {
        let w = Wind::new(WindSpec::default());
        assert_eq!(w.at(Vec3::ONE, 5.0, 3.0, 1.0), Vec3::ZERO);
        let w = Wind::new(spec());
        let open = w.at(Vec3::ONE, 5.0, 3.0, 1.0).length();
        let lee = w.at(Vec3::ONE, 5.0, 3.0, 0.0).length();
        assert!(lee < open * 0.3);
    }

    #[test]
    fn turbulence_has_the_requested_strength() {
        let w = Wind::new(WindSpec { turbulence_ms: 1.0, ..spec() });
        let n = 4000;
        let mut sq = 0.0;
        for k in 0..n {
            let p = Vec3::new(k as f32 * 1.7, 3.0, (k % 37) as f32 * 2.3);
            sq += w.turbulence(p, k as f32 * 0.05).x.powi(2);
        }
        let std = (sq / n as f32).sqrt();
        assert!((0.4..2.0).contains(&std), "std {std}");
    }

    #[test]
    fn gusts_vary_in_time() {
        let w = Wind::new(WindSpec { gust_ms: 4.0, ..spec() });
        let g: Vec<f32> = (0..200).map(|k| w.gust(Vec3::ZERO, k as f32 * 0.1)).collect();
        let (lo, hi) = g.iter().fold((1.0f32, 0.0f32), |(l, h), &x| (l.min(x), h.max(x)));
        assert!(lo >= 0.0 && hi <= 1.0 && hi - lo > 0.3, "{lo}..{hi}");
    }
}
