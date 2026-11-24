//! Terrain as a height grid.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{BuildError, TerrainFeature, TerrainSpec};

/// Square height grid, centered on the origin.
/// `heights[iz * n + ix]` = height at vertex (ix, iz).
#[derive(Debug, Clone)]
pub struct Heightmap {
    /// Number of vertices per side.
    n: usize,
    cell_m: f32,
    heights: Vec<f32>,
}

impl Heightmap {
    pub(crate) fn build(size_m: f32, cell_m: f32, spec: &TerrainSpec) -> Result<Self, BuildError> {
        let cells = (size_m / cell_m).round() as usize;
        if cells == 0 || cells > 4096 {
            return Err(BuildError::Invalid(format!(
                "{cells} cells per side: out of 1..=4096"
            )));
        }
        let n = cells + 1;
        let half = cells as f32 * cell_m / 2.0;
        let fbm_heights = |seed: u32, amplitude_m: f32, wavelength_m: f32, octaves: usize| -> Vec<f32> {
            let fbm = Fbm::<Perlin>::new(seed)
                .set_octaves(octaves)
                .set_frequency(1.0 / wavelength_m as f64);
            (0..n * n)
                .map(|i| {
                    let x = (i % n) as f64 * cell_m as f64 - half as f64;
                    let z = (i / n) as f64 * cell_m as f64 - half as f64;
                    // fBm stays roughly within [-1, 1].
                    fbm.get([x, z]) as f32 * amplitude_m
                })
                .collect()
        };
        let heights = match spec {
            TerrainSpec::Flat { height } => vec![*height; n * n],
            &TerrainSpec::Noise { seed, amplitude_m, wavelength_m, octaves } => {
                fbm_heights(seed, amplitude_m, wavelength_m, octaves)
            }
            TerrainSpec::Landscape { seed, amplitude_m, wavelength_m, octaves, features } => {
                let mut h = fbm_heights(*seed, *amplitude_m, *wavelength_m, *octaves);
                for f in features {
                    for (i, v) in h.iter_mut().enumerate() {
                        let x = (i % n) as f32 * cell_m - half;
                        let z = (i / n) as f32 * cell_m - half;
                        *v = sculpt(*f, x, z, *v);
                    }
                }
                h
            }
        };
        Ok(Self { n, cell_m, heights })
    }

    pub fn vertices_per_side(&self) -> usize {
        self.n
    }

    pub fn cell_m(&self) -> f32 {
        self.cell_m
    }

    pub fn half_size(&self) -> f32 {
        (self.n - 1) as f32 * self.cell_m / 2.0
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Position (x, z) of vertex (ix, iz).
    pub fn vertex_xz(&self, ix: usize, iz: usize) -> (f32, f32) {
        let h = self.half_size();
        (ix as f32 * self.cell_m - h, iz as f32 * self.cell_m - h)
    }

    /// Bilinearly interpolated height at (x, z); clamped at the map edge.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let max = (self.n - 1) as f32;
        let fx = ((x + self.half_size()) / self.cell_m).clamp(0.0, max);
        let fz = ((z + self.half_size()) / self.cell_m).clamp(0.0, max);
        let (ix, iz) = ((fx as usize).min(self.n - 2), (fz as usize).min(self.n - 2));
        let (tx, tz) = (fx - ix as f32, fz - iz as f32);
        let h = |i: usize, j: usize| self.heights[j * self.n + i];
        let a = h(ix, iz) * (1.0 - tx) + h(ix + 1, iz) * tx;
        let b = h(ix, iz + 1) * (1.0 - tx) + h(ix + 1, iz + 1) * tx;
        a * (1.0 - tz) + b * tz
    }

    /// Min and max terrain heights.
    pub fn range(&self) -> (f32, f32) {
        self.heights
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &h| (lo.min(h), hi.max(h)))
    }
}

/// Applies a relief feature to the height `h` at (x, z).
fn sculpt(f: TerrainFeature, x: f32, z: f32, h: f32) -> f32 {
    match f {
        TerrainFeature::Hill { x: cx, z: cz, radius_m, height_m } => {
            let d = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt() / radius_m;
            if d >= 1.0 {
                h
            } else {
                h + height_m * (d * std::f32::consts::FRAC_PI_2).cos().powi(2)
            }
        }
        TerrainFeature::Pad { x: cx, z: cz, size, height_m, blend_m } => {
            // Distance outside the rectangle (0 inside).
            let dx = ((x - cx).abs() - size.0 / 2.0).max(0.0);
            let dz = ((z - cz).abs() - size.1 / 2.0).max(0.0);
            let d = (dx * dx + dz * dz).sqrt();
            if d <= 0.0 {
                return height_m;
            }
            if d >= blend_m {
                return h;
            }
            // Smoothstep from the pad to the natural ground.
            let k = d / blend_m;
            let k = k * k * (3.0 - 2.0 * k);
            height_m + (h - height_m) * k
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise() -> TerrainSpec {
        TerrainSpec::Noise {
            seed: 5,
            amplitude_m: 6.0,
            wavelength_m: 60.0,
            octaves: 4,
        }
    }

    #[test]
    fn height_at_matches_vertices() {
        let t = Heightmap::build(20.0, 1.0, &noise()).unwrap();
        for (ix, iz) in [(0, 0), (3, 7), (20, 20), (20, 0)] {
            let (x, z) = t.vertex_xz(ix, iz);
            let expected = t.heights()[iz * t.vertices_per_side() + ix];
            assert!((t.height_at(x, z) - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn height_at_clamps_outside_map() {
        let t = Heightmap::build(20.0, 1.0, &noise()).unwrap();
        assert_eq!(t.height_at(1e6, 1e6), t.height_at(10.0, 10.0));
    }

    #[test]
    fn noise_stays_within_amplitude() {
        let t = Heightmap::build(200.0, 1.0, &noise()).unwrap();
        let (lo, hi) = t.range();
        assert!(lo >= -6.0 * 1.5 && hi <= 6.0 * 1.5, "{lo}..{hi}");
        assert!(hi - lo > 1.0, "terrain too flat: {lo}..{hi}");
    }

    #[test]
    fn landscape_features_shape_the_ground() {
        let spec = TerrainSpec::Landscape {
            seed: 2,
            amplitude_m: 2.0,
            wavelength_m: 40.0,
            octaves: 3,
            features: vec![
                TerrainFeature::Hill { x: -20.0, z: 0.0, radius_m: 15.0, height_m: 10.0 },
                TerrainFeature::Pad { x: 20.0, z: 0.0, size: (10.0, 6.0), height_m: 1.5, blend_m: 4.0 },
            ],
        };
        let t = Heightmap::build(100.0, 1.0, &spec).unwrap();
        assert!(t.height_at(-20.0, 0.0) > 7.0);
        for (x, z) in [(20.0, 0.0), (24.0, 2.0), (16.0, -3.0)] {
            assert!((t.height_at(x, z) - 1.5).abs() < 1e-4);
        }
    }
}
