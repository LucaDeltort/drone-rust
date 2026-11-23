//! Terrain as a height grid.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{BuildError, TerrainSpec};

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
        let heights = match *spec {
            TerrainSpec::Flat { height } => vec![height; n * n],
            TerrainSpec::Noise {
                seed,
                amplitude_m,
                wavelength_m,
                octaves,
            } => {
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
}
