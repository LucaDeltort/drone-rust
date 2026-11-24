//! Description of a world, serializable to RON.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::BuildError;

/// Everything needed to rebuild a world identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSpec {
    /// Side of the square map, in meters.
    pub size_m: f32,
    /// Step of the height grid, in meters.
    pub cell_m: f32,
    pub terrain: TerrainSpec,
    pub obstacles: ObstaclesSpec,
    /// Wind; calm when omitted.
    #[serde(default)]
    pub wind: crate::wind::WindSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TerrainSpec {
    /// Flat ground at `height` meters.
    Flat { height: f32 },
    /// Fractal Perlin noise terrain (fBm).
    Noise {
        seed: u32,
        /// Max deviation around 0, in meters.
        amplitude_m: f32,
        /// Typical size of a hill, in meters.
        wavelength_m: f32,
        octaves: usize,
    },
    /// Noise terrain sculpted by features: hills, hollows, flat building pads.
    Landscape {
        seed: u32,
        amplitude_m: f32,
        wavelength_m: f32,
        octaves: usize,
        #[serde(default)]
        features: Vec<TerrainFeature>,
    },
}

/// Hand-placed relief, applied in order on top of the noise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TerrainFeature {
    /// Smooth bump (cos² profile); a negative height digs a hollow.
    Hill { x: f32, z: f32, radius_m: f32, height_m: f32 },
    /// Flat platform at absolute height `height_m` over the rectangle `size` (x, z)
    /// centered on (x, z), blending into the terrain over `blend_m`.
    Pad { x: f32, z: f32, size: (f32, f32), height_m: f32, blend_m: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObstaclesSpec {
    None,
    /// Hand-placed obstacles.
    List(Vec<ObstacleSpec>),
    /// Random obstacles, outside a cleared disk around the origin.
    Random {
        seed: u64,
        count: usize,
        clear_radius_m: f32,
    },
    /// Procedural city: streets, blocks, buildings, parks.
    City(crate::city::CitySpec),
}

/// What a surface is made of: how range sensors see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Material {
    /// Matte and opaque: clean returns (walls, ground).
    #[default]
    Concrete,
    /// Windows: most lidar and depth rays go through, some reflect off it.
    Glass,
    /// Dark paint, tinted surfaces: weak returns, only at short range.
    Dark,
    /// Shiny metal (cars, roofs, poles): noisier, sometimes a phantom echo behind it
    /// (the ray bounced away before coming back).
    Metal,
    /// Leaves: partial returns from inside the canopy, short of its outline.
    Foliage,
}

/// An obstacle of a hand-drawn map. Its base height comes from the terrain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObstacleSpec {
    pub shape: ObstacleShape,
    pub x: f32,
    pub z: f32,
    #[serde(default)]
    pub yaw_deg: f32,
    /// Height of the base above the ground, in m: lintels, slabs, roofs, bridges.
    #[serde(default)]
    pub elevation_m: f32,
    #[serde(default)]
    pub material: Material,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ObstacleShape {
    /// Box, `size` = full dimensions (x, height, z) in meters.
    Box { size: Vec3 },
    /// Vertical cylinder (pole, tree).
    Cylinder { radius: f32, height: f32 },
    /// Gable roof: a triangular prism, `size` = (length, height, width); the ridge
    /// runs along the local X axis, at mid-width.
    Roof { size: Vec3 },
}

impl WorldSpec {
    /// Seed-generated world: gentle 200 m × 200 m terrain and 60 obstacles.
    pub fn procedural(seed: u64) -> Self {
        Self {
            size_m: 200.0,
            cell_m: 1.0,
            terrain: TerrainSpec::Noise {
                // The noise takes a 32-bit seed; mix both halves.
                seed: (seed ^ (seed >> 32)) as u32,
                amplitude_m: 6.0,
                wavelength_m: 60.0,
                octaves: 4,
            },
            obstacles: ObstaclesSpec::Random {
                seed,
                count: 60,
                clear_radius_m: 10.0,
            },
            wind: Default::default(),
        }
    }

    pub fn from_ron(text: &str) -> Result<Self, BuildError> {
        ron::from_str(text).map_err(BuildError::Ron)
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .expect("WorldSpec is always serializable")
    }

    pub(crate) fn validate(&self) -> Result<(), BuildError> {
        let bad = |m: &str| Err(BuildError::Invalid(m.into()));
        if !(self.size_m > 0.0 && self.cell_m > 0.0) {
            return bad("size_m and cell_m must be > 0");
        }
        if self.cell_m > self.size_m {
            return bad("cell_m must be <= size_m");
        }
        if let TerrainSpec::Noise { wavelength_m, octaves, .. } | TerrainSpec::Landscape { wavelength_m, octaves, .. } =
            self.terrain
            && (wavelength_m <= 0.0 || octaves == 0)
        {
            return bad("wavelength_m must be > 0 and octaves >= 1");
        }
        if let TerrainSpec::Landscape { features, .. } = &self.terrain {
            for f in features {
                let ok = match *f {
                    TerrainFeature::Hill { radius_m, .. } => radius_m > 0.0,
                    TerrainFeature::Pad { size, blend_m, .. } => size.0 > 0.0 && size.1 > 0.0 && blend_m >= 0.0,
                };
                if !ok {
                    return bad("terrain feature with a non-positive size");
                }
            }
        }
        self.wind.validate().map_err(BuildError::Invalid)?;
        if let ObstaclesSpec::List(list) = &self.obstacles {
            let half = self.size_m / 2.0;
            if list.iter().any(|o| o.x.abs() > half || o.z.abs() > half) {
                return bad("obstacle off the map");
            }
            if list.iter().any(|o| o.elevation_m.is_nan() || o.elevation_m < 0.0) {
                return bad("elevation_m must be >= 0");
            }
        }
        if let ObstaclesSpec::City(city) = &self.obstacles {
            city.validate().map_err(BuildError::Invalid)?;
        }
        Ok(())
    }
}
