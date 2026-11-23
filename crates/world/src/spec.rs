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

/// An obstacle of a hand-drawn map. Its base height comes from the terrain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObstacleSpec {
    pub shape: ObstacleShape,
    pub x: f32,
    pub z: f32,
    #[serde(default)]
    pub yaw_deg: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ObstacleShape {
    /// Box, `size` = full dimensions (x, height, z) in meters.
    Box { size: Vec3 },
    /// Vertical cylinder (pole, tree).
    Cylinder { radius: f32, height: f32 },
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
        if let TerrainSpec::Noise {
            wavelength_m,
            octaves,
            ..
        } = self.terrain
            && (wavelength_m <= 0.0 || octaves == 0)
        {
            return bad("wavelength_m must be > 0 and octaves >= 1");
        }
        if let ObstaclesSpec::List(list) = &self.obstacles {
            let half = self.size_m / 2.0;
            if list.iter().any(|o| o.x.abs() > half || o.z.abs() > half) {
                return bad("obstacle off the map");
            }
        }
        if let ObstaclesSpec::City(city) = &self.obstacles {
            city.validate().map_err(BuildError::Invalid)?;
        }
        Ok(())
    }
}
