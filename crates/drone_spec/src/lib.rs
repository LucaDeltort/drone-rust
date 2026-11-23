//! A drone described as data (`DroneSpec`, RON format).
//!
//! Switching drones needs no recompilation: the engine instantiates the spec.
//! Airframe (kinematic limits), onboard camera, range sensors and
//! navigation sensors.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroneSpec {
    pub name: String,
    pub airframe: Airframe,
    pub camera: FpvCamera,
    #[serde(default)]
    pub sensors: Vec<SensorSpec>,
}

/// A sensor mounted on the drone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSpec {
    pub kind: SensorKind,
    #[serde(default)]
    pub mount: Mount,
    /// Measurement rate, in Hz.
    pub rate_hz: f32,
}

/// Position and orientation of the sensor in the drone frame
/// (X right, Y up, -Z forward).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Mount {
    #[serde(default)]
    pub pos: [f32; 3],
    /// Rotation around Y, in degrees; positive = to the left.
    #[serde(default)]
    pub yaw_deg: f32,
    /// Rotation around X, in degrees; negative = downwards.
    #[serde(default)]
    pub pitch_deg: f32,
}

/// Common defects: standard deviation `noise_std` (m), plus `noise_growth` × d² for
/// sensors whose noise grows with distance, and a per-ray dropout probability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SensorKind {
    /// Horizontal fan of rays over 360°.
    Lidar2D {
        rays: u32,
        range: f32,
        noise_std: f32,
        #[serde(default)]
        dropout: f32,
    },
    /// A single distance: the nearest echo within a cone.
    Sonar {
        cone_deg: f32,
        range: f32,
        noise_std: f32,
        /// Beyond this incidence angle, the surface returns no echo.
        #[serde(default = "default_max_incidence")]
        max_incidence_deg: f32,
    },
    /// Depth image within a field of view (horizontal FOV).
    Depth {
        width: u32,
        height: u32,
        fov_deg: f32,
        range: f32,
        noise_std: f32,
        #[serde(default)]
        noise_growth: f32,
        #[serde(default)]
        dropout: f32,
    },
    /// Inertial unit: acceleration (m/s², drone frame) and yaw rate.
    /// Each axis has white noise and a bias drawn at start-up that slowly drifts
    /// (random walk).
    Imu {
        accel_std: f32,
        /// Standard deviation of the initial accelerometer bias, in m/s².
        accel_bias: f32,
        /// Gyro noise, in °/s.
        gyro_std_deg: f32,
        /// Standard deviation of the initial gyro bias, in °/s.
        gyro_bias_deg: f32,
    },
    /// Barometer: absolute height, noisy, with a slow drift.
    Baro {
        noise_std: f32,
        /// Drift (random walk), in m per √s.
        #[serde(default)]
        drift: f32,
    },
    /// Noisy absolute position. Can be removed to force autonomy.
    Gps { noise_std: f32 },
}

impl SensorKind {
    /// Navigation sensor (reads the drone's state, not the world through raycasts).
    pub fn is_navigation(&self) -> bool {
        matches!(self, SensorKind::Imu { .. } | SensorKind::Baro { .. } | SensorKind::Gps { .. })
    }
}

fn default_max_incidence() -> f32 {
    60.0
}

/// Kinematic limits of the airframe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Airframe {
    pub mass_kg: f32,
    /// Max speed, in m/s.
    pub max_speed: f32,
    /// Max acceleration, in m/s².
    pub max_accel: f32,
    /// Max yaw rate, in °/s.
    pub max_yaw_rate: f32,
    /// Radius of the sphere containing the drone, in m.
    pub radius: f32,
}

/// Onboard camera, fixed at the front of the drone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FpvCamera {
    /// Horizontal field of view, in degrees.
    pub fov_deg: f32,
    /// Tilt downwards (negative) or upwards, in degrees.
    pub pitch_deg: f32,
}

#[derive(Debug)]
pub enum SpecError {
    Ron(ron::error::SpannedError),
    Invalid(String),
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::Ron(e) => write!(f, "invalid RON: {e}"),
            SpecError::Invalid(m) => write!(f, "invalid drone: {m}"),
        }
    }
}

impl std::error::Error for SpecError {}

/// Default preset, embedded in the binary.
pub const SCOUT_RON: &str = include_str!("../../../drones/scout.ron");

impl DroneSpec {
    pub fn from_ron(text: &str) -> Result<Self, SpecError> {
        let spec: Self = ron::from_str(text).map_err(SpecError::Ron)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn scout() -> Self {
        Self::from_ron(SCOUT_RON).expect("drones/scout.ron is valid")
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        let a = &self.airframe;
        let positive = [
            ("mass_kg", a.mass_kg),
            ("max_speed", a.max_speed),
            ("max_accel", a.max_accel),
            ("max_yaw_rate", a.max_yaw_rate),
            ("radius", a.radius),
        ];
        if let Some((name, _)) = positive.iter().find(|(_, v)| !v.is_finite() || *v <= 0.0) {
            return Err(SpecError::Invalid(format!("{name} must be > 0")));
        }
        let fov = self.camera.fov_deg;
        if !(1.0..=170.0).contains(&fov) {
            return Err(SpecError::Invalid(format!("fov_deg {fov} out of 1..=170")));
        }
        for (i, s) in self.sensors.iter().enumerate() {
            s.validate().map_err(|m| SpecError::Invalid(format!("sensor {i}: {m}")))?;
        }
        Ok(())
    }
}

impl SensorSpec {
    fn validate(&self) -> Result<(), String> {
        let pos = |name: &str, v: f32| {
            if v.is_finite() && v > 0.0 { Ok(()) } else { Err(format!("{name} must be > 0")) }
        };
        let unit = |name: &str, v: f32| {
            if (0.0..=1.0).contains(&v) { Ok(()) } else { Err(format!("{name} out of 0..=1")) }
        };
        let non_neg = |name: &str, v: f32| {
            if v.is_finite() && v >= 0.0 { Ok(()) } else { Err(format!("{name} must be >= 0")) }
        };
        pos("rate_hz", self.rate_hz)?;
        match self.kind {
            SensorKind::Lidar2D { rays, range, noise_std, dropout } => {
                if rays == 0 || rays > 10_000 {
                    return Err("rays out of 1..=10000".into());
                }
                pos("range", range)?;
                non_neg("noise_std", noise_std)?;
                unit("dropout", dropout)
            }
            SensorKind::Sonar { cone_deg, range, noise_std, max_incidence_deg } => {
                if !(0.0 < cone_deg && cone_deg < 180.0) {
                    return Err("cone_deg out of ]0, 180[".into());
                }
                if !(0.0..=90.0).contains(&max_incidence_deg) {
                    return Err("max_incidence_deg out of 0..=90".into());
                }
                pos("range", range)?;
                non_neg("noise_std", noise_std)
            }
            SensorKind::Depth { width, height, fov_deg, range, noise_std, noise_growth, dropout } => {
                if !(1..=512).contains(&width) || !(1..=512).contains(&height) {
                    return Err("width and height out of 1..=512".into());
                }
                if !(1.0..=170.0).contains(&fov_deg) {
                    return Err("fov_deg out of 1..=170".into());
                }
                pos("range", range)?;
                non_neg("noise_std", noise_std)?;
                non_neg("noise_growth", noise_growth)?;
                unit("dropout", dropout)
            }
            SensorKind::Imu { accel_std, accel_bias, gyro_std_deg, gyro_bias_deg } => {
                non_neg("accel_std", accel_std)?;
                non_neg("accel_bias", accel_bias)?;
                non_neg("gyro_std_deg", gyro_std_deg)?;
                non_neg("gyro_bias_deg", gyro_bias_deg)
            }
            SensorKind::Baro { noise_std, drift } => {
                non_neg("noise_std", noise_std)?;
                non_neg("drift", drift)
            }
            SensorKind::Gps { noise_std } => non_neg("noise_std", noise_std),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scout_preset_loads() {
        let s = DroneSpec::scout();
        assert_eq!(s.name, "Scout");
        assert!(s.airframe.max_speed > 0.0);
    }

    #[test]
    fn round_trip() {
        let s = DroneSpec::scout();
        let text = ron::ser::to_string_pretty(&s, Default::default()).unwrap();
        assert_eq!(DroneSpec::from_ron(&text).unwrap(), s);
    }

    #[test]
    fn rejects_bad_values() {
        let bad = SCOUT_RON.replace("max_speed: 8.0", "max_speed: 0.0");
        assert!(matches!(DroneSpec::from_ron(&bad), Err(SpecError::Invalid(_))));
        let bad = SCOUT_RON.replace("fov_deg: 90.0", "fov_deg: 200.0");
        assert!(matches!(DroneSpec::from_ron(&bad), Err(SpecError::Invalid(_))));
        let bad = SCOUT_RON.replace("rays: 360", "rays: 0");
        assert!(matches!(DroneSpec::from_ron(&bad), Err(SpecError::Invalid(_))));
    }

    #[test]
    fn all_presets_load() {
        for path in ["scout.ron", "heavy.ron"] {
            let text = std::fs::read_to_string(format!("{}/../../drones/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let spec = DroneSpec::from_ron(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(!spec.sensors.is_empty(), "{path} has no sensor");
        }
    }

    #[test]
    fn sensors_default_to_empty() {
        let text = "DroneSpec(name: \"Nu\", airframe: (mass_kg: 1.0, max_speed: 1.0, max_accel: 1.0, \
                    max_yaw_rate: 1.0, radius: 0.2), camera: (fov_deg: 90.0, pitch_deg: 0.0))";
        assert!(DroneSpec::from_ron(text).unwrap().sensors.is_empty());
    }
}
