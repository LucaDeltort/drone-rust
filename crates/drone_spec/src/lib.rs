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
    /// Delay between the capture and the delivery of a measurement, in s. The
    /// measurement keeps its capture time: the consumer gets it late.
    #[serde(default)]
    pub latency_s: f32,
}

/// Position and orientation of the sensor in the drone frame
/// (X right, Y up, -Z forward).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mount {
    #[serde(default)]
    pub pos: [f32; 3],
    /// Rotation around Y, in degrees; positive = to the left.
    #[serde(default)]
    pub yaw_deg: f32,
    /// Rotation around X, in degrees; negative = downwards.
    #[serde(default)]
    pub pitch_deg: f32,
    /// `true`: on a gimbal that cancels roll and pitch, the sensor only follows the
    /// heading. `false`: fixed to the airframe, it tilts with it.
    #[serde(default = "yes")]
    pub stabilized: bool,
}

impl Default for Mount {
    fn default() -> Self {
        Self { pos: [0.0; 3], yaw_deg: 0.0, pitch_deg: 0.0, stabilized: true }
    }
}

fn yes() -> bool {
    true
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
    /// Spinning multi-channel lidar: `channels` rings between `fov_down_deg` and
    /// `fov_up_deg` (elevation), `rays` per ring over 360°. Measures a 3D point
    /// cloud, delivered as a depth image (`rays` × `channels`).
    Lidar3D {
        channels: u32,
        rays: u32,
        fov_up_deg: f32,
        fov_down_deg: f32,
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
        /// Multipath: chance that an echo off an oblique surface bounces off a second
        /// one before coming back, and reads too far (0: never; scaled by the
        /// obliquity).
        #[serde(default)]
        multipath: f32,
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

/// Limits and flight dynamics of the airframe.
///
/// The drone is a quadrotor: it accelerates by tilting its thrust, and feels drag
/// relative to the air (so the wind). The fields after `radius` have defaults
/// typical of a small quadrotor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Airframe {
    pub mass_kg: f32,
    /// Max speed commanded by the flight controller, in m/s.
    pub max_speed: f32,
    /// Max acceleration commanded by the flight controller, in m/s².
    pub max_accel: f32,
    /// Max yaw rate, in °/s.
    pub max_yaw_rate: f32,
    /// Radius of the sphere containing the drone, in m.
    pub radius: f32,
    /// Max thrust over weight.
    #[serde(default = "defaults::thrust_to_weight")]
    pub thrust_to_weight: f32,
    /// Max tilt of the thrust axis, in degrees.
    #[serde(default = "defaults::max_tilt_deg")]
    pub max_tilt_deg: f32,
    /// Max tilt rate, in °/s.
    #[serde(default = "defaults::tilt_rate_deg")]
    pub tilt_rate_deg: f32,
    /// Time constant of the attitude response, in s.
    #[serde(default = "defaults::attitude_tau_s")]
    pub attitude_tau_s: f32,
    /// Time constant of the motors (thrust response), in s.
    #[serde(default = "defaults::motor_tau_s")]
    pub motor_tau_s: f32,
    /// Linear (rotor) drag, in 1/s: deceleration per m/s of airspeed.
    #[serde(default = "defaults::drag")]
    pub drag: f32,
    /// Quadratic (body) drag, in 1/m: deceleration per (m/s)² of airspeed.
    #[serde(default = "defaults::drag_quad")]
    pub drag_quad: f32,
    /// Rotor (propeller) radius, in m: sets the ground effect and the power drawn.
    /// Defaults to `0.4 * radius`.
    #[serde(default)]
    pub rotor_radius_m: Option<f32>,
    /// Battery; none = endless energy at full voltage.
    #[serde(default)]
    pub battery: Option<Battery>,
}

impl Airframe {
    /// Rotor radius, in m (`rotor_radius_m`, or `0.4 * radius`).
    pub fn rotor_radius(&self) -> f32 {
        self.rotor_radius_m.unwrap_or(0.4 * self.radius)
    }
}

/// Battery pack. The voltage falls linearly with the charge from `v_full` to
/// `v_empty`, and sags under load through `r_internal`; the motors' max thrust
/// scales with (V / `v_full`)².
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Battery {
    /// Energy when full, in Wh.
    pub capacity_wh: f32,
    /// Open-circuit voltage when full, in V.
    pub v_full: f32,
    /// Open-circuit voltage when empty, in V.
    pub v_empty: f32,
    /// Internal resistance, in Ω.
    pub r_internal: f32,
}

mod defaults {
    pub fn thrust_to_weight() -> f32 {
        2.2
    }
    pub fn max_tilt_deg() -> f32 {
        40.0
    }
    pub fn tilt_rate_deg() -> f32 {
        400.0
    }
    pub fn attitude_tau_s() -> f32 {
        0.06
    }
    pub fn motor_tau_s() -> f32 {
        0.03
    }
    pub fn drag() -> f32 {
        0.25
    }
    pub fn drag_quad() -> f32 {
        0.015
    }
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
/// 3D mapping preset (3D lidar), embedded in the binary.
pub const MAPPER_RON: &str = include_str!("../../../drones/mapper.ron");

impl DroneSpec {
    pub fn from_ron(text: &str) -> Result<Self, SpecError> {
        let spec: Self = ron::from_str(text).map_err(SpecError::Ron)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn scout() -> Self {
        Self::from_ron(SCOUT_RON).expect("drones/scout.ron is valid")
    }

    pub fn mapper() -> Self {
        Self::from_ron(MAPPER_RON).expect("drones/mapper.ron is valid")
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        let a = &self.airframe;
        let positive = [
            ("mass_kg", a.mass_kg),
            ("max_speed", a.max_speed),
            ("max_accel", a.max_accel),
            ("max_yaw_rate", a.max_yaw_rate),
            ("radius", a.radius),
            ("tilt_rate_deg", a.tilt_rate_deg),
            ("attitude_tau_s", a.attitude_tau_s),
            ("motor_tau_s", a.motor_tau_s),
        ];
        if let Some((name, _)) = positive.iter().find(|(_, v)| !v.is_finite() || *v <= 0.0) {
            return Err(SpecError::Invalid(format!("{name} must be > 0")));
        }
        if a.thrust_to_weight.is_nan() || a.thrust_to_weight <= 1.05 {
            return Err(SpecError::Invalid("thrust_to_weight must be > 1.05 to hover".into()));
        }
        if !(5.0..=80.0).contains(&a.max_tilt_deg) {
            return Err(SpecError::Invalid("max_tilt_deg out of 5..=80".into()));
        }
        if !(a.drag >= 0.0 && a.drag_quad >= 0.0) {
            return Err(SpecError::Invalid("drag must be >= 0".into()));
        }
        let r = a.rotor_radius();
        if !(r.is_finite() && r > 0.0 && r < a.radius) {
            return Err(SpecError::Invalid("rotor_radius_m must be in ]0, radius[".into()));
        }
        if let Some(b) = a.battery {
            let ok = b.capacity_wh.is_finite()
                && b.capacity_wh > 0.0
                && b.v_empty > 0.0
                && b.v_full > b.v_empty
                && b.v_full.is_finite()
                && b.r_internal >= 0.0;
            if !ok {
                return Err(SpecError::Invalid(
                    "battery: need capacity_wh > 0, 0 < v_empty < v_full, r_internal >= 0".into(),
                ));
            }
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
            SensorKind::Lidar3D { channels, rays, fov_up_deg, fov_down_deg, range, noise_std, dropout } => {
                if !(1..=128).contains(&channels) || !(1..=4096).contains(&rays) {
                    return Err("channels out of 1..=128 or rays out of 1..=4096".into());
                }
                if !(-90.0 <= fov_down_deg && fov_down_deg < fov_up_deg && fov_up_deg <= 90.0) {
                    return Err("need -90 <= fov_down_deg < fov_up_deg <= 90".into());
                }
                pos("range", range)?;
                non_neg("noise_std", noise_std)?;
                unit("dropout", dropout)
            }
            SensorKind::Sonar { cone_deg, range, noise_std, max_incidence_deg, multipath } => {
                if !(0.0..=1.0).contains(&multipath) {
                    return Err("multipath out of 0..=1".into());
                }
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
        for path in ["scout.ron", "heavy.ron", "pro.ron", "cheap.ron", "mapper.ron"] {
            let text = std::fs::read_to_string(format!("{}/../../drones/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            let spec = DroneSpec::from_ron(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(!spec.sensors.is_empty(), "{path} has no sensor");
        }
    }

    #[test]
    fn battery_and_rotor_radius_parse_and_default() {
        let s = DroneSpec::scout();
        assert_eq!(s.airframe.battery, None);
        assert!((s.airframe.rotor_radius() - 0.4 * s.airframe.radius).abs() < 1e-6);
        // The example commented out in mapper.ron.
        let text = MAPPER_RON
            .replace("// battery: Some", "battery: Some")
            .replace("// rotor_radius_m", "rotor_radius_m");
        let a = DroneSpec::from_ron(&text).unwrap().airframe;
        assert_eq!(a.battery.unwrap().capacity_wh, 32.0);
        assert_eq!(a.rotor_radius(), 0.12);
        let bad = text.replace("v_empty: 13.2", "v_empty: 20.0");
        assert!(matches!(DroneSpec::from_ron(&bad), Err(SpecError::Invalid(_))));
    }

    #[test]
    fn sensors_default_to_empty() {
        let text = "DroneSpec(name: \"Nu\", airframe: (mass_kg: 1.0, max_speed: 1.0, max_accel: 1.0, \
                    max_yaw_rate: 1.0, radius: 0.2), camera: (fov_deg: 90.0, pitch_deg: 0.0))";
        assert!(DroneSpec::from_ron(text).unwrap().sensors.is_empty());
    }
}
