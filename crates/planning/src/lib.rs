//! Decision and planning: frontier exploration, A-to-B mission
//! (D* Lite, timed trajectory), tracking.
//!
//! Pure Rust. The planner only reads the map built by the drone
//! ([`mapping::OccupancyGrid`]) and the drone's pose; never the real world.

pub mod control;
pub mod dstar;
pub mod explorer;
pub mod flight3d;
pub mod frontier;
pub mod grid;
pub mod inspect;
pub mod navigator;
pub mod trajectory;

pub use explorer::{Explorer, ExplorerConfig, Status, Strategy};
pub use flight3d::{Flight3d, Flight3dConfig};
pub use navigator::{NavStatus, Navigator, NavigatorConfig};
pub use inspect::{InspectConfig, Inspector, Phase, Target};
pub use trajectory::Trajectory;

use glam::Vec2;

/// Area of interest: a disk on the ground.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zone {
    pub center: Vec2,
    pub radius: f32,
}

impl Zone {
    pub fn contains(&self, p: Vec2) -> bool {
        p.distance(self.center) <= self.radius
    }
}
