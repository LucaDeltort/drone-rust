//! Decision and planning: frontier exploration, A-to-B mission
//! (D* Lite, timed trajectory), tracking.
//!
//! Pure Rust. The planner only reads the map built by the drone
//! ([`mapping::OccupancyGrid`]) and the drone's pose; never the real world.

pub mod control;
pub mod dstar;
pub mod explorer;
pub mod frontier;
pub mod grid;
pub mod navigator;
pub mod trajectory;

pub use explorer::{Explorer, ExplorerConfig, Status, Strategy};
pub use navigator::{NavStatus, Navigator, NavigatorConfig};
pub use trajectory::Trajectory;
