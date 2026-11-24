//! Viewer: 3D world, flyable drone, onboard camera as an inset.
//!
//! Usage:
//!   cargo run -p viewer                                  # random seed
//!   cargo run -p viewer -- --seed 42                     # seed-generated world
//!   cargo run -p viewer -- --map maps/corridor.ron       # hand-drawn map
//!   cargo run -p viewer -- --drone drones/scout.ron      # another drone (default: Mapper)
//!   cargo run -p viewer -- --pose ekf-gps                # estimated pose (truth, ekf-gps, ekf)
//!
//! Common keys: Tab = pilot <-> free camera, [ and ] = onboard camera FOV,
//! N/P = next/previous seed (seed mode), R = reload the map (file mode).
//! Pilot: W/S forward/backward, A/D left/right, Space/Shift up/down, Q/E yaw.
//! Free camera: right click + mouse, WASD, Q/E, Shift, wheel.
//! Missions: X = exploration, G = go to B, B = place B under the cursor,
//! O = unexpected obstacle in front of the drone.
//! 3D map: M = show/hide the voxel map, V = show/hide the real world.

mod drone;
mod map3d;
mod wind;
mod explore;
mod scene;
mod mapview;
mod mission;
mod pose;
mod sensing;

use std::path::PathBuf;

use bevy::{
    camera::visibility::RenderLayers,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_egui::{EguiGlobalSettings, EguiPlugin, PrimaryEguiContext};
use drone_spec::DroneSpec;
use world::{World as SimWorld, WorldSpec};

use crate::drone::{ControlMode, DroneConfig, DronePlugin, FpvFov};

/// Where the displayed world comes from.
#[derive(Resource, Clone, Debug)]
enum Source {
    Seed(u64),
    File(PathBuf),
}

impl Source {
    fn load(&self) -> Result<SimWorld, String> {
        let spec = match self {
            Source::Seed(seed) => WorldSpec::procedural(*seed),
            Source::File(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("{} : {e}", path.display()))?;
                WorldSpec::from_ron(&text).map_err(|e| format!("{} : {e}", path.display()))?
            }
        };
        SimWorld::build(&spec).map_err(|e| e.to_string())
    }

    fn label(&self) -> String {
        match self {
            Source::Seed(seed) => format!("seed {seed}  (N/P: next/previous seed)"),
            Source::File(path) => format!("map {}  (R: reload)", path.display()),
        }
    }
}

struct Args {
    source: Source,
    drone: DroneSpec,
    pose: sim::PoseMode,
}

fn parse_args() -> Result<Args, String> {
    const USAGE: &str = "usage: viewer [--seed N | --map map.ron] [--drone drone.ron] [--pose truth|ekf-gps|ekf]";
    let mut source = None;
    let mut drone = None;
    let mut pose = sim::PoseMode::Truth;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--seed" => {
                let seed = value.parse().map_err(|_| format!("invalid seed: {value}"))?;
                source = Some(Source::Seed(seed));
            }
            "--map" => source = Some(Source::File(value.into())),
            "--drone" => {
                let text = std::fs::read_to_string(&value).map_err(|e| format!("{value}: {e}"))?;
                drone = Some(DroneSpec::from_ron(&text).map_err(|e| format!("{value}: {e}"))?);
            }
            "--pose" => pose = sim::PoseMode::parse(&value).ok_or(USAGE)?,
            _ => return Err(USAGE.into()),
        }
    }
    Ok(Args {
        source: source.unwrap_or_else(|| Source::Seed(random_seed())),
        drone: drone.unwrap_or_else(DroneSpec::mapper),
        pose,
    })
}

fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    t.as_nanos() as u64 % 1_000_000
}

/// Current world, read by the simulation.
#[derive(Resource)]
pub struct CurrentWorld(pub SimWorld);

/// The world has just been (re)built: the drone restarts from the origin.
#[derive(Message)]
pub struct WorldRebuilt;

/// Request to (re)build the world.
#[derive(Message)]
struct Rebuild;

#[derive(Component)]
struct InfoText;

/// Outside camera (free or chasing the drone).
#[derive(Component)]
pub struct OuterCamera;

/// Last world status, shown in the HUD.
#[derive(Resource, Default)]
struct WorldStatus(String);

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    // Immediate, readable failure if the starting map is invalid.
    let world = match args.source.load() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "drone-rust — viewer".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FreeCameraPlugin)
        .add_plugins(EguiPlugin::default())
        // egui context set by hand on the outside camera (not on the onboard
        // camera); egui keeps the keyboard and mouse when it needs them.
        .insert_resource(EguiGlobalSettings {
            auto_create_primary_context: false,
            enable_absorb_bevy_input_system: true,
            ..default()
        })
        .add_plugins((DronePlugin, sensing::SensingPlugin, mapview::MapPlugin, explore::ExplorePlugin, mission::MissionPlugin))
        .add_plugins(pose::PosePlugin(args.pose))
        .add_plugins((map3d::Map3dPlugin, wind::WindPlugin))
        .insert_resource(args.source)
        .insert_resource(CurrentWorld(world))
        .insert_resource(FpvFov(args.drone.camera.fov_deg))
        .insert_resource(DroneConfig(args.drone))
        .init_resource::<WorldStatus>()
        .insert_resource(ClearColor(Color::srgb(0.62, 0.75, 0.88)))
        .add_message::<Rebuild>()
        .add_message::<WorldRebuilt>()
        .add_systems(Startup, (setup, |mut w: MessageWriter<Rebuild>| {
            w.write(Rebuild);
        }))
        .add_systems(Update, (world_keys, rebuild).chain())
        .add_systems(Update, hud)
        .add_systems(Update, auto_screenshot.run_if(|| std::env::var_os("VIEWER_SCREENSHOT").is_some()))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 100.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        // Also lights the drone's layer (invisible from its own camera).
        RenderLayers::from_layers(&[0, drone::DRONE_LAYER]),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-6.0, 4.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
        FreeCamera {
            sensitivity: 0.2,
            friction: 25.0,
            walk_speed: 15.0,
            run_speed: 45.0,
            ..default()
        },
        RenderLayers::from_layers(&[0, drone::DRONE_LAYER]),
        IsDefaultUiCamera,
        PrimaryEguiContext,
        OuterCamera,
    ));
    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(15.0),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
        InfoText,
    ));
}

fn world_keys(input: Res<ButtonInput<KeyCode>>, mut source: ResMut<Source>, mut rebuild: MessageWriter<Rebuild>) {
    let changed = match source.as_mut() {
        Source::Seed(seed) if input.just_pressed(KeyCode::KeyN) => {
            *seed = seed.wrapping_add(1);
            true
        }
        Source::Seed(seed) if input.just_pressed(KeyCode::KeyP) => {
            *seed = seed.wrapping_sub(1);
            true
        }
        Source::File(_) => input.just_pressed(KeyCode::KeyR),
        _ => false,
    };
    if changed {
        rebuild.write(Rebuild);
    }
}

#[allow(clippy::too_many_arguments)]
fn rebuild(
    mut requests: MessageReader<Rebuild>,
    source: Res<Source>,
    old: Query<Entity, With<scene::WorldEntity>>,
    mut current: ResMut<CurrentWorld>,
    mut status: ResMut<WorldStatus>,
    mut rebuilt: MessageWriter<WorldRebuilt>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if requests.read().count() == 0 {
        return;
    }
    status.0 = match source.load() {
        Ok(world) => {
            for e in &old {
                commands.entity(e).despawn();
            }
            scene::spawn_world(&world, &mut commands, &mut meshes, &mut materials);
            let (lo, hi) = world.terrain.range();
            let s = format!(
                "{}\n{} obstacles, terrain {lo:.1} to {hi:.1} m",
                source.label(),
                world.obstacles.len()
            );
            current.0 = world;
            rebuilt.write(WorldRebuilt);
            s
        }
        // Reloaded map invalid: keep the old world and show the error.
        Err(e) => format!("{}\nERROR: {e}", source.label()),
    };
}

fn hud(
    status: Res<WorldStatus>,
    mode: Res<ControlMode>,
    fov: Res<FpvFov>,
    config: Res<DroneConfig>,
    world: Res<CurrentWorld>,
    drones: Query<&drone::SimDrone>,
    mut info: Query<&mut Text, With<InfoText>>,
) {
    let Ok(d) = drones.single() else { return };
    let s = &d.state;
    // Compass heading: 0 = north (-Z), clockwise.
    let heading = (360.0 - s.yaw.to_degrees()).rem_euclid(360.0);
    let controls = match *mode {
        ControlMode::Pilot => {
            "PILOT (Tab: free camera)\nW/S forward/backward, A/D left/right, Space/Shift up/down, Q/E yaw"
        }
        ControlMode::FreeCamera => {
            "FREE CAMERA (Tab: pilot)\nright click + mouse, WASD, Q/E, Shift, wheel"
        }
    };
    let (pitch, roll) = s.tilt();
    let wind = s.flight.wind;
    let thrust = 100.0 * s.flight.thrust / (config.0.airframe.thrust_to_weight * world::drone::G);
    let text = format!(
        "{}\n{}: {:.1} m/s, altitude {:.1} m, heading {:.0} deg, FOV {:.0} deg ([ and ])\n\
         pitch {:+.0} deg, roll {:+.0} deg, thrust {:.0} %, wind {:.1} m/s{}\n{controls}",
        status.0,
        config.0.name,
        s.velocity.length(),
        s.altitude(&world.0),
        heading,
        fov.0,
        pitch.to_degrees(),
        roll.to_degrees(),
        thrust,
        wind.length(),
        if s.flight.contact { "  CONTACT" } else { "" },
    );
    for mut t in &mut info {
        if t.0 != text {
            t.0.clone_from(&text);
        }
    }
}

/// Non-interactive check: `VIEWER_SCREENSHOT=out.png` captures the window, then quits.
/// `VIEWER_AUTOPILOT=1` makes the drone fly in circles meanwhile.
fn auto_screenshot(mut frame: Local<u32>, mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    use bevy::render::view::screenshot::{Screenshot, save_to_disk};
    // `VIEWER_SCREENSHOT_FRAME`: frame to capture (240 by default).
    let at: u32 = std::env::var("VIEWER_SCREENSHOT_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(240);
    *frame += 1;
    if *frame == at {
        let path = std::env::var("VIEWER_SCREENSHOT").unwrap_or_default();
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
    }
    if *frame == at + 60 {
        exit.write(AppExit::Success);
    }
}
