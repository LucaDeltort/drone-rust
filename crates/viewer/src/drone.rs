//! Drone in Bevy: keyboard piloting, fixed-step simulation, cameras.

use bevy::{
    camera::{Viewport, visibility::RenderLayers},
    camera_controller::free_camera::FreeCameraState,
    prelude::*,
};
use drone_spec::DroneSpec;
use world::drone::{Command, DroneState};

use crate::{CurrentWorld, OuterCamera, WorldRebuilt};

/// Layer of the drone mesh: seen from outside, not by its own camera.
pub const DRONE_LAYER: usize = 1;

/// Simulation rate, independent of rendering.
const SIM_HZ: f64 = 100.0;

/// Starting altitude above the ground, in m.
const SPAWN_ALTITUDE: f32 = 2.0;

/// Aspect ratio of the onboard camera inset.
const FPV_ASPECT: f32 = 16.0 / 9.0;

#[derive(Resource)]
pub struct DroneConfig(pub DroneSpec);

/// Current horizontal field of view of the onboard camera, in degrees.
#[derive(Resource)]
pub struct FpvFov(pub f32);

#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    #[default]
    Pilot,
    FreeCamera,
}

/// Current pilot command, applied at each simulation step.
#[derive(Resource, Default)]
pub struct PilotCommand(pub Command);

#[derive(Component)]
pub struct SimDrone {
    pub state: DroneState,
}

#[derive(Component)]
struct FpvCamera;

/// Frame and title of the inset, aligned with the onboard camera viewport.
#[derive(Component)]
struct FpvFrame;

pub struct DronePlugin;

impl Plugin for DronePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Time::<Fixed>::from_hz(SIM_HZ))
            .init_resource::<ControlMode>()
            .init_resource::<PilotCommand>()
            .add_systems(Startup, spawn_drone)
            .add_systems(FixedUpdate, step_drone)
            .add_systems(
                Update,
                (
                    (toggle_mode, pilot_input, fov_keys),
                    reset_on_rebuild,
                    sync_transform,
                    chase_camera,
                )
                    .chain(),
            )
            .add_systems(Update, (layout_fpv, apply_fov));
    }
}

fn spawn_drone(
    mut commands: Commands,
    config: Res<DroneConfig>,
    world: Res<CurrentWorld>,
    fov: Res<FpvFov>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let spec = &config.0;
    let state = DroneState::spawn(&world.0, 0.0, 0.0, SPAWN_ALTITUDE);
    // Mesh drawn for a 0.25 m radius, scaled to the spec.
    let k = spec.airframe.radius / 0.25;
    let layer = RenderLayers::layer(DRONE_LAYER);

    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.15, 0.18),
        ..default()
    });
    let rotor_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.8, 0.8, 0.85, 0.6),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let nose_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.2, 0.1),
        emissive: LinearRgba::rgb(1.5, 0.2, 0.1),
        ..default()
    });
    let body = meshes.add(Cuboid::new(0.14, 0.05, 0.2));
    let arm = meshes.add(Cuboid::new(0.36, 0.015, 0.025));
    let rotor = meshes.add(Cylinder::new(0.075, 0.008));
    let nose = meshes.add(Cuboid::new(0.05, 0.03, 0.03));

    let pitch = spec.camera.pitch_deg.to_radians();
    let mut drone = commands.spawn((
        SimDrone { state },
        Transform::from_translation(state.position).with_rotation(state.rotation()),
        Visibility::default(),
    ));
    drone.with_children(|p| {
        let mut part = |mesh: &Handle<Mesh>, mat: &Handle<StandardMaterial>, t: Transform| {
            p.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(mat.clone()),
                t.with_scale(Vec3::splat(k)),
                layer.clone(),
            ));
        };
        part(&body, &body_mat, Transform::IDENTITY);
        // X-shaped arms, rotors at the four corners.
        for angle in [45.0_f32, -45.0] {
            part(&arm, &body_mat, Transform::from_rotation(Quat::from_rotation_y(angle.to_radians())));
        }
        for (x, z) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let pos = Vec3::new(x, 0.0, z) * 0.127 * k + Vec3::Y * 0.02 * k;
            part(&rotor, &rotor_mat, Transform::from_translation(pos));
        }
        // Front marker (-Z), to read the orientation from outside.
        part(&nose, &nose_mat, Transform::from_translation(Vec3::new(0.0, 0.0, -0.11 * k)));

        // Onboard camera: at the front, tilted by pitch_deg.
        p.spawn((
            Camera3d::default(),
            Camera {
                order: 1,
                ..default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: vertical_fov(fov.0),
                near: 0.05,
                ..default()
            }),
            Transform::from_xyz(0.0, 0.0, -spec.airframe.radius).with_rotation(Quat::from_rotation_x(pitch)),
            FpvCamera,
        ));
    });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                border: UiRect::all(px(2)),
                ..default()
            },
            BorderColor::all(Color::WHITE),
            FpvFrame,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("Onboard view"),
                TextFont::from_font_size(14.0),
                Node {
                    position_type: PositionType::Absolute,
                    top: px(-22),
                    left: px(0),
                    ..default()
                },
            ));
        });
}

/// Horizontal FOV (degrees) to vertical FOV (radians) for the inset aspect ratio.
fn vertical_fov(horizontal_deg: f32) -> f32 {
    2.0 * ((horizontal_deg.to_radians() / 2.0).tan() / FPV_ASPECT).atan()
}

fn toggle_mode(
    input: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<ControlMode>,
    mut cams: Query<(&Transform, &mut FreeCameraState), With<OuterCamera>>,
) {
    if input.just_pressed(KeyCode::Tab) {
        *mode = match *mode {
            ControlMode::Pilot => ControlMode::FreeCamera,
            ControlMode::FreeCamera => ControlMode::Pilot,
        };
    }
    for (transform, mut state) in &mut cams {
        let free = *mode == ControlMode::FreeCamera;
        if free && !state.enabled {
            // The free camera restarts from where the chase left it.
            let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
            state.yaw = yaw;
            state.pitch = pitch;
            state.velocity = Vec3::ZERO;
        }
        state.enabled = free;
    }
}

fn pilot_input(input: Res<ButtonInput<KeyCode>>, mode: Res<ControlMode>, mut cmd: ResMut<PilotCommand>) {
    if *mode != ControlMode::Pilot {
        cmd.0 = Command::default();
        return;
    }
    // Non-interactive check (see auto_screenshot): moves forward while turning, at constant altitude.
    if std::env::var_os("VIEWER_AUTOPILOT").is_some() {
        cmd.0 = Command { forward: 1.0, up: 0.0, yaw: 0.25, right: 0.0 };
        return;
    }
    let axis = |pos: KeyCode, neg: KeyCode| input.pressed(pos) as i32 as f32 - input.pressed(neg) as i32 as f32;
    cmd.0 = Command {
        forward: axis(KeyCode::KeyW, KeyCode::KeyS),
        right: axis(KeyCode::KeyD, KeyCode::KeyA),
        up: axis(KeyCode::Space, KeyCode::ShiftLeft),
        yaw: axis(KeyCode::KeyQ, KeyCode::KeyE),
    };
}

fn fov_keys(input: Res<ButtonInput<KeyCode>>, mut fov: ResMut<FpvFov>) {
    let step = 5.0;
    if input.just_pressed(KeyCode::BracketRight) {
        fov.0 = (fov.0 + step).min(150.0);
    }
    if input.just_pressed(KeyCode::BracketLeft) {
        fov.0 = (fov.0 - step).max(30.0);
    }
}

pub fn step_drone(
    time: Res<Time>,
    cmd: Res<PilotCommand>,
    config: Res<DroneConfig>,
    world: Res<CurrentWorld>,
    mut drones: Query<&mut SimDrone>,
) {
    let dt = time.delta_secs();
    for mut d in &mut drones {
        d.state.step(cmd.0, &config.0.airframe, &world.0, dt);
    }
}

fn reset_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, world: Res<CurrentWorld>, mut drones: Query<&mut SimDrone>) {
    if rebuilt.read().count() == 0 {
        return;
    }
    for mut d in &mut drones {
        d.state = DroneState::spawn(&world.0, 0.0, 0.0, SPAWN_ALTITUDE);
    }
}

fn sync_transform(mut drones: Query<(&SimDrone, &mut Transform)>) {
    for (d, mut t) in &mut drones {
        t.translation = d.state.position;
        t.rotation = d.state.rotation();
    }
}

/// In pilot mode, the outside camera follows the drone from behind.
fn chase_camera(
    time: Res<Time>,
    mode: Res<ControlMode>,
    drones: Query<&SimDrone>,
    mut cams: Query<&mut Transform, With<OuterCamera>>,
) {
    if *mode != ControlMode::Pilot {
        return;
    }
    let (Ok(d), Ok(mut cam)) = (drones.single(), cams.single_mut()) else {
        return;
    };
    let fwd = d.state.forward();
    let target = d.state.position - fwd * 4.0 + Vec3::Y * 1.5;
    // Exponential smoothing, independent of the frame rate.
    let alpha = 1.0 - (-6.0 * time.delta_secs()).exp();
    cam.translation = cam.translation.lerp(target, alpha);
    cam.look_at(d.state.position + fwd * 2.0, Vec3::Y);
}

/// Places the inset at the bottom right, a third of the window width.
fn layout_fpv(
    windows: Query<&Window>,
    mut cams: Query<&mut Camera, With<FpvCamera>>,
    mut frames: Query<&mut Node, With<FpvFrame>>,
) {
    let Ok(window) = windows.single() else { return };
    let size = window.physical_size();
    if size.x == 0 || size.y == 0 {
        return;
    }
    let margin = (16.0 * window.scale_factor()) as u32;
    let w = (size.x as f32 / 3.0) as u32;
    let h = ((w as f32 / FPV_ASPECT) as u32).min(size.y / 2);
    let pos = UVec2::new(size.x.saturating_sub(w + margin), size.y.saturating_sub(h + margin));
    let viewport = Viewport {
        physical_position: pos,
        physical_size: UVec2::new(w, h),
        ..default()
    };
    for mut cam in &mut cams {
        if cam.viewport.as_ref().map(|v| (v.physical_position, v.physical_size))
            != Some((viewport.physical_position, viewport.physical_size))
        {
            cam.viewport = Some(viewport.clone());
        }
    }
    // Frame in logical pixels, 2 px around the viewport.
    let s = window.scale_factor();
    for mut node in &mut frames {
        node.left = px(pos.x as f32 / s - 2.0);
        node.top = px(pos.y as f32 / s - 2.0);
        node.width = px(w as f32 / s + 4.0);
        node.height = px(h as f32 / s + 4.0);
    }
}

fn apply_fov(fov: Res<FpvFov>, mut cams: Query<&mut Projection, With<FpvCamera>>) {
    if !fov.is_changed() {
        return;
    }
    for mut proj in &mut cams {
        if let Projection::Perspective(p) = proj.as_mut() {
            p.fov = vertical_fov(fov.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_fov_is_narrower_for_wide_format() {
        let v = vertical_fov(90.0);
        assert!(v < 90f32.to_radians());
        // tan(v/2) * aspect = tan(h/2)
        assert!(((v / 2.0).tan() * FPV_ASPECT - 1.0).abs() < 1e-5);
    }
}
