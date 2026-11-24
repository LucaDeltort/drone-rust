//! Sensors in the viewer: fixed-step measurements, rays as gizmos, egui settings.

use bevy::{camera::visibility::RenderLayers, prelude::*};
use bevy_egui::egui;
use sensors::{Data, Measurement, Pose, SensorSuite};

use crate::{
    CurrentWorld, WorldRebuilt,
    drone::{DRONE_LAYER, DroneConfig, FpvFov, SimDrone, step_drone},
};

/// Sensor noise seed. Fixed: two identical runs give the same measurements.
pub const SENSOR_SEED: u64 = 0x5EED;

#[derive(Component)]
pub struct Suite(pub SensorSuite);

#[derive(Resource)]
pub struct DebugView {
    rays: bool,
    misses: bool,
}

pub struct SensingPlugin;

impl Plugin for SensingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DebugView { rays: true, misses: false })
            .add_systems(Startup, gizmos_outer_view_only)
            .add_systems(PostStartup, attach_suite)
            .add_systems(FixedUpdate, sense.after(step_drone))
            .add_systems(Update, (clear_on_rebuild, draw_rays).chain())
;
    }
}

/// Rays are only shown in the outside view, not in the onboard camera.
fn gizmos_outer_view_only(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<DefaultGizmoConfigGroup>();
    config.render_layers = RenderLayers::layer(DRONE_LAYER);
    config.line.width = 1.5;
}

fn attach_suite(mut commands: Commands, config: Res<DroneConfig>, drones: Query<Entity, With<SimDrone>>) {
    for e in &drones {
        commands.entity(e).insert(Suite(SensorSuite::new(&config.0.sensors, SENSOR_SEED)));
    }
}

pub fn sense(time: Res<Time>, world: Res<CurrentWorld>, mut drones: Query<(&SimDrone, &mut Suite)>) {
    let t = time.elapsed_secs_f64();
    for (d, mut suite) in &mut drones {
        let body = Pose::from(&d.state);
        suite.0.update(&world.0, &body, t);
    }
}

/// After a world change, the old measurements mean nothing anymore.
fn clear_on_rebuild(mut rebuilt: MessageReader<WorldRebuilt>, mut suites: Query<&mut Suite>) {
    if rebuilt.read().count() == 0 {
        return;
    }
    for mut s in &mut suites {
        s.0.reset();
    }
}

/// Beyond this many beams, a measurement is drawn as thinned-out points.
const DENSE: usize = 1500;

/// Hit color: green near, yellow at mid-range, red at the limit.
fn range_color(d: f32, max: f32) -> Color {
    let k = (d / max).clamp(0.0, 1.0);
    Color::hsl(120.0 * (1.0 - k), 0.9, 0.5)
}

fn draw_rays(view: Res<DebugView>, suites: Query<&Suite>, mut gizmos: Gizmos) {
    if !view.rays {
        return;
    }
    let miss = Color::srgba(0.6, 0.6, 0.6, 0.25);
    for suite in &suites {
        for m in suite.0.latest.iter().flatten() {
            let Measurement { origin, rotation, data, .. } = m;
            match data {
                Data::Scan { beams, max_range } | Data::DepthImage { beams, max_range, .. } if beams.len() > DENSE => {
                    // Point cloud (3D lidar): hit points only, thinned out, or the
                    // rays hide the scene.
                    let stride = beams.len().div_ceil(DENSE);
                    for b in beams.iter().step_by(stride) {
                        if let Some(d) = b.range {
                            let p = *origin + *rotation * b.dir * d;
                            gizmos.line(p - Vec3::Y * 0.06, p + Vec3::Y * 0.06, range_color(d, *max_range));
                        }
                    }
                }
                Data::Scan { beams, max_range } | Data::DepthImage { beams, max_range, .. } => {
                    for b in beams {
                        let dir = *rotation * b.dir;
                        match b.range {
                            Some(d) => {
                                let p = *origin + dir * d;
                                gizmos.line(*origin, p, range_color(d, *max_range).with_alpha(0.5));
                                // Small vertical tick at the hit point.
                                gizmos.line(p - Vec3::Y * 0.05, p + Vec3::Y * 0.05, range_color(d, *max_range));
                            }
                            None if view.misses => gizmos.line(*origin, *origin + dir * *max_range, miss),
                            None => {}
                        }
                    }
                }
                Data::Echo { range, cone_half_angle, max_range } => {
                    let d = range.unwrap_or(*max_range);
                    // Orange: echo received; grey: nothing in the cone.
                    let color = if range.is_some() { Color::srgb(1.0, 0.55, 0.0) } else { miss };
                    let axis = *rotation * Vec3::NEG_Z;
                    let radius = d * cone_half_angle.tan();
                    let center = *origin + axis * d;
                    // Cone disk at the measured distance (circle in the plane normal to the axis).
                    let disk = Isometry3d::new(center, *rotation);
                    gizmos.circle(disk, radius, color);
                    for k in 0..4 {
                        let a = std::f32::consts::FRAC_PI_2 * k as f32;
                        let edge = *rotation * Vec3::new(a.cos() * radius, a.sin() * radius, -d);
                        gizmos.line(*origin, *origin + edge, color);
                    }
                }
                // Navigation sensors: nothing to draw.
                Data::Imu { .. } | Data::Baro { .. } | Data::Gps { .. } => {}
            }
        }
    }
}

/// Sensors section of the side panel.
pub fn panel(ui: &mut egui::Ui, view: &mut DebugView, fov: &mut FpvFov, suites: &mut Query<&mut Suite>) {

            ui.checkbox(&mut view.rays, "Show rays");
            ui.add_enabled(view.rays, egui::Checkbox::new(&mut view.misses, "Show rays without return"));
            let mut f = fov.0;
            if ui.add(egui::Slider::new(&mut f, 30.0..=150.0).text("Onboard camera FOV (°)")).changed() {
                fov.0 = f;
            }
            for mut suite in suites.iter_mut() {
                let SensorSuite { sensors, latest, .. } = &mut suite.0;
                if sensors.is_empty() {
                    ui.label("This drone has no sensor.");
                }
                for (i, (sensor, m)) in sensors.iter_mut().zip(latest.iter()).enumerate() {
                    ui.separator();
                    ui.strong(sensor.label());
                    ui.label(format!("{} Hz — {}", sensor.rate_hz(), summary(m.as_ref())));
                    let n = sensor.noise_mut();
                    ui.push_id(i, |ui| {
                        ui.add(egui::Slider::new(&mut n.std_m, 0.0..=0.5).text("noise σ (m)"));
                        ui.add(
                            egui::Slider::new(&mut n.growth_per_m2, 0.0..=0.05)
                                .text("σ growth (/m²)"),
                        );
                        ui.add(egui::Slider::new(&mut n.dropout, 0.0..=1.0).text("dropouts"));
                    });
                }
            }
}

/// Readable summary of the last measurement.
fn summary(m: Option<&Measurement>) -> String {
    let Some(m) = m else { return "no measurement yet".into() };
    match &m.data {
        Data::Scan { beams, .. } | Data::DepthImage { beams, .. } => {
            let hits: Vec<f32> = beams.iter().filter_map(|b| b.range).collect();
            match hits.iter().copied().min_by(f32::total_cmp) {
                Some(min) => format!("{}/{} returns, nearest {min:.2} m", hits.len(), beams.len()),
                None => format!("0/{} retour", beams.len()),
            }
        }
        Data::Echo { range: Some(r), .. } => format!("echo at {r:.2} m"),
        Data::Echo { range: None, .. } => "no echo".into(),
        Data::Imu { accel, yaw_rate } => format!(
            "acc ({:.2}, {:.2}, {:.2}) m/s², yaw {:.1} °/s",
            accel.x,
            accel.y,
            accel.z,
            yaw_rate.to_degrees()
        ),
        Data::Baro { height } => format!("height {height:.2} m"),
        Data::Gps { std, .. } if !std.is_finite() => "no fix (sky masked)".into(),
        Data::Gps { position, std } => format!("({:.1}, {:.1}, {:.1}) m ±{std:.1}", position.x, position.y, position.z),
    }
}
