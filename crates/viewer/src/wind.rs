//! Wind in the viewer: settings panel, streaks carried by the wind around the
//! drone, and an arrow showing the wind it feels.

use bevy::prelude::*;
use bevy_egui::egui;
use world::WindSpec;
use world::rng::SplitMix64;

use crate::{CurrentWorld, drone::SimDrone};

/// Number of streaks around the drone.
const STREAKS: usize = 400;
/// Half-side of the box of streaks, centered on the drone, in m.
const BOX_M: f32 = 18.0;
/// Lifetime of a streak, in s.
const LIFE_S: f32 = 3.0;

#[derive(Resource)]
pub struct Streaks {
    show: bool,
    points: Vec<(Vec3, f32)>,
    rng: SplitMix64,
}

pub struct WindPlugin;

impl Plugin for WindPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Streaks { show: true, points: Vec::new(), rng: SplitMix64::new(17) })
            .add_systems(Update, (advect, draw).chain())
;
    }
}

fn spawn_point(rng: &mut SplitMix64, center: Vec3, ground: impl Fn(f32, f32) -> f32) -> Vec3 {
    let x = center.x + rng.range(-BOX_M, BOX_M);
    let z = center.z + rng.range(-BOX_M, BOX_M);
    let y = ground(x, z) + rng.range(0.3, (center.y - ground(x, z)).max(2.0) + BOX_M * 0.5);
    Vec3::new(x, y, z)
}

/// Streaks drift with the local wind (gusts, turbulence, shelter included).
fn advect(time: Res<Time>, world: Res<CurrentWorld>, drones: Query<&SimDrone>, mut s: ResMut<Streaks>) {
    let Ok(d) = drones.single() else { return };
    let w = &world.0;
    if !s.show || w.wind.spec().is_calm() {
        s.points.clear();
        return;
    }
    let center = d.state.position;
    let t = d.state.flight.t;
    let dt = time.delta_secs().min(0.05);
    let ground = |x: f32, z: f32| w.terrain.height_at(x, z);
    let Streaks { points, rng, .. } = s.as_mut();
    while points.len() < STREAKS {
        let age = rng.range(0.0, LIFE_S);
        points.push((spawn_point(rng, center, ground), age));
    }
    for (p, age) in points.iter_mut() {
        *age += dt;
        *p += w.wind_at(*p, t) * dt;
        let out = (p.x - center.x).abs() > BOX_M || (p.z - center.z).abs() > BOX_M || p.y < ground(p.x, p.z);
        if *age > LIFE_S || out {
            *p = spawn_point(rng, center, ground);
            *age = 0.0;
        }
    }
}

fn draw(s: Res<Streaks>, world: Res<CurrentWorld>, drones: Query<&SimDrone>, mut gizmos: Gizmos) {
    let Ok(d) = drones.single() else { return };
    let t = d.state.flight.t;
    for &(p, age) in &s.points {
        let v = world.0.wind_at(p, t);
        // Fade in and out over the lifetime.
        let k = (age / LIFE_S * std::f32::consts::PI).sin();
        gizmos.line(p - v * 0.25, p, Color::srgba(0.95, 0.97, 1.0, 0.55 * k));
    }
    // Wind felt by the drone.
    let w = d.state.flight.wind;
    if w.length() > 0.05 {
        let from = d.state.position + Vec3::Y * 0.6;
        gizmos.arrow(from, from + w * 0.3, Color::srgb(0.3, 0.85, 1.0));
    }
}

/// Wind section of the side panel.
pub fn panel(ui: &mut egui::Ui, world: &mut CurrentWorld, s: &mut Streaks, drone: Option<&SimDrone>) {

            let mut spec: WindSpec = *world.0.wind.spec();
            let before = spec;
            ui.add(egui::Slider::new(&mut spec.speed_ms, 0.0..=20.0).text("mean speed at 10 m (m/s)"));
            ui.add(egui::Slider::new(&mut spec.from_deg, 0.0..=360.0).text("from (°, 0 = north)"));
            ui.add(egui::Slider::new(&mut spec.gust_ms, 0.0..=15.0).text("gusts (m/s)"));
            ui.add(egui::Slider::new(&mut spec.turbulence_ms, 0.0..=5.0).text("turbulence σ (m/s)"));
            ui.horizontal(|ui| {
                if ui.button("Calm").clicked() {
                    spec = WindSpec { seed: spec.seed, ..WindSpec::default() };
                }
                if ui.button("Breeze").clicked() {
                    spec = WindSpec { speed_ms: 4.0, gust_ms: 2.0, turbulence_ms: 0.5, ..spec };
                }
                if ui.button("Storm").clicked() {
                    spec = WindSpec { speed_ms: 12.0, gust_ms: 8.0, turbulence_ms: 2.5, ..spec };
                }
            });
            if spec != before {
                world.0.wind.set(spec);
            }
            ui.checkbox(&mut s.show, "Show the airflow");
            if let Some(d) = drone {
                let w = d.state.flight.wind;
                let (pitch, roll) = d.state.tilt();
                ui.label(format!(
                    "Felt: {:.1} m/s   tilt: pitch {:+.0}°, roll {:+.0}°",
                    w.length(),
                    pitch.to_degrees(),
                    roll.to_degrees()
                ));
                let lean = d.state.flight.integral.length();
                ui.label(format!("Wind compensation learned by the controller: {lean:.2}"))
                    .on_hover_text("integral of the velocity loop: the steady push it leans against");
            }
            ui.label("Downwind of buildings the wind drops and gets gustier.");
}
