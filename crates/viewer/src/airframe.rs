//! Airframe section of the side panel: battery, ground effect, and the four motors,
//! with failure injection (a motor's health is the share of its thrust it can still
//! give).

use bevy_egui::egui;
use drone_spec::Airframe;
use world::drone::{DroneState, G};

/// Motor order of the dynamics' mixer.
const MOTORS: [&str; 4] = ["front right", "rear left", "front left", "rear right"];

/// Shows the airframe state. Returns new motor healths if the user changed them.
pub fn panel(ui: &mut egui::Ui, airframe: &Airframe, state: Option<&DroneState>) -> Option<[f32; 4]> {
    let Some(s) = state else {
        ui.label("No drone.");
        return None;
    };
    let f = &s.flight;
    ui.label(format!("{:.1} kg, thrust/weight {:.1}", airframe.mass_kg, airframe.thrust_to_weight));
    match airframe.battery {
        Some(b) => {
            ui.add(egui::ProgressBar::new(f.soc).text(format!("battery {:.0} %", 100.0 * f.soc)));
            ui.label(format!(
                "{:.1} V ({:.1}–{:.1}), {:.1} A, {:.2} Wh drawn of {:.0}",
                f.voltage, b.v_empty, b.v_full, f.current, f.energy_wh, b.capacity_wh
            ));
        }
        None => {
            ui.label(format!("No battery modelled (endless energy): {:.2} Wh drawn", f.energy_wh));
        }
    }
    if f.ground_effect > 1.001 {
        ui.label(format!("Ground effect: thrust ×{:.2}", f.ground_effect));
    }
    ui.separator();
    ui.label("Motors: thrust, and health (drag down to fail one)");
    let max = airframe.thrust_to_weight * G / 4.0;
    let mut health = f.motor_health;
    for (i, name) in MOTORS.iter().enumerate() {
        ui.push_id(i, |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([70.0, 16.0], egui::Label::new(*name));
                ui.add(egui::ProgressBar::new((f.motors[i] / max).clamp(0.0, 1.0)).desired_width(80.0));
                ui.add(egui::Slider::new(&mut health[i], 0.0..=1.0).show_value(false));
            });
        });
    }
    if ui.button("Restore all motors").clicked() {
        health = [1.0; 4];
    }
    (health != f.motor_health).then_some(health)
}
