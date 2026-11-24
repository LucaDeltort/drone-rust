//! Docked layout, Blender style: the 3D view in the middle, the settings in side
//! panels around it (never over it), a status bar at the bottom, and only the
//! flight telemetry as text in the top right corner of the view.
//!
//! egui draws on its own camera, over the whole window; the outside camera
//! renders into the central rectangle left by the panels, and the onboard camera
//! into its slot of the right panel.

use bevy::{
    camera::{CameraOutputMode, Viewport, visibility::RenderLayers},
    prelude::*,
    render::render_resource::BlendState,
    window::PrimaryWindow,
};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, EguiTextureHandle, PrimaryEguiContext, egui};

use crate::{
    CurrentWorld, OuterCamera,
    drone::{DroneConfig, FPV_ASPECT, FpvCamera, FpvFov, SimDrone},
    explore::{self, Exploration},
    map3d::{self, Map3d},
    mapview::{self, Map, MapOverlay, MapSettings},
    mission::{self, Mission},
    pose::{self, Estimator},
    sensing::{self, DebugView, Suite},
    wind::{self, Streaks},
};

/// Rectangles left by the panels, in logical pixels.
#[derive(Resource, Default)]
pub struct Dock {
    /// Central 3D view.
    pub view: Option<egui::Rect>,
    /// Onboard camera slot (`None`: section collapsed).
    pub fpv: Option<egui::Rect>,
}

/// Texts of the corner overlay and of the status bar, filled by the HUD system.
#[derive(Resource, Default)]
pub struct HudText {
    /// Drone telemetry, top right of the view.
    pub telemetry: String,
    /// World source and status, status bar.
    pub status: String,
    /// Keys of the current control mode, status bar.
    pub controls: String,
}

const LEFT_W: f32 = 290.0;
const RIGHT_W: f32 = 300.0;

pub struct DockPlugin;

impl Plugin for DockPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Dock>()
            .init_resource::<HudText>()
            .add_systems(Startup, spawn_ui_camera)
            .add_systems(EguiPrimaryContextPass, layout)
            .add_systems(Update, apply_viewports);
    }
}

/// egui's own camera: the whole window, drawn over the 3D views.
fn spawn_ui_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 10,
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderLayers::none(),
        PrimaryEguiContext,
    ));
}

/// A collapsible section of a side panel.
fn section(ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(egui::RichText::new(title).strong()).default_open(open).show(ui, add);
    ui.add_space(2.0);
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn layout(
    mut contexts: EguiContexts,
    mut dock: ResMut<Dock>,
    hud: Res<HudText>,
    config: Res<DroneConfig>,
    (mut ex, mut mission, mut map, mut est): (ResMut<Exploration>, ResMut<Mission>, ResMut<Map>, ResMut<Estimator>),
    (mut world, mut streaks, mut m3): (ResMut<CurrentWorld>, ResMut<Streaks>, ResMut<Map3d>),
    (mut view, mut fov, mut suites): (ResMut<DebugView>, ResMut<FpvFov>, Query<&mut Suite>),
    (mut settings, mut overlays, drones): (
        ResMut<MapSettings>,
        Query<&mut Visibility, With<MapOverlay>>,
        Query<&SimDrone>,
    ),
) -> Result {
    let texture = contexts.add_image(EguiTextureHandle::Strong(map.image.clone()));
    let ctx = contexts.ctx_mut()?.clone();
    let drone = drones.single().ok();
    let mut root = egui::Ui::new(
        ctx.clone(),
        "dock".into(),
        egui::UiBuilder::new().layer_id(egui::LayerId::background()).max_rect(ctx.viewport_rect()),
    );

    egui::Panel::bottom("status_bar").resizable(false).show(&mut root, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(&hud.status);
            ui.separator();
            ui.weak(&hud.controls);
        });
    });

    egui::Panel::left("left_panel").resizable(true).default_size(LEFT_W).show(&mut root, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            section(ui, "Mission A -> B", true, |ui| mission::panel(ui, &mut mission, &mut ex, &map, &config));
            section(ui, "Exploration", true, |ui| explore::panel(ui, &mut ex, &mut mission, &map, &config));
            section(ui, "3D map & flight", true, |ui| map3d::panel(ui, &mut m3, &mut ex, &mut mission));
            section(ui, "Wind", true, |ui| wind::panel(ui, &mut world, &mut streaks, drone));
            section(ui, "Pose", false, |ui| pose::panel(ui, &mut est, &config));
        });
    });

    let mut fpv = None;
    egui::Panel::right("right_panel").resizable(true).default_size(RIGHT_W).show(&mut root, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            section(ui, "Onboard view", true, |ui| {
                let w = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(egui::vec2(w, w / FPV_ASPECT), egui::Sense::hover());
                ui.painter().rect_stroke(rect, 0.0, (1.0, egui::Color32::GRAY), egui::StrokeKind::Outside);
                fpv = Some(rect);
            });
            section(ui, "Map", true, |ui| {
                mapview::panel(ui, &mut map, &mut settings, &mut overlays, drone, &ex, texture);
            });
            section(ui, "Sensors", false, |ui| sensing::panel(ui, &mut view, &mut fov, &mut suites));
        });
    });

    // What the panels left: the 3D view.
    let view_rect = root.available_rect_before_wrap();
    dock.view = Some(view_rect);
    // The onboard view slot, unless scrolled out of the panel.
    dock.fpv = fpv.filter(|r| r.min.y >= 0.0 && r.max.y <= view_rect.max.y);

    // Telemetry, top right of the 3D view.
    egui::Area::new("telemetry".into())
        .order(egui::Order::Foreground)
        .interactable(false)
        .pivot(egui::Align2::RIGHT_TOP)
        .fixed_pos(view_rect.right_top() + egui::vec2(-10.0, 10.0))
        .show(&ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(110))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(&hud.telemetry).monospace().color(egui::Color32::WHITE));
                });
        });
    Ok(())
}

/// Logical egui rectangle to a physical camera viewport, clamped to the window.
fn to_viewport(r: egui::Rect, window: &Window) -> Option<Viewport> {
    let s = window.scale_factor();
    let size = window.physical_size();
    let x0 = ((r.min.x * s).max(0.0) as u32).min(size.x);
    let y0 = ((r.min.y * s).max(0.0) as u32).min(size.y);
    let x1 = ((r.max.x * s).max(0.0) as u32).min(size.x);
    let y1 = ((r.max.y * s).max(0.0) as u32).min(size.y);
    (x1 > x0 + 1 && y1 > y0 + 1).then(|| Viewport {
        physical_position: UVec2::new(x0, y0),
        physical_size: UVec2::new(x1 - x0, y1 - y0),
        ..default()
    })
}

fn set_viewport(cam: &mut Camera, viewport: Option<Viewport>) {
    let key = |v: &Option<Viewport>| v.as_ref().map(|v| (v.physical_position, v.physical_size));
    if key(&cam.viewport) != key(&viewport) {
        cam.viewport = viewport;
    }
}

fn apply_viewports(
    dock: Res<Dock>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut outer: Query<&mut Camera, (With<OuterCamera>, Without<FpvCamera>)>,
    mut fpv: Query<&mut Camera, (With<FpvCamera>, Without<OuterCamera>)>,
) {
    if let Some(r) = dock.view {
        for mut cam in &mut outer {
            set_viewport(&mut cam, to_viewport(r, &window));
        }
    }
    for mut cam in &mut fpv {
        let viewport = dock.fpv.and_then(|r| to_viewport(r, &window));
        let active = viewport.is_some();
        if cam.is_active != active {
            cam.is_active = active;
        }
        if active {
            set_viewport(&mut cam, viewport);
        }
    }
}
