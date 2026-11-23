//! Full A-to-B missions, headless: real world, sensors, map, navigator.

use super::*;
use drone_spec::DroneSpec;
use glam::Vec3;
use sensors::{Pose, SensorSuite};
use world::{ObstacleShape, World, WorldSpec};

struct Outcome {
    status: NavStatus,
    seconds: f32,
    collisions: u32,
    distance_m: f32,
    /// Travel time predicted by the first trajectory, in s.
    first_eta: Option<f32>,
    stats: NavStats,
}

/// In-mission event: called at each step with the time and the drone,
/// it can modify the world (unexpected obstacle).
type Event<'a> = &'a mut dyn FnMut(f32, &DroneState, &mut World);

fn run(world: &mut World, spec: &DroneSpec, goal: Vec2, limit_s: f32, event: Event) -> Outcome {
    const DT: f32 = 0.01;
    let mut state = DroneState::spawn(world, 0.0, 0.0, 2.0);
    let y = state.position.y;
    let mut grid = OccupancyGrid::new(world.terrain.half_size() * 2.0, 0.25, y - 0.5, y + 0.5);
    grid.mark_free_disk(Vec2::new(0.0, 0.0), 1.0);
    let mut suite = SensorSuite::new(&spec.sensors, 1);
    let mut nav = Navigator::new(NavigatorConfig::new(&spec.airframe), y, goal);
    let mut last_t = vec![f64::NEG_INFINITY; spec.sensors.len()];
    let (mut collisions, mut touching, mut distance_m) = (0, false, 0.0);
    let mut first_eta = None;
    let mut t = 0.0_f32;
    while t < limit_s && nav.status != NavStatus::Reached {
        event(t, &state, world);
        let cmd = nav.update(&grid, &state, &spec.airframe, DT);
        if first_eta.is_none() {
            first_eta = nav.eta();
        }
        let before = state.position;
        state.step(cmd, &spec.airframe, world, DT);
        distance_m += before.distance(state.position);
        let hit = world.collides(state.position, spec.airframe.radius);
        collisions += (hit && !touching) as u32;
        touching = hit;
        let body = Pose::from(&state);
        grid.mark_free_disk(Vec2::new(state.position.x, state.position.z), spec.airframe.radius);
        suite.update(world, &body, t as f64);
        for (k, m) in suite.latest.iter().enumerate() {
            if let Some(m) = m
                && m.t > last_t[k]
            {
                grid.integrate(m);
                last_t[k] = m.t;
            }
        }
        t += DT;
    }
    Outcome { status: nav.status, seconds: t, collisions, distance_m, first_eta, stats: nav.stats }
}

fn corridor() -> World {
    World::build(&WorldSpec::from_ron(include_str!("../../../../maps/corridor.ron")).unwrap()).unwrap()
}

fn check(name: &str, o: &Outcome, limit_s: f32) {
    println!(
        "{name}: {:?} in {:.1} s (predicted at start {:.1} s), {:.0} m, {} collision(s), \
         {} replans, {} cells expanded, {} cells changed",
        o.status,
        o.seconds,
        o.first_eta.unwrap_or(f32::NAN),
        o.distance_m,
        o.collisions,
        o.stats.replans,
        o.stats.expanded,
        o.stats.changed_cells,
    );
    assert_eq!(o.status, NavStatus::Reached, "{name}: B not reached");
    assert_eq!(o.collisions, 0, "{name}: collisions");
    assert!(o.seconds < limit_s, "{name}: too slow");
}

/// B behind the far wall of the corridor: the straight line is blocked by walls
/// the drone does not know at the start.
#[test]
fn scout_reaches_b_behind_the_corridor() {
    let o = run(&mut corridor(), &DroneSpec::scout(), Vec2::new(0.0, -38.0), 60.0, &mut |_, _, _| {});
    check("corridor, B behind the far wall", &o, 60.0);
    // Straight line: 38 m; through the opening of the far wall: ≈ 39.4 m.
    assert!(o.distance_m > 39.0, "detour expected");
}

#[test]
fn scout_goes_around_the_rotated_block() {
    let o = run(&mut corridor(), &DroneSpec::scout(), Vec2::new(-26.0, 20.0), 40.0, &mut |_, _, _| {});
    check("corridor, behind the block", &o, 40.0);
}

#[test]
fn heavy_reaches_b_with_a_forward_depth_camera() {
    let spec = DroneSpec::from_ron(include_str!("../../../../drones/heavy.ron")).unwrap();
    let o = run(&mut corridor(), &spec, Vec2::new(0.0, -38.0), 120.0, &mut |_, _, _| {});
    check("corridor, Heavy", &o, 120.0);
}

/// A wall appears across the path during flight, on cells already seen
/// free: the map corrects itself, D* Lite repairs the path, the drone avoids it.
#[test]
fn scout_avoids_a_wall_that_appears_in_flight() {
    let goal = Vec2::new(30.0, -4.0);
    let mut dropped = false;
    let mut event = |_t: f32, s: &DroneState, w: &mut World| {
        if !dropped && s.position.x > 4.0 {
            dropped = true;
            // 10 m ahead of the drone, 12 m wide, across.
            w.add_obstacle(ObstacleShape::Box { size: Vec3::new(0.5, 4.0, 12.0) }, s.position.x + 10.0, s.position.z, 0.0);
        }
    };
    let o = run(&mut corridor(), &DroneSpec::scout(), goal, 40.0, &mut event);
    assert!(dropped);
    check("corridor, unexpected wall", &o, 40.0);
}

/// B in an enclosed area: no path, the drone must not rush into a wall.
#[test]
fn enclosed_goal_is_reported_without_collision() {
    let mut w = corridor();
    // Hollow box of 4 walls around (20, -20).
    for (x, z, sx, sz) in [(20.0, -23.0, 6.5, 0.5), (20.0, -17.0, 6.5, 0.5), (17.0, -20.0, 0.5, 6.5), (23.0, -20.0, 0.5, 6.5)] {
        w.add_obstacle(ObstacleShape::Box { size: Vec3::new(sx, 4.0, sz) }, x, z, 0.0);
    }
    let o = run(&mut w, &DroneSpec::scout(), Vec2::new(20.0, -20.0), 40.0, &mut |_, _, _| {});
    println!("B enclosed: {:?} after {:.0} s, {} collision(s)", o.status, o.seconds, o.collisions);
    assert_eq!(o.status, NavStatus::NoPath);
    assert_eq!(o.collisions, 0);
}

/// 200 m procedural map: terrain and 60 obstacles, B about 70 m away.
#[test]
fn scout_crosses_the_procedural_map() {
    let mut w = World::build(&WorldSpec::procedural(42)).unwrap();
    let y = DroneState::spawn(&w, 0.0, 0.0, 2.0).position.y;
    // Free point (empty flight slab around) near (50, -50).
    let goal = (0..400)
        .map(|k| {
            let a = k as f32 * 0.5;
            Vec2::new(50.0, -50.0) + Vec2::new(a.cos(), a.sin()) * (k as f32 * 0.05)
        })
        .find(|p| !w.collides(Vec3::new(p.x, y, p.y), 1.5))
        .expect("free point");
    let o = run(&mut w, &DroneSpec::scout(), goal, 120.0, &mut |_, _, _| {});
    check(&format!("seed 42, B = ({:.1}, {:.1})", goal.x, goal.y), &o, 120.0);
}
