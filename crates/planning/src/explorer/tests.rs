//! Full missions, headless: real world, sensors, map, explorer.

use super::*;
use drone_spec::DroneSpec;
use mapping::Cell;
use sensors::{Pose, SensorSuite};
use world::{World, WorldSpec};

struct Outcome {
    status: Status,
    seconds: f32,
    collisions: u32,
    /// Share of the really empty area (slab) that the map knows as free.
    coverage: f32,
    distance_m: f32,
    /// Farthest horizontal distance from the take-off point, in m.
    farthest: f32,
}

fn run(world: &World, spec: &DroneSpec, strategy: Strategy, limit_s: f32) -> Outcome {
    run_with(world, spec, ExplorerConfig::new(strategy, &spec.airframe), limit_s)
}

fn run_with(world: &World, spec: &DroneSpec, cfg: ExplorerConfig, limit_s: f32) -> Outcome {
    const DT: f32 = 0.01;
    let mut state = DroneState::spawn(world, 0.0, 0.0, 2.0);
    let y = state.position.y;
    let mut grid = OccupancyGrid::new(world.terrain.half_size() * 2.0, 0.25, y - 0.5, y + 0.5);
    grid.mark_free_disk(Vec2::new(0.0, 0.0), 1.0);
    let mut suite = SensorSuite::new(&spec.sensors, 1);
    let mut explorer = Explorer::new(cfg, y);
    let mut last_t = vec![f64::NEG_INFINITY; spec.sensors.len()];
    let (mut collisions, mut touching, mut distance_m) = (0, false, 0.0);
    let mut farthest = 0.0_f32;
    let mut t = 0.0_f32;
    while t < limit_s && explorer.status == Status::Exploring {
        let cmd = explorer.update(&grid, &state, &spec.airframe, DT);
        let before = state.position;
        state.step(cmd, &spec.airframe, world, DT);
        distance_m += before.distance(state.position);
        farthest = farthest.max(Vec2::new(state.position.x, state.position.z).length());
        let hit = state.flight.contact;
        // `EXPLORE_TRACE=N`: trace every N hundredths of a second, and at each collision.
        if std::env::var_os("EXPLORE_TRACE").is_some() && ((t * 100.0).round() as i32 % std::env::var("EXPLORE_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(50) == 0 || hit && !touching || explorer.status != Status::Exploring) {
            eprintln!("t={t:.2} pos=({:.2},{:.2},{:.2}) v={:.2} target={:?} path={} frontiers={} hit={hit}", state.position.x, state.position.y, state.position.z, state.velocity.length(), explorer.target, explorer.path.len(), explorer.frontiers.len());
        }
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
    // Coverage: cells whose column is entirely empty within the slab.
    let (nx, nz) = grid.size();
    let (mut empty, mut known) = (0, 0);
    for j in 0..nz {
        for i in 0..nx {
            let c = grid.center(i, j);
            let col = world.column(c.x, c.y, grid.y_lo, grid.y_hi);
            if !col.matter {
                empty += 1;
                known += (grid.cell(i, j) == Cell::Free) as usize;
            }
        }
    }
    Outcome {
        status: explorer.status,
        seconds: t,
        collisions,
        coverage: known as f32 / empty as f32,
        distance_m,
        farthest,
    }
}

fn corridor() -> World {
    World::build(&WorldSpec::from_ron(include_str!("../../../../maps/corridor.ron")).unwrap()).unwrap()
}

fn check(name: &str, o: &Outcome, min_coverage: f32) {
    println!(
        "{name}: {:?} in {:.0} s, {:.0} m travelled, coverage {:.1} %, {} collision(s)",
        o.status,
        o.seconds,
        o.distance_m,
        100.0 * o.coverage,
        o.collisions
    );
    assert_eq!(o.status, Status::Done, "{name}: exploration not finished");
    assert_eq!(o.collisions, 0, "{name}: collisions");
    assert!(o.coverage >= min_coverage, "{name}: coverage {:.3}", o.coverage);
}

#[test]
fn scout_explores_the_corridor_nearest_frontier() {
    let o = run(&corridor(), &DroneSpec::scout(), Strategy::Nearest, 600.0);
    check("corridor, nearest", &o, 0.95);
}

#[test]
fn scout_explores_the_corridor_information_gain() {
    let o = run(&corridor(), &DroneSpec::scout(), Strategy::InfoGain, 600.0);
    check("corridor, info gain", &o, 0.95);
}

/// Restricted to a 10 m disk around the take-off point: the explorer stops
/// once the disk is mapped, instead of going down the whole corridor.
#[test]
fn zone_limits_the_exploration() {
    let spec = DroneSpec::scout();
    let mut cfg = ExplorerConfig::new(Strategy::Nearest, &spec.airframe);
    cfg.zone = Some(crate::Zone { center: Vec2::ZERO, radius: 10.0 });
    let o = run_with(&corridor(), &spec, cfg, 300.0);
    println!("zone of 10 m: {:?} in {:.0} s, farthest {:.1} m", o.status, o.seconds, o.farthest);
    assert_eq!(o.status, Status::Done);
    assert_eq!(o.collisions, 0);
    assert!(o.farthest < 12.0, "left the zone: {} m", o.farthest);
    let full = run(&corridor(), &spec, Strategy::Nearest, 600.0);
    assert!(o.seconds < full.seconds / 2.0);
}

#[test]
fn wrap_stays_in_half_turn() {
    use std::f32::consts::PI;
    assert!((wrap(3.0 * PI / 2.0) + PI / 2.0).abs() < 1e-5);
    assert!((wrap(-3.0 * PI / 2.0) - PI / 2.0).abs() < 1e-5);
    assert!((wrap(0.1) - 0.1).abs() < 1e-6);
}

/// 200 m procedural map: terrain and 60 obstacles. Long (≈ 4 min):
/// `cargo test -p planning -- --ignored --nocapture`.
#[test]
#[ignore]
fn scout_explores_procedural_map() {
    let w = World::build(&WorldSpec::procedural(42)).unwrap();
    for strategy in [Strategy::Nearest, Strategy::InfoGain] {
        let o = run(&w, &DroneSpec::scout(), strategy, 3000.0);
        // Part of the map is cut off by the hills crossing the slab.
        check(&format!("seed 42, {strategy:?}"), &o, 0.6);
    }
}
