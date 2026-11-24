use drone_spec::DroneSpec;
use glam::{Vec2, Vec3};
use mapping::{OccupancyGrid, VoxelMap};
use sensors::{Pose, SensorSuite};
use world::drone::DroneState;
use world::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainSpec, World, WorldSpec};

use super::*;
use crate::{Flight3d, Flight3dConfig};

fn obstacle(shape: ObstacleShape, x: f32, z: f32, elevation_m: f32) -> ObstacleSpec {
    ObstacleSpec { shape, x, z, yaw_deg: 0.0, elevation_m }
}

/// A house (12 × 8 m, 5 m walls, gable roof up to 8 m) 20 m north of the take-off
/// point, a tree beside it, and a tall building far away that must not be visited.
fn estate() -> World {
    World::build(&WorldSpec {
        size_m: 140.0,
        cell_m: 1.0,
        terrain: TerrainSpec::Flat { height: 0.0 },
        obstacles: ObstaclesSpec::List(vec![
            obstacle(ObstacleShape::Box { size: Vec3::new(12.0, 5.0, 8.0) }, 0.0, -20.0, 0.0),
            obstacle(ObstacleShape::Roof { size: Vec3::new(13.0, 3.0, 9.0) }, 0.0, -20.0, 5.0),
            obstacle(ObstacleShape::Cylinder { radius: 0.3, height: 7.0 }, -10.5, -20.0, 0.0),
            obstacle(ObstacleShape::Box { size: Vec3::new(20.0, 25.0, 20.0) }, 45.0, 40.0, 0.0),
        ]),
        wind: Default::default(),
    })
    .unwrap()
}

struct Outcome {
    inspector: Inspector,
    voxels: VoxelMap,
    collisions: u32,
    /// Farthest the drone went from the house, in m.
    farthest: f32,
    end: Vec2,
    seconds: f32,
    /// Tilt reversals per second along the survey: a shaking drone (and camera)
    /// reverses its tilt often.
    reversals_per_s: f32,
}

fn inspect(world: &World, point: Vec2, limit_s: f32) -> Outcome {
    const DT: f32 = 0.01;
    let spec = DroneSpec::mapper();
    let af = spec.airframe;
    let mut state = DroneState::spawn(world, 0.0, 0.0, 2.0);
    let y = state.position.y;
    let mut grid = OccupancyGrid::new(world.terrain.half_size() * 2.0, 0.25, y - 0.5, y + 0.5);
    grid.mark_free_disk(Vec2::ZERO, 1.0);
    let mut voxels = VoxelMap::new(0.4);
    let mut suite = SensorSuite::new(&spec.sensors, 1);
    let mut last_t = vec![f64::NEG_INFINITY; spec.sensors.len()];
    let mut flight = Flight3d::new(Flight3dConfig::default());
    let mut inspector = Inspector::new(InspectConfig::default(), point, Vec2::ZERO);
    let (mut collisions, mut touching, mut farthest) = (0, false, 0.0_f32);
    let mut t = 0.0_f32;
    // Tilt reversals during the survey: the tilt rate turning back on itself.
    let (mut reversals, mut survey_steps, mut prev_rate, mut prev_up) = (0u32, 0u32, Vec3::ZERO, state.flight.up);
    while t < limit_s && !matches!(inspector.phase, Phase::Done | Phase::NotFound) {
        let (cmd, wanted) = inspector.update(&grid, &voxels, &state, &af, DT);
        let cmd = flight.guard(&voxels, &state, &af, cmd, DT, wanted);
        if inspector.phase == Phase::Survey {
            let rate = (state.flight.up - prev_up) / DT;
            if rate.length() > 0.1 && prev_rate.length() > 0.1 && rate.dot(prev_rate) < 0.0 {
                reversals += 1;
            }
            prev_rate = rate;
            survey_steps += 1;
        }
        prev_up = state.flight.up;
        state.step(cmd, &af, world, DT);
        let hit = state.flight.contact;
        if hit && !touching && std::env::var_os("INSPECT_TRACE").is_some() {
            println!("HIT t={t:.1} at {} v {} phase {:?} next {}/{} wp {:?}", state.position, state.velocity, inspector.phase, inspector.next, inspector.plan.len(), inspector.plan.get(inspector.next).map(|w| w.pos));
        }
        collisions += (hit && !touching) as u32;
        touching = hit;
        farthest = farthest.max(Vec2::new(state.position.x, state.position.z).distance(point));
        suite.update(world, &Pose::from(&state), t as f64);
        for (k, m) in suite.latest.iter().enumerate() {
            if let Some(m) = m
                && m.t > last_t[k]
            {
                voxels.integrate(m);
                last_t[k] = m.t;
            }
        }
        let dirty = voxels.take_dirty();
        flight.project(&voxels, &mut grid, &dirty);
        grid.mark_free_disk(Vec2::new(state.position.x, state.position.z), af.radius);
        t += DT;
    }
    println!(
        "{:?} in {:.0} s, target {:?}, {} waypoints, {} skipped, {} collision(s), farthest {:.1} m",
        inspector.phase,
        t,
        inspector.target,
        inspector.plan.len(),
        inspector.skipped,
        collisions,
        farthest
    );
    let reversals_per_s = reversals as f32 / (survey_steps.max(1) as f32 * DT);
    println!("survey: {reversals_per_s:.2} tilt reversals/s");
    Outcome {
        end: Vec2::new(state.position.x, state.position.z),
        inspector,
        voxels,
        collisions,
        farthest,
        seconds: t,
        reversals_per_s,
    }
}

#[test]
fn ring_goes_around_at_the_standoff() {
    let half = Vec2::new(6.0, 4.0);
    let r = ring(Vec2::ZERO, half, 4.0, 2.0);
    assert!(r.len() > 20);
    for w in r.windows(2) {
        assert!(w[0].distance(w[1]) < 2.0 + 1e-3);
    }
    for p in &r {
        // Distance from the box: the standoff.
        let d = (p.abs() - half).max(Vec2::ZERO).length();
        assert!((d - 4.0).abs() < 0.05, "{p} at {d}");
    }
}

#[test]
fn inspects_the_house_only_and_comes_back() {
    let o = inspect(&estate(), Vec2::new(0.0, -20.0), 240.0);
    assert_eq!(o.inspector.phase, Phase::Done);
    assert_eq!(o.collisions, 0);
    // The structure found is the house, not the tree beside it.
    let t = o.inspector.target.unwrap();
    assert!((t.center - Vec2::new(0.0, -20.0)).length() < 1.5, "{t:?}");
    assert!((t.half - Vec2::new(6.5, 4.5)).abs().max_element() < 1.5, "{t:?}");
    assert!((t.top - 8.0).abs() < 1.0, "{t:?}");
    // Never wandered off towards the far building.
    assert!(o.farthest < 20.0, "{}", o.farthest);
    assert!(o.end.length() < 1.0, "not home: {}", o.end);
    assert!(o.inspector.skipped < o.inspector.plan.len() / 4);
    // All four walls and the roof are mapped.
    let count = |f: &dyn Fn(Vec3) -> bool| o.voxels.occupied().filter(|&k| f(o.voxels.center(k))).count();
    let wall = |x0: f32, x1: f32, z0: f32, z1: f32| {
        count(&|c: Vec3| (1.0..4.5).contains(&c.y) && (x0..x1).contains(&c.x) && (z0..z1).contains(&c.z))
    };
    for (name, n) in [
        ("south", wall(-5.0, 5.0, -16.5, -15.5)),
        ("north", wall(-5.0, 5.0, -24.5, -23.5)),
        ("east", wall(5.5, 6.5, -23.0, -17.0)),
        ("west", wall(-6.5, -5.5, -23.0, -17.0)),
    ] {
        assert!(n > 40, "{name} wall: {n} voxels");
    }
    let roof = count(&|c: Vec3| c.y > 5.5 && c.x.abs() < 6.0 && (c.z + 20.0).abs() < 4.0);
    assert!(roof > 150, "roof: {roof} voxels");
    println!("inspection done in {:.0} s", o.seconds);
    // Smooth flight along the rings: no stop-and-go at each waypoint (it shook
    // the drone at about 0.9 reversals per second).
    assert!(o.reversals_per_s < 0.3, "{} tilt reversals/s", o.reversals_per_s);
}

#[test]
fn nothing_to_inspect_on_open_ground() {
    let o = inspect(&estate(), Vec2::new(-30.0, 30.0), 120.0);
    assert_eq!(o.inspector.phase, Phase::NotFound);
    assert_eq!(o.collisions, 0);
}

/// The house map, in its gusty wind: the house and its garage are inspected, the
/// big building across the way is left alone.
#[test]
fn inspects_the_house_of_the_house_map() {
    let w = World::build(&WorldSpec::from_ron(include_str!("../../../../maps/house.ron")).unwrap()).unwrap();
    let o = inspect(&w, Vec2::new(-22.0, -18.0), 400.0);
    assert_eq!(o.inspector.phase, Phase::Done);
    assert_eq!(o.collisions, 0);
    let t = o.inspector.target.unwrap();
    // House (x -29..-15) and garage (to x = -9), not the building (x >= 11).
    assert!(t.center.x - t.half.x > -31.0 && t.center.x + t.half.x < -5.0, "{t:?}");
    assert!(o.farthest < 30.0, "{}", o.farthest);
}
