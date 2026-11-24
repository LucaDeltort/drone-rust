use drone_spec::DroneSpec;
use glam::{Vec2, Vec3};
use mapping::{OccupancyGrid, VoxelMap};
use sensors::{Pose, SensorSuite};
use world::drone::DroneState;
use world::{ObstacleShape, ObstacleSpec, ObstaclesSpec, TerrainFeature, TerrainSpec, World, WorldSpec};

use super::*;
use crate::{NavStatus, Navigator, NavigatorConfig};

struct Outcome {
    status: NavStatus,
    seconds: f32,
    collisions: u32,
    distance_m: f32,
    max_y: f32,
    /// Lowest height above the ground during the flight (after take-off).
    min_agl: f32,
}

fn fly(world: &World, goal: Vec2, limit_s: f32) -> Outcome {
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
    let mut nav = Navigator::new(NavigatorConfig::new(&af), y, goal);
    let mut flight = Flight3d::new(Flight3dConfig::default());
    let (mut collisions, mut touching, mut distance_m) = (0, false, 0.0);
    let (mut max_y, mut min_agl) = (f32::NEG_INFINITY, f32::INFINITY);
    let mut t = 0.0_f32;
    while t < limit_s && nav.status != NavStatus::Reached {
        let cmd = nav.update(&grid, &state, &af, DT);
        let cmd = flight.guard(&voxels, &state, &af, cmd, DT, None);

        let before = state.position;
        state.step(cmd, &af, world, DT);
        distance_m += Vec2::new(before.x - state.position.x, before.z - state.position.z).length();
        let hit = state.flight.contact;
        collisions += (hit && !touching) as u32;
        touching = hit;
        max_y = max_y.max(state.position.y);
        if t > 2.0 {
            min_agl = min_agl.min(state.altitude(world));
        }
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
    let o = Outcome { status: nav.status, seconds: t, collisions, distance_m, max_y, min_agl };
    println!(
        "{:?} in {:.1} s, {:.0} m, {} collision(s), max height {:.1} m, min AGL {:.1} m",
        o.status, o.seconds, o.distance_m, o.collisions, o.max_y, o.min_agl
    );
    o
}

fn wall_world(height: f32, terrain: TerrainSpec) -> World {
    World::build(&WorldSpec {
        size_m: 120.0,
        cell_m: 1.0,
        terrain,
        obstacles: ObstaclesSpec::List(vec![ObstacleSpec {
            shape: ObstacleShape::Box { size: Vec3::new(40.0, height, 0.5) },
            x: 0.0,
            z: -15.0,
            yaw_deg: 0.0,
            elevation_m: 0.0,
        }]),
        wind: Default::default(),
    })
    .unwrap()
}

fn flat() -> TerrainSpec {
    TerrainSpec::Flat { height: 0.0 }
}

#[test]
fn low_obstacle_is_flagged_free_and_tall_one_occupied() {
    let mut voxels = VoxelMap::new(0.4);
    // Ground, a 3 m wall at x = 5, a 20 m tower at x = 10 (vertical rays down).
    for x in 0..30 {
        let x = x as f32 * 0.4 + 0.2;
        let top = if (5.0..5.4).contains(&x) {
            3.0
        } else if (10.0..10.4).contains(&x) {
            20.0
        } else {
            0.1
        };
        for _ in 0..3 {
            voxels.insert_ray(Vec3::new(x, 30.0, 0.2), Vec3::new(x, top - 0.05, 0.2), true);
            voxels.insert_ray(Vec3::new(x, 30.0, 0.2), Vec3::new(x, 0.1, 0.2), true);
        }
    }
    let f = Flight3d::new(Flight3dConfig::default());
    let col = |x: f32| voxels.column_key(x, 0.2);
    assert_eq!(f.column_state(&voxels, col(5.2)), Cell::Free);
    assert_eq!(f.column_state(&voxels, col(10.2)), Cell::Occupied);
    assert_eq!(f.column_state(&voxels, col(2.2)), Cell::Free);
    assert_eq!(f.column_state(&voxels, voxels.column_key(50.0, 50.0)), Cell::Unknown);
}

/// A 3 m wall, 40 m wide, between A and B: the drone climbs over it instead of
/// going around (which would take about 50 m).
#[test]
fn flies_over_a_low_wall() {
    let w = wall_world(3.0, flat());
    let o = fly(&w, Vec2::new(0.0, -30.0), 60.0);
    assert_eq!(o.status, NavStatus::Reached);
    assert_eq!(o.collisions, 0);
    assert!(o.distance_m < 36.0, "went around: {} m", o.distance_m);
    assert!(o.max_y > 3.0 + 1.5, "{}", o.max_y);
    assert!(o.min_agl > 1.5, "{}", o.min_agl);
}

/// A 25 m wall is too tall to fly over: the drone goes around it.
#[test]
fn goes_around_a_tall_wall() {
    let w = wall_world(25.0, flat());
    let o = fly(&w, Vec2::new(0.0, -30.0), 120.0);
    assert_eq!(o.status, NavStatus::Reached);
    assert_eq!(o.collisions, 0);
    // No higher than needed to clear the tallest obstacle it may fly over.
    assert!(o.max_y < 17.0, "{}", o.max_y);
}

/// A hill on the way: the drone follows the relief, keeping its clearance.
#[test]
fn follows_the_relief() {
    let terrain = TerrainSpec::Landscape {
        seed: 1,
        amplitude_m: 0.5,
        wavelength_m: 30.0,
        octaves: 2,
        features: vec![TerrainFeature::Hill { x: 0.0, z: -20.0, radius_m: 12.0, height_m: 8.0 }],
    };
    let w = World::build(&WorldSpec {
        size_m: 120.0,
        cell_m: 1.0,
        terrain,
        obstacles: ObstaclesSpec::None,
        wind: Default::default(),
    })
    .unwrap();
    let o = fly(&w, Vec2::new(0.0, -40.0), 60.0);
    assert_eq!(o.status, NavStatus::Reached);
    assert_eq!(o.collisions, 0);
    assert!(o.min_agl > 1.5, "{}", o.min_agl);
    assert!(o.max_y > 8.0, "{}", o.max_y);
}

/// The house map, in its gusty wind: B is behind the house. The drone flies over
/// the roof rather than around it, and touches nothing.
#[test]
fn flies_over_the_house_in_the_wind() {
    let w = World::build(&WorldSpec::from_ron(include_str!("../../../../maps/house.ron")).unwrap()).unwrap();
    let o = fly(&w, Vec2::new(-22.0, -34.0), 90.0);
    assert_eq!(o.status, NavStatus::Reached);
    assert_eq!(o.collisions, 0);
    // The roof ridge tops out at 10 m.
    assert!(o.max_y > 10.0, "{}", o.max_y);
    assert!(o.min_agl > 1.5, "{}", o.min_agl);
}
