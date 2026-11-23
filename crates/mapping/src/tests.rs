use super::*;
use drone_spec::DroneSpec;
use sensors::{Pose, SensorSuite};
use world::{World, WorldSpec};

fn grid() -> OccupancyGrid {
    // 20 m map, 0.5 m cells, slab from 1 to 3 m.
    OccupancyGrid::new(20.0, 0.5, 1.0, 3.0)
}

#[test]
fn horizontal_hit_frees_path_and_marks_end() {
    let mut g = grid();
    for _ in 0..3 {
        g.integrate_ray(Vec3::new(0.1, 2.0, 0.1), Vec3::X, Some(5.0), 10.0, 1.0);
    }
    let (ie, je) = g.index(5.1, 0.1).unwrap();
    assert_eq!(g.cell(ie, je), Cell::Occupied);
    for x in [0.5, 2.0, 4.0] {
        let (i, j) = g.index(x, 0.1).unwrap();
        assert_eq!(g.cell(i, j), Cell::Free, "x = {x}");
    }
    // Beyond the hit: unknown.
    let (i, j) = g.index(7.0, 0.1).unwrap();
    assert_eq!(g.cell(i, j), Cell::Unknown);
}

#[test]
fn ground_hit_below_slab_is_not_an_obstacle() {
    let mut g = grid();
    // From 2 m, 45° downwards: leaves the slab (y = 1) at 1 m horizontal
    // distance, hits the ground (y = 0) at 2 m.
    let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
    for _ in 0..3 {
        g.integrate_ray(Vec3::new(0.1, 2.0, 0.1), dir, Some(2.0 * 2f32.sqrt()), 10.0, 1.0);
    }
    let c = g.counts();
    assert_eq!(c.occupied, 0);
    assert!(c.free >= 2);
    let (i, j) = g.index(1.8, 0.1).unwrap();
    assert_eq!(g.cell(i, j), Cell::Unknown, "outside the slab: nothing learned");
}

#[test]
fn ray_from_above_only_frees_inside_slab() {
    let mut g = grid();
    // From 6 m, above the slab, slanting downwards.
    let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
    for _ in 0..3 {
        g.integrate_ray(Vec3::new(0.1, 6.0, 0.1), dir, None, 20.0, 1.0);
    }
    // Within the slab between x ≈ 3 (y = 3) and x ≈ 5 (y = 1).
    let (i, j) = g.index(1.0, 0.1).unwrap();
    assert_eq!(g.cell(i, j), Cell::Unknown);
    let (i, j) = g.index(4.0, 0.1).unwrap();
    assert_eq!(g.cell(i, j), Cell::Free);
}

#[test]
fn no_return_can_be_ignored() {
    let mut g = grid();
    g.miss_as_free = false;
    assert_eq!(g.integrate_ray(Vec3::new(0.0, 2.0, 0.0), Vec3::X, None, 10.0, 1.0), 0);
    assert_eq!(g.counts().free, 0);
}

#[test]
fn log_odds_are_clamped_so_the_map_can_change_its_mind() {
    let mut g = grid();
    let o = Vec3::new(0.1, 2.0, 0.1);
    for _ in 0..100 {
        g.integrate_ray(o, Vec3::X, Some(5.0), 10.0, 1.0);
    }
    let (i, j) = g.index(5.1, 0.1).unwrap();
    assert_eq!(g.logodds(i, j), g.params.max);
    // The obstacle disappears (the ray comes back from a farther wall): a few passes
    // are enough to free the cell.
    let mut passes = 0;
    while g.cell(i, j) != Cell::Free {
        g.integrate_ray(o, Vec3::X, Some(8.0), 10.0, 1.0);
        passes += 1;
        assert!(passes < 20);
    }
}

#[test]
fn rays_leaving_the_map_are_cut() {
    let mut g = grid();
    g.integrate_ray(Vec3::new(9.0, 2.0, 0.0), Vec3::X, Some(5.0), 10.0, 1.0);
    assert!(g.counts().free <= 2);
    assert_eq!(g.counts().occupied, 0);
}

/// Map slab around the flight altitude, as in the viewer.
fn slab(world: &World, altitude: f32) -> (f32, f32) {
    let y = world.terrain.height_at(0.0, 0.0) + altitude;
    (y - 0.5, y + 0.5)
}

/// Makes the drone measure at several points and integrates everything into the map.
fn map_with(world: &World, spec: &DroneSpec, points: &[(f32, f32, f32)]) -> OccupancyGrid {
    let (lo, hi) = slab(world, 2.0);
    let mut g = OccupancyGrid::new(world.terrain.half_size() * 2.0, 0.25, lo, hi);
    let mut suite = SensorSuite::new(&spec.sensors, 1);
    let mut t = 0.0;
    for &(x, z, yaw) in points {
        let pos = Vec3::new(x, (lo + hi) / 2.0, z);
        let body = Pose::new(pos, Quat::from_rotation_y(yaw));
        // One second per point: several measurements from each sensor.
        for _ in 0..100 {
            let before: Vec<_> = suite.latest.iter().map(|m| m.as_ref().map(|m| m.t)).collect();
            suite.update(world, &body, t);
            for (k, m) in suite.latest.iter().enumerate() {
                if let Some(m) = m
                    && Some(m.t) != before[k]
                {
                    g.integrate(m);
                }
            }
            t += 0.01;
        }
    }
    g
}

fn corridor() -> World {
    World::build(&WorldSpec::from_ron(include_str!("../../../maps/corridor.ron")).unwrap()).unwrap()
}

fn assert_coherent(world: &World, g: &OccupancyGrid, min_occupied: usize) {
    let a = g.agreement(|x, z| {
        let c = world.column(x, z, g.y_lo, g.y_hi);
        (c.matter, c.space)
    });
    assert!(a.occupied >= min_occupied, "too few obstacles mapped: {a:?}");
    assert!(a.occupied_ratio() >= 0.97, "occupied: {a:?}");
    assert!(a.free_ratio() >= 0.995, "free: {a:?}");
}

#[test]
fn scout_maps_the_corridor_coherently() {
    let w = corridor();
    let g = map_with(&w, &DroneSpec::scout(), &[(0.0, 0.0, 0.0), (0.0, -15.0, 0.0), (0.0, -28.0, 0.0), (12.0, 0.0, 0.0)]);
    assert_coherent(&w, &g, 100);
    // The left wall is known as occupied, the middle of the corridor as free.
    let (i, j) = g.index(-4.0, -20.0).unwrap();
    assert_eq!(g.cell(i, j), Cell::Occupied);
    let (i, j) = g.index(0.0, -20.0).unwrap();
    assert_eq!(g.cell(i, j), Cell::Free);
}

#[test]
fn heavy_depth_camera_maps_coherently() {
    let w = corridor();
    let heavy = DroneSpec::from_ron(include_str!("../../../drones/heavy.ron")).unwrap();
    // Forward camera: spin in place to cover.
    let turns: Vec<_> = (0..8).map(|k| (0.0, -10.0, k as f32 * std::f32::consts::FRAC_PI_4)).collect();
    let g = map_with(&w, &heavy, &turns);
    assert_coherent(&w, &g, 100);
}

#[test]
fn hills_crossing_the_slab_are_mapped_as_obstacles() {
    let w = World::build(&WorldSpec::procedural(42)).unwrap();
    let g = map_with(&w, &DroneSpec::scout(), &[(0.0, 0.0, 0.0), (8.0, 8.0, 0.0), (-8.0, 5.0, 0.0)]);
    assert_coherent(&w, &g, 10);
}

#[test]
fn lost_beams_do_not_empty_known_walls() {
    let mut g = grid();
    let o = Vec3::new(0.1, 2.0, 0.1);
    for _ in 0..5 {
        g.integrate_ray(o, Vec3::X, Some(5.0), 10.0, 1.0);
    }
    // Dropouts (no return) in the same direction: the wall stays, and nothing behind.
    for _ in 0..50 {
        g.integrate_ray(o, Vec3::X, None, 10.0, 1.0);
    }
    let (i, j) = g.index(5.1, 0.1).unwrap();
    assert_eq!(g.cell(i, j), Cell::Occupied);
    let (i, j) = g.index(6.0, 0.1).unwrap();
    assert_eq!(g.cell(i, j), Cell::Unknown);
}

/// A cell receiving as many hits as free passes stays unknown and
/// becomes ambiguous; a cell simply not yet seen does not.
#[test]
fn contradictory_evidence_makes_a_cell_ambiguous() {
    let mut g = grid();
    // Rays along a row of cells, through their centers.
    let origin = Vec3::new(0.0, 1.0, 0.25);
    let (i, j) = g.index(3.1, 0.25).unwrap();
    for _ in 0..3 {
        // Hit at 3.0 m (the next cell, pushed by half a cell), then a ray
        // crossing it and stopping farther.
        g.integrate_ray(origin, Vec3::X, Some(3.0), 10.0, 0.47);
        g.integrate_ray(origin, Vec3::X, Some(6.0), 10.0, 1.0);
    }
    assert_eq!(g.cell(i, j), Cell::Unknown, "log-odds {}", g.logodds(i, j));
    assert!(g.ambiguous(i, j));
    let (a, b) = g.index(-5.0, -5.0).unwrap();
    assert!(!g.ambiguous(a, b));
}
