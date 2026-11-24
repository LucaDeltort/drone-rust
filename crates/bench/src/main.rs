//! Headless benchmark: matrix of drones × maps × missions × pose modes ×
//! seeds, exported to CSV.
//!
//! Usage (in release, much faster):
//!   cargo run --release -p bench                          # full matrix
//!
//! Maps: corridor, seedN (procedural), downtown, suburbs,
//! old-town, dense-city, or a path to a RON file.
//!   cargo run --release -p bench -- --drones scout,pro --maps corridor \
//!       --missions explore-nearest,a-b --poses truth,ekf-gps --seeds 1,2 \
//!       --out bench/results.csv
//!   --scan-match off: `ekf` mode without scan matching (the IMU drift alone).
//!
//! Each simulation is independent and deterministic (fixed seed, fixed step); they
//! run in parallel on all cores, and the CSV keeps the matrix order.
//! The compute time per step is measured under this parallelism: comparable between
//! rows of the same run, not in absolute terms.
//!
//! Success: exploration finished without collision, coverage ≥ 90 % and map correct at
//! ≥ 90 %; A to B really reached (ground truth, within 1 m) without collision.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use drone_spec::DroneSpec;
use glam::{Vec2, Vec3};
use mapping::OccupancyGrid;
use planning::{Explorer, ExplorerConfig, NavStatus, Navigator, NavigatorConfig, Status, Strategy};
use sim::{PoseMode, Sim, SimConfig, coverage, empty_cells};
use world::{World, WorldSpec};

const DRONES: [(&str, &str); 4] = [
    ("scout", include_str!("../../../drones/scout.ron")),
    ("heavy", include_str!("../../../drones/heavy.ron")),
    ("cheap", include_str!("../../../drones/cheap.ron")),
    ("pro", include_str!("../../../drones/pro.ron")),
];
/// Hand-drawn or RON-described maps shipped with the project.
const MAPS: [(&str, &str); 5] = [
    ("corridor", include_str!("../../../maps/corridor.ron")),
    ("downtown", include_str!("../../../maps/downtown.ron")),
    ("suburbs", include_str!("../../../maps/suburbs.ron")),
    ("old-town", include_str!("../../../maps/old_town.ron")),
    ("dense-city", include_str!("../../../maps/dense_city.ron")),
];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mission {
    Explore(Strategy),
    AToB,
}

impl Mission {
    fn label(&self) -> &'static str {
        match self {
            Mission::Explore(Strategy::Nearest) => "explore-nearest",
            Mission::Explore(Strategy::InfoGain) => "explore-infogain",
            Mission::AToB => "a-b",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s {
            "explore-nearest" => Some(Mission::Explore(Strategy::Nearest)),
            "explore-infogain" => Some(Mission::Explore(Strategy::InfoGain)),
            "a-b" => Some(Mission::AToB),
            _ => None,
        }
    }
}

struct Args {
    drones: Vec<String>,
    maps: Vec<String>,
    missions: Vec<Mission>,
    poses: Vec<PoseMode>,
    seeds: Vec<u64>,
    explore_limit_s: f32,
    a_b_limit_s: f32,
    out: String,
    threads: usize,
    scan_match: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        drones: DRONES.iter().map(|d| d.0.to_string()).collect(),
        maps: ["corridor", "seed42", "seed7", "downtown", "suburbs", "old-town", "dense-city"]
            .map(String::from)
            .to_vec(),
        missions: vec![Mission::Explore(Strategy::Nearest), Mission::Explore(Strategy::InfoGain), Mission::AToB],
        poses: vec![PoseMode::Truth, PoseMode::Ekf { gps: true }, PoseMode::Ekf { gps: false }],
        seeds: vec![1],
        explore_limit_s: 1200.0,
        a_b_limit_s: 120.0,
        out: "bench/results.csv".into(),
        threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        scan_match: true,
    };
    let list = |v: &str| v.split(',').map(str::to_string).collect::<Vec<_>>();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().ok_or(format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--drones" => a.drones = list(&v),
            "--maps" => a.maps = list(&v),
            "--missions" => {
                a.missions = list(&v).iter().map(|m| Mission::parse(m).ok_or(format!("unknown mission: {m}"))).collect::<Result<_, _>>()?
            }
            "--poses" => {
                a.poses = list(&v).iter().map(|p| PoseMode::parse(p).ok_or(format!("unknown pose: {p}"))).collect::<Result<_, _>>()?
            }
            "--seeds" => a.seeds = list(&v).iter().map(|s| s.parse().map_err(|_| format!("seed: {s}"))).collect::<Result<_, _>>()?,
            "--explore-limit" => a.explore_limit_s = v.parse().map_err(|_| "--explore-limit: number expected")?,
            "--a-b-limit" => a.a_b_limit_s = v.parse().map_err(|_| "--a-b-limit: number expected")?,
            "--out" => a.out = v,
            "--threads" => a.threads = v.parse().map_err(|_| "--threads: integer expected")?,
            "--scan-match" => {
                a.scan_match = match v.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err("--scan-match: on or off expected".into()),
                }
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    Ok(a)
}

fn load_drone(name: &str) -> Result<DroneSpec, String> {
    if let Some((_, text)) = DRONES.iter().find(|d| d.0 == name) {
        return DroneSpec::from_ron(text).map_err(|e| e.to_string());
    }
    let text = std::fs::read_to_string(name).map_err(|e| format!("{name}: {e}"))?;
    DroneSpec::from_ron(&text).map_err(|e| format!("{name}: {e}"))
}

fn load_map(name: &str) -> Result<World, String> {
    let spec = if let Some((_, text)) = MAPS.iter().find(|m| m.0 == name) {
        WorldSpec::from_ron(text).map_err(|e| format!("{name}: {e}"))?
    } else if let Some(seed) = name.strip_prefix("seed") {
        WorldSpec::procedural(seed.parse().map_err(|_| format!("map: {name}"))?)
    } else {
        let text = std::fs::read_to_string(name).map_err(|e| format!("{name}: {e}"))?;
        WorldSpec::from_ron(&text).map_err(|e| format!("{name}: {e}"))?
    };
    World::build(&spec).map_err(|e| e.to_string())
}

/// B: behind the far wall for the corridor; elsewhere, a free point near
/// (50, -50), where the flight slab is empty within 1.5 m around.
fn goal_for(map: &str, world: &World, y: f32) -> Vec2 {
    if map == "corridor" {
        return Vec2::new(0.0, -38.0);
    }
    (0..2000)
        .map(|k| {
            let a = k as f32 * 0.5;
            Vec2::new(50.0, -50.0) + Vec2::new(a.cos(), a.sin()) * (k as f32 * 0.05)
        })
        .find(|p| !world.collides(Vec3::new(p.x, y, p.y), 1.5))
        .unwrap_or(Vec2::new(50.0, -50.0))
}

struct Case {
    drone: String,
    map: String,
    mission: Mission,
    pose: PoseMode,
    seed: u64,
}

#[derive(Default)]
struct Row {
    success: bool,
    end: String,
    time_s: f32,
    distance_m: f32,
    collisions: u32,
    coverage: [Option<f32>; 3],
    map_occupied_ok: f32,
    map_free_ok: f32,
    pose_rms_m: f32,
    pose_max_m: f32,
    step_mean_ms: f32,
    step_max_ms: f32,
    goal: Option<Vec2>,
    /// Electrical energy drawn, in Wh.
    energy_wh: f32,
    /// Battery state of charge at the end (1 without battery).
    soc_end: f32,
}

/// Thresholds of a successful exploration: share of the free area known, share of the
/// map's occupied cells that really are.
const MIN_COVERAGE: f32 = 0.9;
const MIN_MAP_OK: f32 = 0.9;

/// Times at which coverage is recorded, in s (the last one is the mission end).
const COVERAGE_AT: [f32; 2] = [60.0, 180.0];

fn run(case: &Case, args: &Args) -> Result<Row, String> {
    let spec = load_drone(&case.drone)?;
    let world = load_map(&case.map)?;
    let cfg = SimConfig { mode: case.pose, seed: case.seed, scan_match: args.scan_match, ..Default::default() };
    let mut sim = Sim::new(world, &spec, cfg)?;
    let empty = empty_cells(&sim.world, &sim.grid);
    let altitude = sim.altitude();
    let airframe = sim.spec.airframe;
    let dt = sim.cfg.dt;

    let mut row = Row::default();
    let (mut step_total, mut step_max) = (Duration::ZERO, Duration::ZERO);
    let mut steps = 0u32;
    // Compute time of a step: estimation and map (in `Sim`) + decision.
    let mut timed = |sim: &mut Sim, decide: &mut dyn FnMut(&OccupancyGrid, &world::drone::DroneState) -> world::drone::Command| {
        let before = sim.compute;
        let start = Instant::now();
        let cmd = decide(&sim.grid, &sim.belief());
        let decision = start.elapsed();
        sim.step(cmd);
        let total = decision + (sim.compute - before);
        step_total += total;
        step_max = step_max.max(total);
        steps += 1;
    };

    match case.mission {
        Mission::Explore(strategy) => {
            let mut explorer = Explorer::new(ExplorerConfig::new(strategy, &airframe), altitude);
            let mut next_mark = 0;
            while sim.t < args.explore_limit_s && explorer.status == Status::Exploring {
                timed(&mut sim, &mut |g, s| explorer.update(g, s, &airframe, dt));
                if next_mark < COVERAGE_AT.len() && sim.t >= COVERAGE_AT[next_mark] {
                    row.coverage[next_mark] = Some(coverage(&sim.grid, &empty));
                    next_mark += 1;
                }
            }
            // Mission finished before a sample: the coverage no longer moves.
            let last = coverage(&sim.grid, &empty);
            for c in row.coverage[..COVERAGE_AT.len()].iter_mut().skip(next_mark) {
                *c = Some(last);
            }
            row.coverage[2] = Some(last);
            // Successful: finished, without collision, and with a map that covers and is correct
            // (a wrong pose can "finish" an exploration on a useless map).
            let (y_lo, y_hi) = (sim.grid.y_lo, sim.grid.y_hi);
            let occupied_ok = sim
                .grid
                .agreement(|x, z| {
                    let c = sim.world.column(x, z, y_lo, y_hi);
                    (c.matter, c.space)
                })
                .occupied_ratio();
            row.success = explorer.status == Status::Done
                && sim.collisions == 0
                && last >= MIN_COVERAGE
                && occupied_ok >= MIN_MAP_OK;
            row.end = match explorer.status {
                Status::Done => "done".into(),
                Status::Exploring => "timeout".into(),
            };
        }
        Mission::AToB => {
            let goal = goal_for(&case.map, &sim.world, sim.truth.position.y);
            row.goal = Some(goal);
            let mut nav = Navigator::new(NavigatorConfig::new(&airframe), altitude, goal);
            // B is really reached (ground truth), not only according to the drone.
            let reached = |sim: &Sim| Vec2::new(sim.truth.position.x, sim.truth.position.z).distance(goal) < 1.0;
            while sim.t < args.a_b_limit_s && nav.status != NavStatus::Reached {
                timed(&mut sim, &mut |g, s| nav.update(g, s, &airframe, dt));
            }
            let truly = reached(&sim);
            row.success = truly && sim.collisions == 0;
            row.end = match (nav.status, truly) {
                (NavStatus::Reached, true) => "B reached".into(),
                (NavStatus::Reached, false) => "B believed reached (wrong pose)".into(),
                (NavStatus::NoPath, _) => "no path".into(),
                (NavStatus::Flying, _) => "timeout".into(),
            };
        }
    }

    // Map correctness: occupied and free cells matching the real world.
    let (y_lo, y_hi) = (sim.grid.y_lo, sim.grid.y_hi);
    let agreement = sim.grid.agreement(|x, z| {
        let c = sim.world.column(x, z, y_lo, y_hi);
        (c.matter, c.space)
    });
    row.map_occupied_ok = agreement.occupied_ratio();
    row.map_free_ok = agreement.free_ratio();
    row.time_s = sim.t;
    row.distance_m = sim.distance_m;
    row.collisions = sim.collisions;
    row.pose_rms_m = sim.pose_rms();
    row.pose_max_m = sim.max_error_m;
    row.step_mean_ms = step_total.as_secs_f32() * 1000.0 / steps.max(1) as f32;
    row.step_max_ms = step_max.as_secs_f32() * 1000.0;
    row.energy_wh = sim.truth.flight.energy_wh;
    row.soc_end = sim.truth.flight.soc;
    Ok(row)
}

const HEADER: &str = "drone,map,mission,pose,seed,success,end,time_s,distance_m,collisions,\
coverage_60s,coverage_180s,coverage_end,map_occupied_ok,map_free_ok,\
pose_error_rms_m,pose_error_max_m,step_mean_ms,step_max_ms,b_x,b_z,energy_wh,soc_end";

fn csv_line(c: &Case, r: &Result<Row, String>) -> String {
    let opt = |v: Option<f32>| v.map_or(String::new(), |v| format!("{v:.4}"));
    let head = format!("{},{},{},{},{}", c.drone, c.map, c.mission.label(), c.pose.label(), c.seed);
    match r {
        Ok(r) => format!(
            "{head},{},{},{:.2},{:.1},{},{},{},{},{:.4},{:.4},{:.3},{:.3},{:.3},{:.3},{},{},{:.2},{:.3}",
            r.success as u8,
            r.end,
            r.time_s,
            r.distance_m,
            r.collisions,
            opt(r.coverage[0]),
            opt(r.coverage[1]),
            opt(r.coverage[2]),
            r.map_occupied_ok,
            r.map_free_ok,
            r.pose_rms_m,
            r.pose_max_m,
            r.step_mean_ms,
            r.step_max_ms,
            opt(r.goal.map(|g| g.x)),
            opt(r.goal.map(|g| g.y)),
            r.energy_wh,
            r.soc_end,
        ),
        // Impossible case (for example EKF without IMU): a row saying so.
        Err(e) => format!("{head},0,\"error: {}\",,,,,,,,,,,,,,,,,", e.replace('"', "'")),
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let mut cases = Vec::new();
    for d in &args.drones {
        for m in &args.maps {
            for &mission in &args.missions {
                for &pose in &args.poses {
                    for &seed in &args.seeds {
                        cases.push(Case { drone: d.clone(), map: m.clone(), mission, pose, seed });
                    }
                }
            }
        }
    }
    eprintln!("{} simulations on {} threads", cases.len(), args.threads);
    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<Result<Row, String>>>> = cases.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..args.threads.max(1) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(case) = cases.get(k) else { break };
                    let t0 = Instant::now();
                    let r = run(case, &args);
                    let line = csv_line(case, &r);
                    eprintln!("[{}/{}] {:.0} s  {line}", k + 1, cases.len(), t0.elapsed().as_secs_f32());
                    *results[k].lock().unwrap() = Some(r);
                }
            });
        }
    });

    let mut out = String::from(HEADER);
    out.push('\n');
    for (case, r) in cases.iter().zip(&results) {
        let r = r.lock().unwrap().take().expect("every case has run");
        out.push_str(&csv_line(case, &r));
        out.push('\n');
    }
    if let Some(dir) = std::path::Path::new(&args.out).parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir).expect("output directory");
    }
    std::fs::write(&args.out, out).expect("writing the CSV");
    eprintln!("{} in {:.0} s", args.out, started.elapsed().as_secs_f32());
}
