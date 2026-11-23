# drone-rust

An autonomous drone simulator in Rust. It provides:

- a 3D world: procedural, hand-drawn, or a city;
- drones described in RON, with their sensors;
- an occupancy map built in flight;
- autonomous exploration;
- A → B navigation on an unknown map;
- a pose estimated by a Kalman filter;
- a headless benchmark that compares all of this in CSV.

The viewer uses Bevy 0.20 and egui. The core (world, sensors, mapping, planning,
estimation, simulation) does not depend on Bevy, so it also runs from the command line.

## Quick start

Requires Rust 1.98+ (edition 2024).

```bash
cargo run --release -p viewer -- --map maps/downtown.ron
```

Then press `X` to start autonomous exploration. To fly to a goal instead, point at the
ground and press `B` to place it, then press `G` to go there.

## Viewer

```bash
cargo run -p viewer                                   # procedural world, random seed
cargo run -p viewer -- --seed 42                      # procedural world, fixed seed
cargo run -p viewer -- --map maps/corridor.ron        # hand-drawn map
cargo run -p viewer -- --drone drones/pro.ron         # another drone (default: Scout)
cargo run -p viewer -- --pose ekf                     # estimated pose: truth, ekf-gps, ekf
```

You can combine options. Bevy is slow in debug builds, so use `--release` in cities.

### Keys

| Context | Keys |
|---|---|
| General | `Tab` pilot ↔ free camera · `[` `]` onboard camera FOV · `N`/`P` next/previous seed (seed mode) · `R` reload the map (file mode) |
| Manual piloting | `W`/`S` forward/backward · `A`/`D` left/right · `Space`/`Shift` up/down · `Q`/`E` yaw |
| Free camera | right click + mouse, `WASD`, `Q`/`E`, `Shift`, wheel |
| Missions | `X` exploration · `B` place B under the cursor · `G` go to B (start/stop) · `O` drop a wall in front of the drone |

### Panels

- **Sensors**: latest measurement of each sensor, with noise controls. The panel scrolls when the list is long.
- **Map**: top-down occupancy grid (free / occupied / unknown), the slab altitude, and how well the map matches the terrain.
- **Exploration**: strategy, status, distance travelled, known area, collisions.
- **Mission A -> B**: goal, status, predicted and remaining time, distance travelled, D* Lite statistics.
- **Pose**: pose mode, real error and announced uncertainty. In EKF mode, a magenta ghost shows where the drone *believes* it is, with a 2σ circle.
- An inset shows the onboard camera view.

## Maps

| File | Bench name | Content |
|---|---|---|
| `maps/corridor.ron` | `corridor` | Corridor of walls, poles and a rotated block; 80 m |
| `maps/downtown.ron` | `downtown` | Tight orthogonal grid, tall buildings, towers, squares |
| `maps/suburbs.ron` | `suburbs` | Low scattered houses, gardens and parks; grid rotated by 15° |
| `maps/old_town.ron` | `old-town` | Grid at 35°, skewed and offset buildings, gentle terrain |
| `maps/dense_city.ron` | `dense-city` | 160 m, 3.5 m streets: the hardest one |
| — | `seedN` | Procedural world of seed N (e.g. `seed42`) |

A map is a `WorldSpec` in RON. The frame has Y up, the origin at the center, and units in meters.
The drone takes off from the origin.

```ron
WorldSpec(
    size_m: 80.0,
    cell_m: 1.0,
    terrain: Flat(height: 0.0),            // or Noise(seed: .., amplitude_m: .., ..)
    obstacles: List([
        (shape: Box(size: (0.5, 4.0, 30.0)), x: -4.0, z: -20.0),
        (shape: Cylinder(radius: 0.4, height: 6.0), x: 10.0, z: 5.0),
        (shape: Box(size: (5.0, 8.0, 3.0)), x: -15.0, z: 12.0, yaw_deg: 30.0),
    ]),
)
```

`obstacles` also accepts `None`, `Random(..)` and `City(..)`. A city is generated from a few
parameters:

- block and street size, lots per block and alley width;
- building heights and density;
- park and tower probability;
- grid rotation and old-town disorder (`jitter_deg`, `jitter_m`);
- a cleared square at the start.

See the commented `maps/*.ron` files and `crates/world/src/city.rs`.

## Drones

| File | Profile | Sensors |
|---|---|---|
| `drones/scout.ron` | light, fast (8 m/s) | 2D lidar 360 rays 12 m, forward sonar, IMU, barometer |
| `drones/heavy.ron` | heavy, slow (5 m/s), 0.4 m radius | depth camera, downward sonar, IMU, barometer |
| `drones/cheap.ron` | low-end (4 m/s) | 4 sonars, low-grade IMU, barometer |
| `drones/pro.ron` | high-end (8 m/s) | 2D lidar 720 rays 20 m, depth camera, sonar, good IMU, barometer, GPS |

A `DroneSpec` describes:

- the airframe: mass, max speed and acceleration, max yaw rate, radius;
- the onboard camera;
- a list of sensors, each with its mount and rate.

The sensor types are `Lidar2D`, `Sonar` and `Depth` for perception, and `Imu`, `Baro`
and `Gps` for navigation. Noise, dropouts and biases are configurable per sensor.

## Benchmark

The benchmark is a headless binary. It runs in parallel on all cores and is
deterministic (fixed seed and fixed step).

```bash
cargo run --release -p bench
```

```bash
cargo run --release -p bench -- --drones scout,pro --maps corridor,downtown --missions explore-nearest,a-b --poses truth,ekf-gps --seeds 1,2 --out bench/results.csv
```

| Option | Default | Values |
|---|---|---|
| `--drones` | all | `scout`, `heavy`, `cheap`, `pro` |
| `--maps` | all | names from the map table, `seed42`, `seed7`, any `seedN`, or a `.ron` path |
| `--missions` | all | `explore-nearest`, `explore-infogain`, `a-b` |
| `--poses` | all | `truth`, `ekf-gps`, `ekf` |
| `--seeds` | `1` | list of seeds |
| `--explore-limit` | `1200` | max duration of an exploration, s |
| `--a-b-limit` | `120` | max duration of an A → B, s |
| `--threads` | number of cores | |
| `--out` | `bench/results.csv` | |

**Success criteria:**
- **Exploration**: finished, with no collision, at least 90 % coverage, and at least
  90 % of the occupied cells correct.
- **A → B**: the *true* position ends within 1 m of B, with no collision.

The CSV has one row per simulation. Besides success, it records time, distance,
collisions, coverage at 60 s, 180 s and at the end, map correctness (occupied and free),
pose error (RMS and max), and compute time per step (mean and max).

## Results

The full matrix has 4 drones × 7 maps × 3 missions × 3 pose modes = 252 runs, with
seed 1 and the default limits. The raw data is in [`bench/results.csv`](bench/results.csv).

### Success by mission and pose mode

| Mission | `truth` | `ekf-gps` | `ekf` |
|---|---|---|---|
| A → B | 22/28 | 23/28 | 7/28 |
| Exploration, nearest frontier | 11/28 | 6/28 | 0/28 |
| Exploration, information gain | 12/28 | 9/28 | 0/28 |

### Perfect pose (`truth`), by drone

| Drone | A → B | Exploration (both strategies) | Collisions |
|---|---|---|---|
| Scout | 7/7 | 11/14 | 0 |
| Pro | 6/7 | 10/14 | 0 |
| Heavy | 6/7 | 2/14 | 0 |
| Cheap | 3/7 | 0/14 | 30 |

### Takeaways

- **With a perfect pose, the lidar drones fly cleanly.**
  - Scout and Pro have zero collisions on every map, including the four cities.
  - Their failed explorations are timeouts in the largest cities: the map is correct,
    but 1200 s is not enough to finish.
- **The Heavy never collides, but it is slow in cities.**
  - It is slow (5 m/s) and wide (0.8 m plus margins), and it only sees through a
    forward depth camera and a downward sonar.
  - At 1200 s it has mapped 50–70 % of downtown, old-town and dense-city.
- **Sonars are not enough.**
  - The Cheap's four sonars miss oblique walls and see little.
  - Outside the corridor, its coverage stays at 15–40 %, and only 46–89 % of its
    occupied cells are correct.
  - It collides, mostly on the hilly procedural maps.
- **The GPS keeps the estimated pose usable.**
  - With `ekf-gps`, the median RMS pose error is 0.34 m (0.58 m at most).
  - A → B does as well as with the perfect pose.
  - Exploration loses some runs: an error of a few tens of centimeters is enough to
    blur the walls of a narrow street.
- **Without GPS, the IMU drift ruins long missions.**
  - The median RMS error is about 42 m after a 1200 s exploration.
  - No exploration succeeds, and A → B only works over short distances.
  - The drone sometimes "reaches" B on a map built in the wrong place
    (`B believed reached (wrong pose)`).
- **Compute is cheap.** One step (estimation + map + decision) takes 0.38 ms in the
  median and under 2 ms on average for every run, while all cores are busy.

## Architecture

The project is a Cargo workspace with one crate per layer:

| Crate | Role |
|---|---|
| `world` | Terrain, obstacles, raycasts and collisions (8 m grid spatial index), procedural cities, drone kinematics |
| `drone_spec` | RON description of a drone, with validation |
| `sensors` | 2D lidar, sonar, depth camera; noisy IMU, barometer and GPS |
| `mapping` | 2D occupancy grid: ray casting, per-cell hit/miss evidence, ambiguous cells |
| `planning` | Frontiers, exploration (nearest / information gain), incremental D* Lite, timed trajectory, safe-speed control |
| `estimation` | 11-state EKF (position, velocity, yaw, accelerometer and gyro biases) on IMU + barometer + GPS |
| `sim` | Headless simulation (drone + sensors + estimation + map), shared by tests and the bench |
| `bench` | Benchmark matrix → CSV |
| `viewer` | Bevy application |

### Key ideas

- **Optimistic planning.** Unknown cells are assumed free.
  - When the map reveals an obstacle, D* Lite repairs the path instead of starting
    over.
  - The configuration space is inflated by the drone's radius plus a margin, and the
    inflation is updated incrementally.
- **Timed trajectory.** The speed profile is computed forward and backward under an
  acceleration limit. In corners, the speed drops according to the allowed deviation.
- **Safe speed.** At every step, the speed towards each nearby obstacle is capped so
  that the drone can stop before reaching it. The margin is wider for occupied cells
  than for unknown ones.
- **Exploration.**
  - Frontiers lie between free and unknown cells.
  - The drone does a 360° scan before declaring the map finished.
  - Ambiguous cells are not frontiers. These are cells with contradictory evidence,
    usually seen at grazing angles.
- **Noisy pose.** In `ekf` and `ekf-gps` modes, the map is built in the *estimated*
  pose, so EKF drift shows up directly as a wrong map.

## Tests

```bash
cargo test --workspace --release
```

```bash
cargo clippy --workspace --all-targets
```

There are about a hundred tests, for example:

- D* Lite against Dijkstra, and after an incremental repair;
- full A → B missions, including a wall that appears in flight and an enclosed goal;
- the EKF: a perfect IMU gives the exact pose, the GPS keeps the error low, and the
  biases are recovered;
- the indexed raycast against a brute-force one;
- the shipped maps all build.

## Known limitations

- Without GPS, IMU drift corrupts the map on long missions.
- The Cheap drone's sonars miss walls seen at an angle.
- The occupancy map is 2D, at flight altitude: the drone never flies under or over
  obstacles.
- There is no SLAM: the map never corrects the pose.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this project, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
