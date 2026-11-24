# drone-rust

An autonomous drone simulator in Rust. It provides:

- a 3D world: procedural, hand-drawn, or a city, on sculpted relief, with wind;
- quadrotor flight dynamics: the drone tilts to accelerate, brake and hold against
  the wind;
- drones described in RON, with their sensors;
- an occupancy map built in flight, and a 3D voxel map;
- autonomous exploration;
- A → B navigation on an unknown map;
- a pose estimated by a Kalman filter, corrected by matching the scans on the map;
- a headless benchmark that compares all of this in CSV.

The viewer uses Bevy 0.20 and egui. The core (world, sensors, mapping, planning,
estimation, simulation) does not depend on Bevy, so it also runs from the command line.

![The drone autonomously exploring a city in the viewer: lidar rays, occupancy map filling in](docs/demo.gif)

## Quick start

Requires Rust 1.98+ (edition 2024).

```bash
cargo run --release -p viewer -- --map maps/downtown.ron
```

Then press `X` to start autonomous exploration. To fly to a goal instead, point at the
ground and press `B` to place it, then press `G` to go there.

For 3D mapping, fly the house map, in its gusty wind:

```bash
cargo run --release -p viewer -- --map maps/house.ron
```

Press `X`, then `V` to hide the real world and see only the voxel map the drone built.
With a 3D lidar the drone flies in 3D: it follows the relief, climbs over the house,
fences and cars, and only goes around what is too tall.

## Viewer

```bash
cargo run -p viewer                                   # procedural world, random seed
cargo run -p viewer -- --seed 42                      # procedural world, fixed seed
cargo run -p viewer -- --map maps/corridor.ron        # hand-drawn map
cargo run -p viewer -- --drone drones/pro.ron         # another drone (default: Mapper)
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
| 3D map | `M` show/hide the voxel map · `V` show/hide the real world |

### Panels

- **Sensors**: latest measurement of each sensor, with noise controls. The panel scrolls when the list is long.
- **Map**: top-down occupancy grid (free / occupied / unknown), the slab altitude, and how well the map matches the terrain.
- **Exploration**: strategy, status, distance travelled, known area, collisions.
- **Mission A -> B**: goal, status, predicted and remaining time, distance travelled, D* Lite statistics.
- **Pose**: pose mode, real error and announced uncertainty, scan matching on/off. In EKF mode, a magenta ghost shows where the drone *believes* it is, with a 2σ circle.
- **Wind** (collapsed, movable): mean speed and direction, gusts, turbulence, presets
  (calm, breeze, storm), the airflow streaks, and the wind the drone feels.
- **3D map & flight** (movable): voxel count, show/hide, clear; 3D flight on/off,
  height above surfaces, tallest obstacle to fly over, and the distance kept from
  walls (applied at the next mission start).
- An inset shows the onboard camera view. The camera is fixed to the airframe, so
  the horizon tilts when the drone does.
- The HUD shows the tilt (pitch, roll), the thrust and the wind felt.

## Maps

| File | Bench name | Content |
|---|---|---|
| `maps/corridor.ron` | `corridor` | Corridor of walls, poles and a rotated block; 80 m |
| `maps/downtown.ron` | `downtown` | Tight orthogonal grid, tall buildings, towers, squares |
| `maps/suburbs.ron` | `suburbs` | Low scattered houses, gardens and parks; grid rotated by 15° |
| `maps/old_town.ron` | `old-town` | Grid at 35°, skewed and offset buildings, gentle terrain |
| `maps/dense_city.ron` | `dense-city` | 160 m, 3.5 m streets: the hardest one |
| `maps/house.ron` | — | A house (open door, upper floor, gable roof, porch) and a large L-shaped building (ground-level passage, penthouse), on hills, in a gusty wind |
| — | `seedN` | Procedural world of seed N (e.g. `seed42`) |

A map is a `WorldSpec` in RON. The frame has Y up, the origin at the center, and units in meters.
The drone takes off from the origin.

```ron
WorldSpec(
    size_m: 80.0,
    cell_m: 1.0,
    terrain: Flat(height: 0.0),            // or Noise(..), or Landscape(..)
    wind: (speed_ms: 4.0, from_deg: 250.0, gust_ms: 3.0, turbulence_ms: 0.8),  // optional
    obstacles: List([
        (shape: Box(size: (0.5, 4.0, 30.0)), x: -4.0, z: -20.0),
        (shape: Cylinder(radius: 0.4, height: 6.0), x: 10.0, z: 5.0),
        (shape: Box(size: (5.0, 8.0, 3.0)), x: -15.0, z: 12.0, yaw_deg: 30.0),
        (shape: Roof(size: (6.0, 2.0, 4.0)), x: -15.0, z: 12.0, yaw_deg: 30.0, elevation_m: 8.0),
    ]),
)
```

- `Landscape` is a noise terrain sculpted by `features`: `Hill` (a negative height
  digs a hollow) and `Pad`, a flat platform for a building.
- `elevation_m` lifts an obstacle above the ground: lintels, floor slabs, roofs,
  canopies, bridges.
- `Roof` is a gable roof, ridge along its length.
- The wind (`from_deg`: where it comes from, 0 = north) grows with height, comes in
  gusts that travel downwind, carries turbulent eddies, and drops in the lee of
  obstacles, where it gets gustier.

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
| `drones/mapper.ron` | 3D mapping (6 m/s), viewer default | 3D lidar 16 × 360 rays 25 m, sonar, IMU, barometer, GPS |

A `DroneSpec` describes:

- the airframe: mass, max speed and acceleration, max yaw rate, radius, and
  optionally its dynamics: thrust-to-weight, max tilt and tilt rate, attitude and
  motor time constants, linear and quadratic drag;
- the onboard camera;
- a list of sensors, each with its mount and rate.

The sensor types are `Lidar2D`, `Lidar3D`, `Sonar` and `Depth` for perception, and
`Imu`, `Baro` and `Gps` for navigation. Noise, dropouts and biases are configurable per sensor.

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
| `--scan-match` | `on` | `on`, `off`: in `ekf` mode, correct the pose by scan matching |

**Success criteria:**
- **Exploration**: finished, with no collision, at least 90 % coverage, and at least
  90 % of the occupied cells correct.
- **A → B**: the *true* position ends within 1 m of B, with no collision.

The CSV has one row per simulation. Besides success, it records time, distance,
collisions, coverage at 60 s, 180 s and at the end, map correctness (occupied and free),
pose error (RMS and max), and compute time per step (mean and max).

## Results

The full matrix has 4 drones × 7 maps × 3 missions × 3 pose modes = 252 runs, with
seed 1 and the default limits. These results predate the flight dynamics and the
wind: rerun the bench to update them. The raw data is in [`bench/results.csv`](bench/results.csv).

### Success by mission and pose mode

| Mission | `truth` | `ekf-gps` | `ekf` |
|---|---|---|---|
| A → B | 24/28 | 22/28 | 10/28 |
| Exploration, nearest frontier | 10/28 | 7/28 | 0/28 |
| Exploration, information gain | 11/28 | 6/28 | 0/28 |

### Perfect pose (`truth`), by drone

| Drone | A → B | Exploration (both strategies) | Collisions |
|---|---|---|---|
| Scout | 7/7 | 11/14 | 0 |
| Pro | 6/7 | 8/14 | 0 |
| Heavy | 6/7 | 2/14 | 0 |
| Cheap | 5/7 | 0/14 | 38 |

### Pose error without GPS (`ekf`), by drone

Median RMS error over the 21 runs of each drone. Scan matching needs a lidar or a
depth camera; the Cheap only has sonars.

| Drone | Without scan matching | With scan matching | A → B |
|---|---|---|---|
| Scout | 34 m | 2.9 m | 5/7 |
| Pro | 35 m | 1.1 m | 5/7 |
| Heavy | 34 m | 17 m | 0/7 |
| Cheap | 67 m | 63 m | 0/7 |

### Takeaways

- **With a perfect pose, the lidar drones fly cleanly.**
  - Scout and Pro have zero collisions on every map, including the four cities.
  - Their failed explorations are mostly timeouts in the largest maps: the map is
    correct and 93–98 % covered, but 1200 s is not enough to finish.
- **The Heavy never collides, but it is slow in cities.**
  - It is slow (5 m/s) and wide (0.8 m plus margins), and it only sees through a
    forward depth camera and a downward sonar.
  - At 1200 s it has mapped about half of downtown and old-town.
- **Sonars are not enough.**
  - The Cheap's four sonars miss oblique walls and see little.
  - Outside the corridor, its coverage stays under 30 %, and only 55–83 % of its
    occupied cells are correct.
  - It collides on the hilly procedural maps.
- **The GPS keeps the estimated pose usable.**
  - With `ekf-gps`, the median RMS pose error is 0.34 m (0.57 m at most).
  - A → B does almost as well as with the perfect pose.
  - Exploration loses some runs: an error of a few tens of centimeters is enough to
    blur the walls of a narrow street.
- **Without GPS, scan matching holds the lidar drones to a few meters.**
  - It brings the median error of the Scout and the Pro from about 35 m down to
    1–3 m, and A → B from 7 to 10 successes.
  - The Heavy's depth camera only sees 87° ahead: few matches, so less gain.
  - No exploration succeeds yet. Open areas, where nothing is in range, still let
    the IMU drift, and an error of 1–3 m is enough to fail the 90 % map criterion.
  - The drone sometimes "reaches" B on a map built in the wrong place
    (`B believed reached (wrong pose)`).
- **Compute is cheap.** One step (estimation + map + decision) takes 0.34 ms in the
  median and under 2 ms on average for every run, while all cores are busy.

## Architecture

The project is a Cargo workspace with one crate per layer:

| Crate | Role |
|---|---|
| `world` | Terrain and relief, obstacles (boxes, cylinders, roofs), raycasts and contacts (8 m grid spatial index), procedural cities, wind field, quadrotor flight dynamics |
| `drone_spec` | RON description of a drone, with validation |
| `sensors` | 2D and 3D lidar, sonar, depth camera; noisy IMU, barometer and GPS |
| `mapping` | 2D occupancy grid: ray casting, per-cell hit/miss evidence, ambiguous cells; scan-to-map matching (point-to-plane ICP); sparse 3D voxel map (3D DDA) |
| `planning` | Frontiers, exploration (nearest / information gain), incremental D* Lite, timed trajectory, safe-speed control |
| `estimation` | 11-state EKF (position, velocity, yaw, accelerometer and gyro biases) on IMU + barometer + GPS + scan matching |
| `sim` | Headless simulation (drone + sensors + estimation + map), shared by tests and the bench |
| `bench` | Benchmark matrix → CSV |
| `viewer` | Bevy application |

### Key ideas

- **Quadrotor dynamics.** The drone only pushes along its thrust axis.
  - The pilot (or the planner) commands a velocity. A PI loop turns the error into
    a wanted acceleration, plus a feed-forward of the still-air drag. Its integral
    learns the steady push of the wind, so the drone ends up leaning into it.
  - The thrust axis turns towards the wanted one with a lag and a bounded tilt rate,
    and the motors follow with their own lag: braking first means swinging the tilt
    to the other side.
  - Drag acts on the airspeed (velocity relative to the wind). A gale stronger than
    the tilt limit allows wins.
  - Obstacles push the drone back out and absorb most of the impact.
  - The 2D and 3D lidars sit on a stabilized mount (heading only), so the slab map
    stays horizontal; the onboard camera tilts with the airframe.
- **Safe speed accounts for the response time.** The stopping distance includes the
  0.2 s the airframe needs before it decelerates, and the look-ahead covers a full stop
  from top speed.
- **3D map.** Every lidar or depth ray frees the voxels it crosses and marks the one
  it hits, in a sparse hash map: facades, roofs, overhangs, the relief.
- **3D flight** (viewer default for a drone with a 3D lidar).
  - The voxel map is projected onto the 2D planning grid: only obstacles taller
    than `max_overfly_m` (12 m) above their ground block; the 2D planners are
    unchanged.
  - The target altitude is the highest surface under and ahead of the drone (ground,
    roofs, canopies) plus a clearance (2.5 m): it follows the relief and climbs over
    obstacles, then comes back down slowly.
  - Whatever still stands at the drone's height caps its horizontal speed: it
    waits to have climbed before going on.
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
- **Scan matching.** Each lidar or depth scan is aligned on the map before being
  added to it, and the result corrects the EKF as a pose measurement (x, z, yaw).
  - Point-to-plane ICP: a scan endpoint is pulled across the wall it hits, never
    along it, so a corridor constrains the pose across, not along. Poles and
    corners give point-to-point constraints.
  - The map remembers where the hits fell inside each cell. Matching on cell
    centers instead biases each match by up to half a cell, and the bias feeds back
    into the map.
  - Only hits within 70 % of the sensor range are used: the map knows walls up to
    the range of the previous scans, and the far hits would pull the pose backwards.
  - The Hessian of the fit gives the measurement covariance. After a stretch with
    nothing to match, the search widens to 3σ of the pose uncertainty to lock on again.

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
- scan matching: a pose offset is recovered within its covariance, and a scan along a
  corridor does not slide;
- the indexed raycast against a brute-force one;
- the shipped maps all build;
- flight dynamics: nose down to accelerate, nose up to brake, holding still in a
  steady wind by leaning into it, drifting in a gale, bouncing off a wall;
- a 3D lidar circling a house maps its walls and roof, with no voxel in the air;
- the Mapper flies around the house, in its gusty wind, without touching anything;
- 3D flight: over a low wall, around a 25 m one, along a hill, over the house roof,
  never closer than 1.5 m to the ground.

## Known limitations

- Without GPS, the pose only holds where scans see structure: in open areas or
  along featureless corridors, the IMU drift takes over.
- The Cheap drone's sonars miss walls seen at an angle.
- 3D flight goes over obstacles, never under them (porch, passage): the projection
  keeps only the top of each column. Without a 3D lidar, planning stays on the 2D slab.
- A facade seen from below looks short: the drone climbs along it until it sees it
  is too tall, then goes around.
- The wind's shelter is a single upwind raycast: no flow around buildings.
- Scan matching is scan-to-map without loop closure: an error made while
  exploring is written into the map and never undone.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this project, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
