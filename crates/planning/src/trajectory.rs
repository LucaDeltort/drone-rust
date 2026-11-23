//! Timed trajectory: a smoothed path (polyline) and a speed profile that
//! respects the max speed, the max acceleration and a slow pass through corners.
//!
//! The profile is the classic "forward / backward" one: accelerate as hard as possible from the
//! current speed, brake as hard as possible before each corner and before the arrival.

use glam::Vec2;

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub points: Vec<Vec2>,
    /// Arc length of each vertex, in m.
    pub s: Vec<f32>,
    /// Speed through each vertex, in m/s.
    pub v: Vec<f32>,
    vmax: f32,
    accel: f32,
    /// Samples (time, arc length) to travel the trajectory over time.
    timeline: Vec<(f32, f32)>,
}

/// Sampling step of the travel time, in m.
const TIME_STEP_M: f32 = 0.1;

impl Trajectory {
    /// `v0`: current speed along the first segment. `corner_error_m`: deviation
    /// tolerated in a corner, which sets its cornering speed.
    pub fn plan(points: Vec<Vec2>, v0: f32, vmax: f32, accel: f32, corner_error_m: f32) -> Self {
        assert!(points.len() >= 2);
        let n = points.len();
        let mut s = vec![0.0; n];
        for k in 1..n {
            s[k] = s[k - 1] + points[k - 1].distance(points[k]);
        }
        // Corners: changing direction by an angle θ at speed v pushes the drone off
        // by about v² · sin(θ/2) / (2a); this offset is bounded by `corner_error_m`.
        let mut v = vec![vmax; n];
        for k in 1..n - 1 {
            let a = (points[k] - points[k - 1]).normalize_or_zero();
            let b = (points[k + 1] - points[k]).normalize_or_zero();
            let half = (a.dot(b).clamp(-1.0, 1.0).acos() / 2.0).sin();
            if half > 1e-3 {
                v[k] = vmax.min((2.0 * accel * corner_error_m / half).sqrt());
            }
        }
        v[0] = v0.clamp(0.0, vmax);
        v[n - 1] = 0.0;
        for k in 0..n - 1 {
            let l = s[k + 1] - s[k];
            v[k + 1] = v[k + 1].min((v[k] * v[k] + 2.0 * accel * l).sqrt());
        }
        for k in (0..n - 1).rev() {
            let l = s[k + 1] - s[k];
            v[k] = v[k].min((v[k + 1] * v[k + 1] + 2.0 * accel * l).sqrt());
        }
        let mut traj = Self { points, s, v, vmax, accel, timeline: Vec::new() };
        traj.timeline = traj.integrate_time();
        traj
    }

    pub fn length(&self) -> f32 {
        *self.s.last().unwrap()
    }

    /// Predicted travel time, in s.
    pub fn duration(&self) -> f32 {
        self.timeline.last().map_or(0.0, |&(t, _)| t)
    }

    /// Predicted speed at arc length `s`.
    pub fn speed_at(&self, s: f32) -> f32 {
        let s = s.clamp(0.0, self.length());
        let k = self.segment(s);
        let (a, l) = (self.accel, s - self.s[k]);
        let to_end = self.s[k + 1] - s;
        self.vmax
            .min((self.v[k] * self.v[k] + 2.0 * a * l).sqrt())
            .min((self.v[k + 1] * self.v[k + 1] + 2.0 * a * to_end).sqrt())
    }

    /// Point at arc length `s`.
    pub fn point_at(&self, s: f32) -> Vec2 {
        let s = s.clamp(0.0, self.length());
        let k = self.segment(s);
        let l = self.s[k + 1] - self.s[k];
        let t = if l > 0.0 { (s - self.s[k]) / l } else { 0.0 };
        self.points[k].lerp(self.points[k + 1], t)
    }

    /// Predicted position at time `t` (s) after the start.
    pub fn sample(&self, t: f32) -> Vec2 {
        let i = self.timeline.partition_point(|&(ti, _)| ti < t);
        let s = match i {
            0 => 0.0,
            i if i >= self.timeline.len() => self.length(),
            i => {
                let ((t0, s0), (t1, s1)) = (self.timeline[i - 1], self.timeline[i]);
                s0 + (s1 - s0) * (t - t0) / (t1 - t0).max(1e-6)
            }
        };
        self.point_at(s)
    }

    /// Segment containing arc length `s`.
    fn segment(&self, s: f32) -> usize {
        self.s.partition_point(|&x| x <= s).clamp(1, self.s.len() - 1) - 1
    }

    fn integrate_time(&self) -> Vec<(f32, f32)> {
        let total = self.length();
        let mut out = vec![(0.0, 0.0)];
        let (mut t, mut s) = (0.0, 0.0);
        while s < total {
            let ds = TIME_STEP_M.min(total - s);
            // Constant acceleration over the step: exact time = 2·ds / (v0 + v1).
            let (v0, v1) = (self.speed_at(s), self.speed_at(s + ds));
            t += 2.0 * ds / (v0 + v1).max(1e-3);
            s += ds;
            out.push((t, s));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_from_rest_is_a_trapezoid() {
        // 40 m, 8 m/s, 4 m/s²: 8 m to accelerate, 8 m to brake, 24 m at 8 m/s.
        let t = Trajectory::plan(vec![Vec2::ZERO, Vec2::new(40.0, 0.0)], 0.0, 8.0, 4.0, 0.3);
        assert!((t.speed_at(20.0) - 8.0).abs() < 1e-4);
        assert!((t.speed_at(2.0) - 4.0).abs() < 1e-3);
        assert!(t.speed_at(40.0) < 1e-3);
        // 2 s + 3 s + 2 s.
        assert!((t.duration() - 7.0).abs() < 0.1, "{}", t.duration());
        assert!(t.sample(3.5).distance(Vec2::new(20.0, 0.0)) < 0.2);
    }

    #[test]
    fn slows_down_in_corners_and_respects_acceleration() {
        let pts = vec![Vec2::ZERO, Vec2::new(30.0, 0.0), Vec2::new(30.0, 30.0)];
        let t = Trajectory::plan(pts, 0.0, 8.0, 4.0, 0.3);
        // 90° corner: v² · sin 45° / (2a) ≤ 0.3.
        let vc = t.v[1];
        assert!(vc < 8.0 && vc * vc * std::f32::consts::FRAC_1_SQRT_2 / 8.0 <= 0.3 + 1e-4);
        // Acceleration bounded everywhere.
        for w in t.timeline.windows(2) {
            let (a, b) = (t.speed_at(w[0].1), t.speed_at(w[1].1));
            assert!((b - a).abs() / (w[1].0 - w[0].0) <= 4.0 * 1.05 + 0.1);
        }
    }

    #[test]
    fn short_path_never_reaches_top_speed() {
        let t = Trajectory::plan(vec![Vec2::ZERO, Vec2::new(2.0, 0.0)], 0.0, 8.0, 4.0, 0.3);
        assert!(t.speed_at(1.0) <= (2.0f32 * 4.0).sqrt() + 1e-4);
    }
}
