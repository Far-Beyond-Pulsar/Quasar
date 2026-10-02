//! Zero-allocation binaural (headphone) renderer for the scene pipeline.
//!
//! **This is a parametric approximation, not a measured HRTF.** There is no
//! HRTF dataset in the repo, so each ear is modelled with textbook components
//! driven only by the source direction relative to the head:
//!
//! * **ITD** - Woodworth spherical-head model, `itd = (a/c)(theta + sin theta)`
//!   with `theta = asin(lateral)` the lateral angle (so elevation moves the
//!   source along the cone of confusion like a real head), applied as a
//!   fractional delay per ear (4-point Catmull-Rom/Hermite interpolation).
//!   The delay is ramped linearly per sample across each block, so a moving
//!   source gives a smooth (slightly Doppler-like) pitch change, never a click.
//! * **ILD / head shadow** - Brown & Duda one-pole/one-zero filter
//!   `H(s) = (1 + alpha(theta_ear) s/2w0) / (1 + s/2w0)`, `w0 = c/a`,
//!   `alpha(theta) = 1 + a_min/2 + (1 - a_min/2) cos(theta * 180/150 deg)`,
//!   `a_min = 0.1`, with `theta_ear` the angle between the source and that ear's
//!   axis. DC gain is 1; the ipsilateral ear gets up to +6 dB of high-frequency
//!   boost, the contralateral ear a high-frequency cut (about -11 dB).
//!   Discretised with the bilinear transform.
//! * **Front/back** - a high shelf (4 kHz) that cuts rear sources by up to 5 dB
//!   (pinna shadowing), scaled by `(1 - front)/2` so it is continuous around
//!   the head and identical for both ears.
//! * **Elevation** - a peaking "pinna notch" whose centre rises from ~4.5 kHz
//!   (below) through 7.5 kHz (horizon) to ~10.5 kHz (above) and whose depth
//!   grows with `|elevation|`; the far ear gets half the depth.
//!
//! Filter coefficients and delays are computed once per block for the block's
//! target direction and linearly interpolated per sample from the previous
//! block's values (no zipper noise). The state (delay lines, filter memories)
//! is allocated in [`ParametricBinauralRenderer::new`]; `render_add` never
//! allocates. Left/right are exactly mirror-symmetric (`az -> -az` swaps ears).
//!
//! Where a measured HRTF plugs in: implement [`BinauralRenderer`] for a
//! HRIR-pair convolver (e.g. a SOFA-backed partitioned FFT convolver with
//! HRIR interpolation and an onset-aligned ITD) and construct it where the
//! engine currently constructs [`ParametricBinauralRenderer`]
//! (`SpatialAudioEngine::rebuild_scene_render`). Nothing else in the pipeline
//! depends on the concrete type.
//!
//! Conventions match the rest of Quasar: azimuth 0 = ahead (-Z), +azimuth
//! toward +X (the listener's right); elevation +up.

use crate::biquad::BiquadFilter;
use std::f32::consts::{FRAC_PI_2, PI};

/// Renders one mono signal to a binaural pair for a given listener-space direction.
///
/// Implementations must be allocation-free in `render_add`.
pub trait BinauralRenderer: Send {
    /// Add the binaural rendering of `input` (arriving from listener-space
    /// `azimuth`/`elevation`, radians) onto `left` and `right`.
    ///
    /// Processes `min(input, left, right)` samples. The direction is treated as
    /// the target at the end of the block; the renderer ramps from the previous
    /// block's direction.
    fn render_add(&mut self, input: &[f32], azimuth: f32, elevation: f32, left: &mut [f32], right: &mut [f32]);

    /// Clear all internal state (delay lines, filter memories, ramp history).
    fn reset(&mut self);
}

/// Construction parameters of [`ParametricBinauralRenderer`].
#[derive(Clone, Copy, Debug)]
pub struct BinauralConfig {
    pub sample_rate: f32,
    /// Spherical head radius in metres (default 0.0875, the Woodworth/Duda average).
    pub head_radius: f32,
    /// Speed of sound in m/s.
    pub speed_of_sound: f32,
}

impl BinauralConfig {
    /// Default head (radius 0.0875 m, c = 343 m/s) at `sample_rate`.
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate, head_radius: 0.0875, speed_of_sound: 343.0 }
    }
}

/// Woodworth ITD in seconds for a lateral angle (radians, `+` = toward the
/// ear whose lead is wanted, clamped to +-pi/2). Odd in `lateral`.
pub fn woodworth_itd_seconds(lateral: f32, head_radius: f32, speed_of_sound: f32) -> f32 {
    let th = lateral.clamp(-FRAC_PI_2, FRAC_PI_2);
    head_radius / speed_of_sound * (th + th.sin())
}

/// Minimum fractional delay (samples) so the 4-point interpolator never reads "the future".
const BASE_DELAY: f32 = 2.0;
/// Brown-Duda head-shadow constants.
const ALPHA_MIN: f32 = 0.1;
const THETA_MIN: f32 = 150.0 * PI / 180.0;
/// Rear-source high-shelf: corner and maximum cut.
const REAR_SHELF_HZ: f32 = 4000.0;
const REAR_SHELF_DB: f32 = 5.0;
/// Pinna notch: centre frequency at the horizon / swing per unit normalised elevation, Q, depth.
const NOTCH_HZ: f32 = 7500.0;
const NOTCH_SWING_HZ: f32 = 3000.0;
const NOTCH_Q: f32 = 2.5;
const NOTCH_BASE_DB: f32 = 1.5;
const NOTCH_ELEV_DB: f32 = 6.0;

// Flat per-ear parameter vector (so interpolation is one loop).
const P_DELAY: usize = 0; // samples
const P_SHADOW: usize = 1; // b0, b1, a1 (3)
const P_SHELF: usize = 4; // b0 b1 b2 a1 a2 (5)
const P_NOTCH: usize = 9; // b0 b1 b2 a1 a2 (5)
const NUM_PARAMS: usize = 14;
type EarParams = [f32; NUM_PARAMS];

/// Per-ear state: delay ring, head-shadow memory, shelf and notch biquads.
struct EarState {
    buf: Vec<f32>,
    mask: usize,
    /// Index of the newest sample.
    w: usize,
    sx1: f32,
    sy1: f32,
    shelf: BiquadFilter,
    notch: BiquadFilter,
}

impl EarState {
    fn new(min_len: usize) -> Self {
        let len = min_len.next_power_of_two().max(16);
        Self {
            buf: vec![0.0; len],
            mask: len - 1,
            w: 0,
            sx1: 0.0,
            sy1: 0.0,
            shelf: BiquadFilter::new(),
            notch: BiquadFilter::new(),
        }
    }

    fn clear(&mut self) {
        for v in self.buf.iter_mut() {
            *v = 0.0;
        }
        self.w = 0;
        self.sx1 = 0.0;
        self.sy1 = 0.0;
        self.shelf.reset();
        self.notch.reset();
    }

    /// Push `x`, read it back `delay` samples late (4-point Catmull-Rom), then
    /// head shadow -> shelf -> notch with the given (already interpolated) params.
    #[inline]
    fn tick(&mut self, x: f32, p: &EarParams) -> f32 {
        self.w = (self.w + 1) & self.mask;
        self.buf[self.w] = x;

        let max_d = (self.buf.len() - 4) as f32;
        let d = p[P_DELAY].clamp(1.0, max_d);
        let di = d as usize; // >= 1
        let t = d - di as f32;
        let at = |k: usize| self.buf[self.w.wrapping_sub(k) & self.mask];
        // Newest is delay 0; older samples have larger delay, so `t` walks from
        // delay `di` toward `di + 1`.
        let (pm1, p0, p1, p2) = (at(di - 1), at(di), at(di + 1), at(di + 2));
        let m0 = 0.5 * (p1 - pm1);
        let m1 = 0.5 * (p2 - p0);
        let t2 = t * t;
        let t3 = t2 * t;
        let y = (2.0 * t3 - 3.0 * t2 + 1.0) * p0
            + (t3 - 2.0 * t2 + t) * m0
            + (-2.0 * t3 + 3.0 * t2) * p1
            + (t3 - t2) * m1;

        // Brown-Duda one-pole/one-zero.
        let mut s = p[P_SHADOW] * y + p[P_SHADOW + 1] * self.sx1 - p[P_SHADOW + 2] * self.sy1;
        if s.abs() < 1e-24 {
            s = 0.0; // flush denormals
        }
        self.sx1 = y;
        self.sy1 = s;

        self.shelf.set_coefficients(p[P_SHELF], p[P_SHELF + 1], p[P_SHELF + 2], p[P_SHELF + 3], p[P_SHELF + 4]);
        self.notch.set_coefficients(p[P_NOTCH], p[P_NOTCH + 1], p[P_NOTCH + 2], p[P_NOTCH + 3], p[P_NOTCH + 4]);
        let s = self.shelf.process(s);
        self.notch.process(s)
    }
}

/// Parametric (Woodworth ITD + Brown-Duda head shadow + pinna EQ) binaural renderer.
///
/// One instance holds the state for one (listener, scene output) pair.
pub struct ParametricBinauralRenderer {
    cfg: BinauralConfig,
    max_itd_samples: f32,
    /// `[left, right]`.
    ears: [EarState; 2],
    prev: [EarParams; 2],
    valid: bool,
}

impl ParametricBinauralRenderer {
    /// Create a renderer. Allocates (API thread only).
    pub fn new(cfg: BinauralConfig) -> Self {
        let mut cfg = cfg;
        cfg.sample_rate = if cfg.sample_rate.is_finite() && cfg.sample_rate > 8000.0 { cfg.sample_rate } else { 48_000.0 };
        cfg.head_radius = if cfg.head_radius.is_finite() { cfg.head_radius.clamp(0.05, 0.12) } else { 0.0875 };
        cfg.speed_of_sound = if cfg.speed_of_sound.is_finite() && cfg.speed_of_sound > 100.0 { cfg.speed_of_sound } else { 343.0 };
        let max_itd_samples =
            woodworth_itd_seconds(FRAC_PI_2, cfg.head_radius, cfg.speed_of_sound) * cfg.sample_rate;
        let need = (BASE_DELAY + max_itd_samples).ceil() as usize + 8;
        Self {
            cfg,
            max_itd_samples,
            ears: [EarState::new(need), EarState::new(need)],
            prev: [[0.0; NUM_PARAMS]; 2],
            valid: false,
        }
    }

    /// Renderer with default head parameters at `sample_rate`.
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(BinauralConfig::new(sample_rate))
    }

    pub fn config(&self) -> &BinauralConfig {
        &self.cfg
    }

    /// Woodworth ITD in seconds for a source at (`azimuth`, `elevation`);
    /// positive when the right ear hears it first.
    pub fn itd_seconds(&self, azimuth: f32, elevation: f32) -> f32 {
        let x = (azimuth.sin() * elevation.cos()).clamp(-1.0, 1.0);
        woodworth_itd_seconds(x.asin(), self.cfg.head_radius, self.cfg.speed_of_sound)
    }

    /// Parameters of one ear. `x_ear` = cosine of the angle between the source
    /// and this ear's axis (+1 = straight at the ear), `front` = forward
    /// component of the source direction, `el_n` = elevation / 90 degrees.
    fn ear_params(&self, x_ear: f32, front: f32, el_n: f32) -> EarParams {
        let sr = self.cfg.sample_rate;
        let mut p = [0.0_f32; NUM_PARAMS];

        // Delay: this ear leads by `itd` (odd in x_ear); split the ITD evenly.
        let itd = woodworth_itd_seconds(x_ear.asin(), self.cfg.head_radius, self.cfg.speed_of_sound) * sr;
        p[P_DELAY] = BASE_DELAY + 0.5 * (self.max_itd_samples - itd);

        // Head shadow (bilinear transform of the Brown-Duda one-pole/zero).
        let theta = x_ear.acos();
        let alpha = 1.0 + 0.5 * ALPHA_MIN + (1.0 - 0.5 * ALPHA_MIN) * (theta * PI / THETA_MIN).cos();
        // tau * K, with tau = a / (2c), K = 2 fs
        let tk = self.cfg.head_radius * sr / self.cfg.speed_of_sound;
        let inv_a0 = 1.0 / (1.0 + tk);
        p[P_SHADOW] = (1.0 + alpha * tk) * inv_a0;
        p[P_SHADOW + 1] = (1.0 - alpha * tk) * inv_a0;
        p[P_SHADOW + 2] = (1.0 - tk) * inv_a0;

        // Front/back pinna shelf.
        let rear = 0.5 * (1.0 - front);
        let mut bq = BiquadFilter::new();
        bq.set_high_shelf(REAR_SHELF_HZ.min(0.45 * sr), -REAR_SHELF_DB * rear, sr);
        p[P_SHELF..P_SHELF + 5].copy_from_slice(&bq.coefficients());

        // Elevation notch.
        let ipsi = 0.5 + 0.5 * x_ear.max(0.0);
        let depth = -(NOTCH_BASE_DB + NOTCH_ELEV_DB * el_n.abs()) * ipsi;
        let fc = (NOTCH_HZ + NOTCH_SWING_HZ * el_n).clamp(2000.0, 0.45 * sr);
        bq.set_peaking(fc, NOTCH_Q, depth, sr);
        p[P_NOTCH..P_NOTCH + 5].copy_from_slice(&bq.coefficients());
        p
    }

    fn target_params(&self, az: f32, el: f32) -> [EarParams; 2] {
        let (sa, ca) = az.sin_cos();
        let ce = el.cos();
        let x = (sa * ce).clamp(-1.0, 1.0);
        let front = (ca * ce).clamp(-1.0, 1.0);
        let el_n = (el / FRAC_PI_2).clamp(-1.0, 1.0);
        [self.ear_params(-x, front, el_n), self.ear_params(x, front, el_n)]
    }
}

impl BinauralRenderer for ParametricBinauralRenderer {
    fn render_add(&mut self, input: &[f32], azimuth: f32, elevation: f32, left: &mut [f32], right: &mut [f32]) {
        let n = input.len().min(left.len()).min(right.len());
        if n == 0 {
            return;
        }
        let az = if azimuth.is_finite() { azimuth } else { 0.0 };
        let el = if elevation.is_finite() { elevation } else { 0.0 };
        let target = self.target_params(az, el);
        if !self.valid {
            self.prev = target;
            self.valid = true;
        }
        let inv_n = 1.0 / n as f32;
        let mut cur = [[0.0_f32; NUM_PARAMS]; 2];
        for i in 0..n {
            let t = (i + 1) as f32 * inv_n;
            for e in 0..2 {
                for k in 0..NUM_PARAMS {
                    cur[e][k] = self.prev[e][k] + (target[e][k] - self.prev[e][k]) * t;
                }
            }
            let x = input[i];
            left[i] += self.ears[0].tick(x, &cur[0]);
            right[i] += self.ears[1].tick(x, &cur[1]);
        }
        self.prev = target;
    }

    fn reset(&mut self) {
        self.ears[0].clear();
        self.ears[1].clear();
        self.valid = false;
    }
}
