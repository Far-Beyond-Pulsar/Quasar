use quasar_core::bands::Band8;
use quasar_core::param_exchange::{EarlyReflectionCoeffs, SpatialCoefficients};
use std::f32::consts::PI;

/// Maximum number of early reflections the crossfader can hold without
/// allocating. Targets with more reflections are truncated.
pub const MAX_CROSSFADE_REFLECTIONS: usize = 64;

/// Manages smooth transitions between spatial coefficient sets.
///
/// The compute thread publishes new params; the audio thread reads them
/// and crossfades from the in-flight value at retarget time to the target
/// over a fixed window.
///
/// The blend is applied to PARAMETERS (gains, delays, angles), not to audio
/// signals, so it is a plain linear interpolation `from + (to - from) * t`
/// with `t = frame_counter / fade_frames`. An equal-power (cos/sin) law is
/// only correct for uncorrelated signals; applied to parameters its weights
/// sum to up to sqrt(2) and a constant parameter would overshoot. The type
/// name is kept for API compatibility.
///
/// Azimuth is interpolated along the shortest arc across the ±π wrap.
///
/// Early reflections are matched between the start and target sets by nearest
/// `delay_samples` (greedy). Matched pairs interpolate every field; reflections
/// that only exist in the target fade in from zero gain, and ones that vanish
/// fade out to zero gain (delay/angles held). While a fade is running
/// `current.early_reflections` therefore holds the union of both sets; once it
/// completes it is exactly the target's list.
///
/// No allocation happens after construction (all vectors reserve
/// [`MAX_CROSSFADE_REFLECTIONS`] up front).
pub struct EqualPowerCrossfader {
    current: SpatialCoefficients,
    target: SpatialCoefficients,
    /// Snapshot of `current` when the fade started. Its `early_reflections`
    /// is aligned index-for-index with `ref_to`.
    from: SpatialCoefficients,
    /// Per-reflection end values, aligned with `from.early_reflections`.
    ref_to: Vec<EarlyReflectionCoeffs>,
    fade_frames: u32,
    frame_counter: u32,
}

/// Shortest signed angular difference `b - a`, wrapped to [-π, π].
#[inline]
fn wrap_pi(mut x: f32) -> f32 {
    if x > PI || x < -PI {
        x = (x + PI).rem_euclid(2.0 * PI) - PI;
    }
    x
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Interpolate an angle along the shortest arc; result stays in [-π, π].
#[inline]
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let r = a + wrap_pi(b - a) * t;
    wrap_pi(r)
}

#[inline]
fn lerp_band(a: &Band8, b: &Band8, t: f32, out: &mut Band8) {
    for i in 0..8 {
        out.0[i] = lerp(a.0[i], b.0[i], t);
    }
}

/// Copy `src` into `dst` without allocating (within reserved capacity);
/// truncates to [`MAX_CROSSFADE_REFLECTIONS`].
fn copy_reflections(dst: &mut Vec<EarlyReflectionCoeffs>, src: &[EarlyReflectionCoeffs]) {
    dst.clear();
    let n = src.len().min(MAX_CROSSFADE_REFLECTIONS);
    dst.extend_from_slice(&src[..n]);
}

/// Copy every field of `src` into `dst` in place (no allocation).
fn copy_coeffs(dst: &mut SpatialCoefficients, src: &SpatialCoefficients) {
    dst.source_id = src.source_id;
    dst.direct_gain = src.direct_gain;
    dst.direct_delay_samples = src.direct_delay_samples;
    dst.direct_azimuth = src.direct_azimuth;
    dst.direct_elevation = src.direct_elevation;
    dst.late_t60 = src.late_t60;
    dst.late_gain_db = src.late_gain_db;
    dst.version = src.version;
    copy_reflections(&mut dst.early_reflections, &src.early_reflections);
}

/// Clone `src` into a value whose reflection vector has full reserved capacity.
fn with_capacity(src: &SpatialCoefficients) -> SpatialCoefficients {
    let mut c = SpatialCoefficients {
        source_id: src.source_id,
        direct_gain: src.direct_gain,
        direct_delay_samples: src.direct_delay_samples,
        direct_azimuth: src.direct_azimuth,
        direct_elevation: src.direct_elevation,
        early_reflections: Vec::with_capacity(MAX_CROSSFADE_REFLECTIONS),
        late_t60: src.late_t60,
        late_gain_db: src.late_gain_db,
        version: src.version,
    };
    copy_reflections(&mut c.early_reflections, &src.early_reflections);
    c
}

impl EqualPowerCrossfader {
    /// Create a new crossfader.
    ///
    /// `fade_ms`: fade duration in milliseconds (typically 10-20ms).
    /// `sample_rate`: audio sample rate in Hz.
    /// `initial`: starting spatial coefficients.
    pub fn new(fade_ms: f32, sample_rate: f32, initial: SpatialCoefficients) -> Self {
        let fade_frames = ((fade_ms / 1000.0) * sample_rate).round() as u32;
        let mut ref_to = Vec::with_capacity(MAX_CROSSFADE_REFLECTIONS);
        copy_reflections(&mut ref_to, &initial.early_reflections);
        Self {
            current: with_capacity(&initial),
            target: with_capacity(&initial),
            from: with_capacity(&initial),
            ref_to,
            fade_frames: fade_frames.max(1),
            frame_counter: fade_frames.max(1),
        }
    }

    /// Set a new target. If the source_id differs, snap immediately.
    /// Otherwise snapshots the in-flight `current` value as the fade start and
    /// restarts the fade, so retargeting mid-fade is continuous.
    ///
    /// Called from the audio thread after picking up new triple-buffer data.
    /// Copies into preallocated storage; never allocates.
    pub fn set_target(&mut self, target: &SpatialCoefficients) {
        copy_coeffs(&mut self.target, target);
        if self.current.source_id != target.source_id {
            // Source identity changed — instant switch to avoid stale panning
            self.snap_internal();
            return;
        }

        // Fade start = whatever `current` is right now (possibly mid-fade).
        let c = &self.current;
        self.from.source_id = c.source_id;
        self.from.direct_gain = c.direct_gain;
        self.from.direct_delay_samples = c.direct_delay_samples;
        self.from.direct_azimuth = c.direct_azimuth;
        self.from.direct_elevation = c.direct_elevation;
        self.from.late_t60 = c.late_t60;
        self.from.late_gain_db = c.late_gain_db;

        self.align_reflections();
        self.frame_counter = 0;
    }

    /// Build `from.early_reflections` / `ref_to` as index-aligned lists, and set
    /// `current.early_reflections` to the fade-start list.
    ///
    /// Greedy matching: each target reflection takes the nearest unused current
    /// reflection by `delay_samples`. Unmatched target entries start at zero gain
    /// (fade in); unmatched current entries end at zero gain (fade out). Uses a
    /// fixed stack mask, no allocation.
    fn align_reflections(&mut self) {
        let mut used = [false; MAX_CROSSFADE_REFLECTIONS];
        let src = &self.current.early_reflections;
        let tgt = &self.target.early_reflections;

        self.from.early_reflections.clear();
        self.ref_to.clear();

        for t in tgt.iter() {
            let mut best: Option<usize> = None;
            let mut best_d = f32::INFINITY;
            for (i, s) in src.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let d = (s.delay_samples - t.delay_samples).abs();
                if d < best_d {
                    best_d = d;
                    best = Some(i);
                }
            }
            match best {
                Some(i) => {
                    used[i] = true;
                    self.from.early_reflections.push(src[i].clone());
                }
                None => {
                    // New reflection: fade in from silence, geometry held.
                    let mut z = t.clone();
                    z.gain = Band8::zeros();
                    self.from.early_reflections.push(z);
                }
            }
            self.ref_to.push(t.clone());
        }
        // Vanishing reflections: fade out to silence, geometry held.
        for (i, s) in src.iter().enumerate() {
            if used[i] || self.from.early_reflections.len() >= MAX_CROSSFADE_REFLECTIONS {
                continue;
            }
            let mut z = s.clone();
            z.gain = Band8::zeros();
            self.from.early_reflections.push(s.clone());
            self.ref_to.push(z);
        }

        // `current` starts the fade at `from` (identical values, union length).
        let (cur, from) = (&mut self.current.early_reflections, &self.from.early_reflections);
        cur.clear();
        cur.extend_from_slice(from);
    }

    /// Instantly make `current` and `from` equal to `target`.
    fn snap_internal(&mut self) {
        copy_coeffs(&mut self.current, &self.target);
        copy_coeffs(&mut self.from, &self.target);
        copy_reflections(&mut self.ref_to, &self.target.early_reflections);
        self.frame_counter = self.fade_frames; // already at target
    }

    /// Returns true if the crossfade is complete (target reached).
    pub fn is_complete(&self) -> bool {
        self.frame_counter >= self.fade_frames
    }

    /// Returns the current blend factor t ∈ [0,1].
    pub fn blend_factor(&self) -> f32 {
        if self.fade_frames == 0 {
            return 1.0;
        }
        (self.frame_counter as f32 / self.fade_frames as f32).min(1.0)
    }

    /// Get the current active coefficients (blended from→target).
    pub fn current_coefficients(&self) -> &SpatialCoefficients {
        &self.current
    }

    /// Get the target coefficients.
    pub fn target_coefficients(&self) -> &SpatialCoefficients {
        &self.target
    }

    /// Advance the crossfade by one audio block.
    ///
    /// `block_size` is the number of frames in the block being processed. The
    /// frame counter advances by `block_size` (clamped to `fade_frames`) so a
    /// `fade_ms` fade converges in ~`fade_ms` of real time regardless of block
    /// size — a 15 ms fade takes ~2 blocks at 512 frames/block, not ~3.8 s.
    /// `current` is recomputed from the fixed `from`/`target` pair (never
    /// compounded in place). Returns the current blend factor.
    pub fn advance(&mut self, block_size: usize) -> f32 {
        if self.frame_counter >= self.fade_frames {
            return 1.0;
        }
        self.frame_counter = self
            .frame_counter
            .saturating_add(block_size as u32)
            .min(self.fade_frames);
        let t = self.blend_factor();

        if self.frame_counter >= self.fade_frames {
            // Exactly the target at t = 1 (including its reflection list).
            copy_coeffs(&mut self.current, &self.target);
            return t;
        }

        lerp_band(&self.from.direct_gain, &self.target.direct_gain, t, &mut self.current.direct_gain);
        self.current.direct_delay_samples =
            lerp(self.from.direct_delay_samples, self.target.direct_delay_samples, t);
        self.current.direct_azimuth =
            lerp_angle(self.from.direct_azimuth, self.target.direct_azimuth, t);
        self.current.direct_elevation =
            lerp(self.from.direct_elevation, self.target.direct_elevation, t);
        lerp_band(&self.from.late_t60, &self.target.late_t60, t, &mut self.current.late_t60);
        self.current.late_gain_db = lerp(self.from.late_gain_db, self.target.late_gain_db, t);

        for ((dst, a), b) in self
            .current
            .early_reflections
            .iter_mut()
            .zip(self.from.early_reflections.iter())
            .zip(self.ref_to.iter())
        {
            dst.azimuth = lerp_angle(a.azimuth, b.azimuth, t);
            dst.elevation = lerp(a.elevation, b.elevation, t);
            dst.delay_samples = lerp(a.delay_samples, b.delay_samples, t);
            lerp_band(&a.gain, &b.gain, t, &mut dst.gain);
        }

        t
    }

    /// Snap to `coefficients` instantly (no crossfade), copying BY REFERENCE into the
    /// preallocated storage: never allocates, safe on the audio thread. Used for the first
    /// real update of a pair so it does not glide from the default coefficients (#119).
    pub fn snap_to_ref(&mut self, coefficients: &SpatialCoefficients) {
        copy_coeffs(&mut self.target, coefficients);
        self.snap_internal();
    }

    /// Reset to a new starting point instantly (no crossfade).
    pub fn snap_to(&mut self, coefficients: SpatialCoefficients) {
        copy_coeffs(&mut self.target, &coefficients);
        self.snap_internal();
    }
}
