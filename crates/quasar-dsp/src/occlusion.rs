use quasar_core::bands::{Band8, FREQ_BAND_CENTRES};
use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use crate::fractional_delay::HermiteInterpolatingDelayLine;
use crate::node_graph::AudioNode;

/// Number of EQ sections / bands.
const BANDS: usize = 8;
/// Quality factor of the peaking sections (about one octave wide).
const PEAK_Q: f64 = std::f64::consts::SQRT_2;
/// Quality factor of the two shelving sections (maximally flat transition, no overshoot).
const SHELF_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// Shape depth floor (dB) relative to the loudest band: bands are never asked to go lower.
const MIN_SHAPE_DB: f64 = -80.0;
/// Bounds for the solved per-section gains (dB), so the design can never run away.
const SECTION_DB_RANGE: (f64, f64) = (-90.0, 30.0);
/// Refinement iterations of the per-block filter design.
const DESIGN_ITERATIONS: usize = 4;
/// Largest delay slope (samples per sample) that is rendered as a pitch-shifting
/// ramp (Doppler). 0.25 is an equivalent radial speed of ~86 m/s. A block whose
/// target would need a steeper slope is rendered as a crossfade between the old
/// and the new delay instead (no pitch glide, no click).
pub const MAX_DELAY_SLEW: f32 = 0.25;

/// Per-source direct-path stage: propagation delay + per-band air absorption /
/// occlusion filtering.
///
/// ```text
/// input -> fractional delay (Hermite, ramped per sample) -> 8-section EQ -> x gain
/// ```
///
/// **Filtering.** `SpatialCoefficients::direct_gain` holds one linear gain per
/// octave band (`FREQ_BAND_CENTRES`, 62.5 Hz .. 8 kHz). The loudest band becomes
/// a broadband scalar gain; the remaining *shape* (always <= 0 dB) is realised by
/// a cascade of a low shelf (band 0), six peaking sections (bands 1..=6) and a
/// high shelf (band 7), so everything above 8 kHz keeps following the top band.
/// Per block the section gains are solved (inverse of the sections' own overlap
/// matrix, refined by a few iterations against the exact response) so the
/// cascade hits each band gain at its centre frequency (within a fraction of a
/// dB for physical, smooth gain vectors; bands more than 80 dB below the loudest
/// are floored). Between blocks the biquad coefficients and the scalar gain are
/// interpolated linearly per sample, so a gain change never produces a zipper
/// step, and filter state persists across blocks.
///
/// **Delay.** The delay (`direct_delay_samples`, samples at the device rate) is
/// ramped linearly per sample from the previous block's value to the new
/// target, so a moving emitter produces a Doppler shift (slope `r` samples/sample
/// gives a pitch factor `1 - r`). Slopes beyond [`MAX_DELAY_SLEW`] (teleports,
/// first updates) are rendered as a one-block crossfade between the old and the
/// new delay instead. For smooth Doppler the engine's crossfade time should
/// roughly match the compute update interval, because the delay trajectory is
/// exactly the crossfaded coefficient.
///
/// The first block after construction / [`AudioNode::reset`] starts at its target
/// (no ramp from an arbitrary value). All memory is allocated in [`Self::new`];
/// processing is allocation-free.
pub struct AirAbsorptionOcclusionNode {
    input_channels: u16,
    output_channels: u16,
    sample_rate: f32,
    max_delay: f32,
    /// One delay line per channel.
    delay_lines: Vec<HermiteInterpolatingDelayLine>,
    /// Per channel, per section transposed-direct-form-II state `[z1, z2]`.
    states: Vec<[[f32; 2]; BANDS]>,
    design: ShapeDesigner,
    /// Last solved design (a pure function of the band gains): `(gain bits, scalar, coefficients)`.
    /// A steady scene presents the same gains every block, so the solve is skipped.
    design_cache: Option<([u32; BANDS], f32, [[f32; 5]; BANDS])>,
    /// Coefficients `[b0, b1, b2, a1, a2]` per section at the end of the last block.
    cur_coefs: [[f32; 5]; BANDS],
    /// Broadband gain at the end of the last block.
    cur_scalar: f32,
    /// Delay (samples) at the end of the last block.
    cur_delay: f32,
    primed: bool,
}

impl AirAbsorptionOcclusionNode {
    /// Create the node. `max_delay_secs` bounds the propagation delay (it is
    /// clamped to that capacity at run time).
    pub fn new(input_channels: u16, sample_rate: f32, max_delay_secs: f32) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let delay_lines: Vec<_> = (0..input_channels.max(1))
            .map(|_| HermiteInterpolatingDelayLine::new(max_delay_secs.max(0.0), sr))
            .collect();
        let max_delay = delay_lines[0].max_samples().saturating_sub(3) as f32;
        let channels = input_channels.max(1) as usize;
        Self {
            input_channels,
            output_channels: input_channels,
            sample_rate: sr,
            max_delay,
            delay_lines,
            states: vec![[[0.0; 2]; BANDS]; channels],
            design: ShapeDesigner::new(sr as f64),
            design_cache: None,
            cur_coefs: IDENTITY_COEFS,
            cur_scalar: 1.0,
            cur_delay: 0.0,
            primed: false,
        }
    }

    /// Prime the node: start from exactly these band gains and this delay (no
    /// ramp from the previous values). The next `process` / `process_raw` call
    /// ramps from here to the gains / delay it is given.
    ///
    /// `process_raw` and `process` take their gains from
    /// `SpatialCoefficients::direct_gain` and their delay from the explicit
    /// argument / `direct_delay_samples`; this method does not override them.
    pub fn update_occlusion(&mut self, attenuation: &Band8, delay_samples: f32) {
        let (scalar, coefs) = self.design_for(attenuation);
        self.cur_scalar = scalar;
        self.cur_coefs = coefs;
        self.cur_delay = self.clamp_delay(delay_samples);
        self.primed = true;
    }

    /// Render one block with an explicit delay (`delay_samples` overrides
    /// `params.direct_delay_samples`); the band gains come from
    /// `params.direct_gain`.
    pub fn process_raw(
        &mut self,
        input: &AudioBuffer,
        output: &mut AudioBuffer,
        params: &SpatialCoefficients,
        delay_samples: f32,
    ) {
        self.process_with_gains(input, output, &params.direct_gain, delay_samples);
    }

    /// Render one block with explicit per-band `gains` (linear) and delay. The engine passes
    /// `direct_gain x directivity_gain` here (#74).
    pub fn process_with_gains(
        &mut self,
        input: &AudioBuffer,
        output: &mut AudioBuffer,
        gains: &Band8,
        delay_samples: f32,
    ) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let n = input.samples() as usize;
        let channels = (self.input_channels as usize).min(self.delay_lines.len());

        let (target_scalar, target_coefs) = self.design_for(gains);
        let target_delay = self.clamp_delay(delay_samples);

        if !self.primed {
            self.cur_scalar = target_scalar;
            self.cur_coefs = target_coefs;
            self.cur_delay = target_delay;
            self.primed = true;
        }

        let inv_n = 1.0 / n.max(1) as f32;
        let d0 = self.cur_delay;
        let delay_slope = (target_delay - d0) * inv_n;
        let jump = delay_slope.abs() > MAX_DELAY_SLEW;

        // Sections whose coefficients are an exact identity at both ends and whose
        // state has died out are skipped (the common "no shaping needed" case).
        let mut active = [false; BANDS];
        let mut step = [[0.0_f32; 5]; BANDS];
        for s in 0..BANDS {
            let identity = self.cur_coefs[s] == IDENTITY_COEF && target_coefs[s] == IDENTITY_COEF;
            let mut live = !identity;
            if identity {
                for st in self.states.iter().take(channels) {
                    if st[s][0].abs() + st[s][1].abs() > 1e-20 {
                        live = true;
                    }
                }
            }
            active[s] = live;
            for k in 0..5 {
                step[s][k] = (target_coefs[s][k] - self.cur_coefs[s][k]) * inv_n;
            }
        }

        let g0 = self.cur_scalar;
        let gstep = (target_scalar - g0) * inv_n;

        // Live sections, in cascade order, with their start coefficients / per-sample slopes.
        let mut live = [0usize; BANDS];
        let mut n_live = 0;
        for s in 0..BANDS {
            if active[s] {
                live[n_live] = s;
                n_live += 1;
            }
        }
        let ramping = (0..n_live).any(|l| step[live[l]] != [0.0; 5]);
        let nn = n.min(DEFAULT_BLOCK_SIZE);

        for ch in 0..channels {
            // Push the whole block, then read it back (see `HermiteInterpolatingDelayLine::tap_back`):
            // a steady delay is one FIR pass, a gliding one a gather + vectorised interpolation.
            let line = &mut self.delay_lines[ch];
            line.push_slice(&input.channel(ch as u16)[..nn]);
            let mut y = [0.0_f32; DEFAULT_BLOCK_SIZE];
            if jump {
                let mut b = [0.0_f32; DEFAULT_BLOCK_SIZE];
                line.tap_block_const(d0, &mut y[..nn]);
                line.tap_block_const(target_delay, &mut b[..nn]);
                for i in 0..nn {
                    let w = (i + 1) as f32 * inv_n;
                    y[i] = (1.0 - w) * y[i] + w * b[i];
                }
            } else if delay_slope == 0.0 {
                line.tap_block_const(d0, &mut y[..nn]);
            } else {
                let mut d = [0.0_f32; DEFAULT_BLOCK_SIZE];
                for (i, v) in d[..nn].iter_mut().enumerate() {
                    *v = d0 + delay_slope * (i + 1) as f32;
                }
                line.tap_many(&d[..nn], nn - 1, true, &mut y[..nn]);
            }

            // EQ cascade, section-major in groups of 4 / 2 / 1 sections fused in one pass.
            let st = &mut self.states[ch];
            let mut l = 0;
            while l < n_live {
                let left = n_live - l;
                let take = if left >= 8 { 8 } else if left >= 4 { 4 } else if left >= 2 { 2 } else { 1 };
                match take {
                    8 => run_group::<8>(&mut y[..nn], &live[l..l + 8], &self.cur_coefs, &step, st, ramping),
                    4 => run_group::<4>(&mut y[..nn], &live[l..l + 4], &self.cur_coefs, &step, st, ramping),
                    2 => run_group::<2>(&mut y[..nn], &live[l..l + 2], &self.cur_coefs, &step, st, ramping),
                    _ => run_group::<1>(&mut y[..nn], &live[l..l + 1], &self.cur_coefs, &step, st, ramping),
                }
                l += take;
            }
            let out = output.channel_mut(ch as u16);
            for (i, (o, v)) in out[..nn].iter_mut().zip(&y[..nn]).enumerate() {
                *o = *v * (g0 + gstep * (i + 1) as f32);
            }
        }

        self.cur_scalar = target_scalar;
        self.cur_coefs = target_coefs;
        self.cur_delay = target_delay;
    }

    /// Solve the cascade for `gains`: returns the broadband scalar and the
    /// section coefficients.
    fn design_for(&mut self, gains: &Band8) -> (f32, [[f32; 5]; BANDS]) {
        let mut key = [0u32; BANDS];
        for (k, v) in key.iter_mut().zip(gains.0.iter()) {
            *k = v.to_bits();
        }
        if let Some((ck, s, c)) = &self.design_cache {
            if *ck == key {
                return (*s, *c);
            }
        }
        let (scalar, coefs) = self.design_uncached(gains);
        self.design_cache = Some((key, scalar, coefs));
        (scalar, coefs)
    }

    fn design_uncached(&mut self, gains: &Band8) -> (f32, [[f32; 5]; BANDS]) {
        let mut g = [0.0_f64; BANDS];
        let mut scalar = 0.0_f64;
        for (i, v) in g.iter_mut().enumerate() {
            let x = gains.0[i];
            *v = if x.is_finite() { (x as f64).clamp(0.0, 16.0) } else { 0.0 };
            scalar = scalar.max(*v);
        }
        if scalar < 1e-9 {
            // Silence: no shaping needed, the scalar gain does the work.
            return (0.0, IDENTITY_COEFS);
        }
        let mut shape_db = [0.0_f64; BANDS];
        for i in 0..BANDS {
            let rel = (g[i] / scalar).max(1e-12);
            shape_db[i] = (20.0 * rel.log10()).max(MIN_SHAPE_DB);
        }
        (scalar as f32, self.design.solve(&shape_db))
    }

    fn clamp_delay(&self, d: f32) -> f32 {
        if d.is_finite() { d.clamp(0.0, self.max_delay) } else { 0.0 }
    }

    /// Sample rate the node was built for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

impl AudioNode for AirAbsorptionOcclusionNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &SpatialCoefficients) {
        self.process_raw(input, output, params, params.direct_delay_samples);
    }

    fn reset(&mut self) {
        for l in self.delay_lines.iter_mut() {
            l.clear();
        }
        for st in self.states.iter_mut() {
            *st = [[0.0; 2]; BANDS];
        }
        self.cur_coefs = IDENTITY_COEFS;
        self.cur_scalar = 1.0;
        self.cur_delay = 0.0;
        self.primed = false;
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}

// ── Filter design ────────────────────────────────────────────────────────

const IDENTITY_COEF: [f32; 5] = [1.0, 0.0, 0.0, 0.0, 0.0];
const IDENTITY_COEFS: [[f32; 5]; BANDS] = [IDENTITY_COEF; BANDS];

/// Kind of one EQ section.
#[derive(Clone, Copy)]
enum Kind {
    LowShelf,
    Peak,
    HighShelf,
}

/// Designs the 8-section cascade for a target dB shape. Built once (allocation
/// free afterwards: fixed-size arrays only).
struct ShapeDesigner {
    sample_rate: f64,
    /// Section frequency (corner for shelves, centre for peaks) and kind.
    sections: [(Kind, f64); BANDS],
    /// `cos(w)`, `cos(2w)` at each band centre (for response evaluation).
    cos1: [f64; BANDS],
    cos2: [f64; BANDS],
    /// Inverse of the sections' overlap matrix: section dB = `minv` * target dB.
    minv: [[f64; BANDS]; BANDS],
}

impl ShapeDesigner {
    fn new(sample_rate: f64) -> Self {
        let c = FREQ_BAND_CENTRES;
        let mut sections = [(Kind::Peak, 0.0); BANDS];
        for i in 0..BANDS {
            sections[i] = (Kind::Peak, c[i] as f64);
        }
        // Shelves sit on the outer band edges (half an octave from the outer centres).
        sections[0] = (Kind::LowShelf, c[0] as f64 * std::f64::consts::SQRT_2);
        sections[BANDS - 1] = (Kind::HighShelf, c[BANDS - 1] as f64 / std::f64::consts::SQRT_2);

        let mut cos1 = [0.0; BANDS];
        let mut cos2 = [0.0; BANDS];
        for i in 0..BANDS {
            let w = 2.0 * std::f64::consts::PI * (c[i] as f64) / sample_rate;
            cos1[i] = w.cos();
            cos2[i] = (2.0 * w).cos();
        }

        let mut d = Self { sample_rate, sections, cos1, cos2, minv: [[0.0; BANDS]; BANDS] };

        // Overlap matrix from the sections' response at a -6 dB probe gain.
        let probe = -6.0;
        let mut m = [[0.0_f64; BANDS]; BANDS];
        for j in 0..BANDS {
            let mut gains = [0.0_f64; BANDS];
            gains[j] = probe;
            let coef = d.coefs(&gains);
            let resp = d.response_db(&coef);
            for i in 0..BANDS {
                m[i][j] = resp[i] / probe;
            }
        }
        d.minv = invert(m);
        d
    }

    /// Section coefficients (normalised, `a0 = 1`) for per-section gains in dB.
    fn coefs(&self, gains_db: &[f64; BANDS]) -> [[f64; 5]; BANDS] {
        let mut out = [[0.0_f64; 5]; BANDS];
        for s in 0..BANDS {
            out[s] = section_coef(self.sections[s].0, self.sections[s].1, gains_db[s], self.sample_rate);
        }
        out
    }

    /// Cascade magnitude response (dB) at the band centres.
    fn response_db(&self, coefs: &[[f64; 5]; BANDS]) -> [f64; BANDS] {
        let mut out = [0.0_f64; BANDS];
        for i in 0..BANDS {
            let (c1, c2) = (self.cos1[i], self.cos2[i]);
            let mut db = 0.0;
            for c in coefs.iter() {
                let (b0, b1, b2, a1, a2) = (c[0], c[1], c[2], c[3], c[4]);
                let num = b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (b0 * b1 + b1 * b2) * c1 + 2.0 * b0 * b2 * c2;
                let den = 1.0 + a1 * a1 + a2 * a2 + 2.0 * (a1 + a1 * a2) * c1 + 2.0 * a2 * c2;
                db += 10.0 * (num.max(1e-300) / den.max(1e-300)).log10();
            }
            out[i] = db;
        }
        out
    }

    /// Solve for coefficients whose response at the band centres is `target_db`.
    fn solve(&self, target_db: &[f64; BANDS]) -> [[f32; 5]; BANDS] {
        let mut p = mat_vec(&self.minv, target_db);
        clamp_all(&mut p);
        let mut coefs = self.coefs(&p);
        for _ in 0..DESIGN_ITERATIONS {
            let actual = self.response_db(&coefs);
            let mut err = [0.0_f64; BANDS];
            let mut worst = 0.0_f64;
            for i in 0..BANDS {
                err[i] = target_db[i] - actual[i];
                worst = worst.max(err[i].abs());
            }
            if worst < 0.02 {
                break;
            }
            let corr = mat_vec(&self.minv, &err);
            for i in 0..BANDS {
                p[i] += corr[i];
            }
            clamp_all(&mut p);
            coefs = self.coefs(&p);
        }
        let mut out = [[0.0_f32; 5]; BANDS];
        for s in 0..BANDS {
            // An exactly-flat section is emitted as an exact identity (so it can be skipped).
            if p[s].abs() < 1e-4 {
                out[s] = IDENTITY_COEF;
            } else {
                for k in 0..5 {
                    out[s][k] = coefs[s][k] as f32;
                }
            }
        }
        out
    }
}

fn clamp_all(p: &mut [f64; BANDS]) {
    for v in p.iter_mut() {
        *v = v.clamp(SECTION_DB_RANGE.0, SECTION_DB_RANGE.1);
    }
}

fn mat_vec(m: &[[f64; BANDS]; BANDS], v: &[f64; BANDS]) -> [f64; BANDS] {
    let mut out = [0.0; BANDS];
    for i in 0..BANDS {
        let mut s = 0.0;
        for j in 0..BANDS {
            s += m[i][j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Gauss-Jordan inverse with partial pivoting. Falls back to the identity if the
/// matrix is singular (cannot happen for the fixed section layout).
fn invert(mut a: [[f64; BANDS]; BANDS]) -> [[f64; BANDS]; BANDS] {
    let mut inv = [[0.0_f64; BANDS]; BANDS];
    for i in 0..BANDS {
        inv[i][i] = 1.0;
    }
    for col in 0..BANDS {
        let mut piv = col;
        for r in col + 1..BANDS {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            let mut id = [[0.0_f64; BANDS]; BANDS];
            for i in 0..BANDS {
                id[i][i] = 1.0;
            }
            return id;
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for k in 0..BANDS {
            a[col][k] /= d;
            inv[col][k] /= d;
        }
        for r in 0..BANDS {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    for k in 0..BANDS {
                        a[r][k] -= f * a[col][k];
                        inv[r][k] -= f * inv[col][k];
                    }
                }
            }
        }
    }
    inv
}

/// RBJ cookbook section, normalised so `a0 = 1`: `[b0, b1, b2, a1, a2]`.
fn section_coef(kind: Kind, freq: f64, gain_db: f64, sample_rate: f64) -> [f64; 5] {
    let a = 10.0_f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * freq / sample_rate;
    let (sin_w0, cos_w0) = w0.sin_cos();
    match kind {
        Kind::Peak => {
            let alpha = sin_w0 / (2.0 * PEAK_Q);
            let a0 = 1.0 + alpha / a;
            [
                (1.0 + alpha * a) / a0,
                -2.0 * cos_w0 / a0,
                (1.0 - alpha * a) / a0,
                -2.0 * cos_w0 / a0,
                (1.0 - alpha / a) / a0,
            ]
        }
        Kind::LowShelf => {
            let alpha = sin_w0 / (2.0 * SHELF_Q);
            let t = 2.0 * a.sqrt() * alpha;
            let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + t;
            [
                a * ((a + 1.0) - (a - 1.0) * cos_w0 + t) / a0,
                2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
                a * ((a + 1.0) - (a - 1.0) * cos_w0 - t) / a0,
                -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
                ((a + 1.0) + (a - 1.0) * cos_w0 - t) / a0,
            ]
        }
        Kind::HighShelf => {
            let alpha = sin_w0 / (2.0 * SHELF_Q);
            let t = 2.0 * a.sqrt() * alpha;
            let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + t;
            [
                a * ((a + 1.0) + (a - 1.0) * cos_w0 + t) / a0,
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
                a * ((a + 1.0) + (a - 1.0) * cos_w0 - t) / a0,
                2.0 * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
                ((a + 1.0) - (a - 1.0) * cos_w0 - t) / a0,
            ]
        }
    }
}

/// Run `N` consecutive live sections (`idx`, indices into the cascade) over `y` in place, fused in
/// one pass so their independent recursions overlap. Section `s` uses the coefficients
/// `base[s] + (i + 1) * step[s]` (accumulated per sample) at sample `i` when `ramp`, the constant
/// `base[s]` otherwise.
#[inline(always)]
fn run_group<const N: usize>(
    y: &mut [f32],
    idx: &[usize],
    base: &[[f32; 5]; BANDS],
    step: &[[f32; 5]; BANDS],
    states: &mut [[f32; 2]; BANDS],
    ramp: bool,
) {
    let mut b = [[0.0_f32; 5]; N];
    let mut d = [[0.0_f32; 5]; N];
    let mut z = [[0.0_f32; 2]; N];
    for k in 0..N {
        b[k] = base[idx[k]];
        d[k] = step[idx[k]];
        z[k] = states[idx[k]];
    }
    if ramp {
        // Coefficients advance by `step` per sample (accumulated, exactly as the per-sample
        // interpolation always did, so the output is unchanged).
        let mut c = b;
        for v in y.iter_mut() {
            let mut x = *v;
            for k in 0..N {
                for j in 0..5 {
                    c[k][j] += d[k][j];
                }
                let out = c[k][0] * x + z[k][0];
                z[k][0] = c[k][1] * x - c[k][3] * out + z[k][1];
                z[k][1] = c[k][2] * x - c[k][4] * out;
                x = out;
            }
            *v = x;
        }
    } else {
        for v in y.iter_mut() {
            let mut x = *v;
            for k in 0..N {
                let c = &b[k];
                let out = c[0] * x + z[k][0];
                z[k][0] = c[1] * x - c[3] * out + z[k][1];
                z[k][1] = c[2] * x - c[4] * out;
                x = out;
            }
            *v = x;
        }
    }
    for k in 0..N {
        states[idx[k]] = z[k];
    }
}
