//! Feedback Delay Network late reverberation.
//!
//! # Structure (#66)
//!
//! * **32 delay lines**, integer lengths: the 48 kHz design lengths (2.3 .. 95 ms)
//!   are scaled by `fs / 48000` and moved to the nearest unused **prime** (distinct
//!   primes are pairwise coprime, so the mode frequencies do not stack and the echo
//!   density keeps growing), see [`fdn_delay_lengths`]. The total delay in seconds
//!   (the modal density) and the delay lengths in milliseconds are therefore the same
//!   at every sample rate to within the rounding.
//! * **One read per line.** Every line is a ring of exactly `d_i` samples: the sample
//!   written now is read `d_i` samples later, with no interpolation. The loop delay is
//!   exactly `d_i` (the old implementation read `tap(L)` before pushing, a
//!   loop of `L + 1` samples, through a Hermite interpolator).
//! * **Householder feedback matrix** `A = I - (2/N) 1 1^T` (orthogonal, `O(N)`).
//! * **Block-friendly loop.** All lines are at least `chunk` samples long (the shortest
//!   delay), so a chunk of up to `chunk` samples can be read from every line, filtered,
//!   mixed and written back line by line instead of sample by sample: the inner loops
//!   run over contiguous samples (auto-vectorisable), the feedback dependency never
//!   crosses a chunk.
//! * **Input / output vectors.** Input is injected through Walsh-Hadamard row
//!   [`BUS_INPUT_ROW`] and output channel `k` is row `k + 1` of the line outputs,
//!   `y_k = (1/sqrt N) sum_i H[k+1][i] f_i`. Rows are mutually orthogonal and zero-sum,
//!   so the output channels are decorrelated copies of one diffuse tail, and neither
//!   the input nor an output couples to the all-ones eigenvector of `A` (which would
//!   ring as an undamped common mode). One 32-point fast Walsh-Hadamard transform
//!   produces every row.
//!
//! # Decay (#63)
//!
//! Line `i` carries the gain that makes the loop lose exactly 60 dB in `T60`,
//! `g_i = 10^(-3 d_i / (fs T60))`. The 8-band T60 is fitted with three bands:
//! `T_low` = mean of the 62.5 / 125 / 250 Hz bands, `T_mid` = 500 Hz .. 2 kHz,
//! `T_high` = 4 / 8 kHz. Each line is `g_mid * lowshelf * highshelf` with
//! first-order shelves (corners [`LOW_SHELF_HZ`] / [`HIGH_SHELF_HZ`]) whose shelf gains are
//! `g_low / g_mid` and `g_high / g_mid`. Because the shelf poles are fixed and only
//! the numerators depend on the gains, T60 changes are applied by interpolating the
//! gains linearly per sample (no zipper noise, no filter instability, #122). A flat T60
//! reduces to the plain gains and is accurate to a few percent; a strongly
//! frequency-dependent T60 follows the 3-band fit (first-order shelves have gentle
//! slopes, so the 250 Hz and 2 kHz bands only reach about 2/3 of the way).
//!
//! # Level
//!
//! The wet level is calibrated: with a unitary matrix and unit-norm input / output vectors
//! one output has impulse-response energy `E ~ fs T60 / (13.8 sum d_i)`; the node divides by
//! `sqrt(E)`, so [`FdnReverbNode::set_wet`] is the RMS gain of one output channel for a
//! white-noise input, independent of the T60 (the T60 is smoothed too, so the level
//! follows it continuously).
//!
//! All memory is allocated at construction. NEVER allocates during processing.

use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use crate::node_graph::AudioNode;
use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;

/// Number of delay lines.
pub const FDN_LINES: usize = 32;
/// Input row of the Walsh-Hadamard matrix (a zero-sum row not used by the first outputs).
pub const BUS_INPUT_ROW: usize = 21;
/// Output channels one FDN can feed (rows 1..=31 of the 32x32 Hadamard matrix).
pub const FDN_MAX_BUS_OUTPUTS: usize = 31;
/// Corner (Hz) of the low shelf (between the 250 Hz and 500 Hz bands).
pub const LOW_SHELF_HZ: f32 = 354.0;
/// Corner (Hz) of the high shelf (between the 2 kHz and 4 kHz bands).
pub const HIGH_SHELF_HZ: f32 = 2828.0;

/// Design delay lengths (samples at 48 kHz, 2.3 .. 95 ms).
const BASE_LENGTHS_48K: [usize; FDN_LINES] = [
    719, 857, 1103, 1321, 1613, 1871, 2213, 2657, 3079, 3491, 109, 151, 197, 251, 313, 401, 503,
    613, 823, 947, 1069, 1229, 1453, 1741, 1997, 2347, 2741, 3041, 3373, 3733, 4099, 4561,
];

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Delay lengths (samples) of the 32 lines for `sample_rate`: the 48 kHz design
/// lengths scaled by `fs / 48000`, each moved up to the nearest prime not already used.
/// Distinct primes are pairwise coprime. API / construction time only.
pub fn fdn_delay_lengths(sample_rate: f32) -> [usize; FDN_LINES] {
    let scale = if sample_rate.is_finite() && sample_rate > 8000.0 { sample_rate / 48_000.0 } else { 1.0 };
    let mut out = [0usize; FDN_LINES];
    for (i, &base) in BASE_LENGTHS_48K.iter().enumerate() {
        let mut n = ((base as f32 * scale).round() as usize).max(8);
        while !is_prime(n) || out[..i].contains(&n) {
            n += 1;
        }
        out[i] = n;
    }
    out
}

/// Entry `i` of Walsh-Hadamard row `row`: `+-1`.
#[inline]
fn hadamard(row: usize, i: usize) -> f32 {
    if (row & i).count_ones() % 2 == 0 { 1.0 } else { -1.0 }
}

/// In-place 32-point fast Walsh-Hadamard transform over the ROWS of a `[32][c]` array
/// (row stride `c`): `row[r] <- sum_i hadamard(r, i) row[i]` (natural / Hadamard order).
fn fwht_rows(v: &mut [f32], c: usize) {
    let mut h = 1;
    while h < FDN_LINES {
        let mut i = 0;
        while i < FDN_LINES {
            for j in i..i + h {
                let (lo, hi) = v.split_at_mut((j + h) * c);
                let a = &mut lo[j * c..j * c + c];
                let b = &mut hi[..c];
                for k in 0..c {
                    let (x, y) = (a[k], b[k]);
                    a[k] = x + y;
                    b[k] = x - y;
                }
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Per-line loop gains: `[g_mid, G_low, G_high]` (linear; shelf gains are relative to mid).
type LineGains = [f32; 3];

/// Implementation of a Feedback Delay Network reverberator (see the module docs).
///
/// Used either as a mono-in / N-out [`AudioNode`] (the wet signal only) or as
/// a shared bus through [`FdnReverbNode::process_bus`].
pub struct FdnReverbNode {
    sample_rate: f32,
    lens: [usize; FDN_LINES],
    /// All rings back to back; line `i` is `ring[off[i] .. off[i] + lens[i]]`.
    ring: Vec<f32>,
    off: [usize; FDN_LINES],
    pos: [usize; FDN_LINES],
    /// Chunk length: the shortest delay.
    chunk: usize,
    /// Shelf filter memories (transposed direct form II).
    low_state: [f32; FDN_LINES],
    high_state: [f32; FDN_LINES],
    /// Loop gains per line: previous block's value (ramp start) and target.
    gains_cur: [LineGains; FDN_LINES],
    gains_tgt: [LineGains; FDN_LINES],
    /// Output scale (wet x level normalisation), ramp start and target.
    out_cur: f32,
    out_tgt: f32,
    /// `sum_i d_i` (level calibration).
    total_len: f32,
    wet: f32,
    t60: Band8,
    /// `K = tan(pi fc / fs)` and `1 / (1 + K)` of the two shelves.
    low_k: f32,
    low_inv: f32,
    high_k: f32,
    high_inv: f32,
    /// Integer pre-delay ring.
    pre: Vec<f32>,
    pre_pos: usize,
    pre_len: usize,
    max_pre: usize,
    /// False until the first block after construction / reset (starts at its target).
    ramp_valid: bool,
    /// Per-output decorrelation delays (samples) and their history rings (see `OUT_DELAY_STEP`).
    out_delay: [usize; FDN_MAX_BUS_OUTPUTS],
    ohist: Vec<f32>,
    omask: usize,
    owr: usize,
    /// `[32][chunk]` filtered line outputs / their Hadamard transform, and chunk scratch.
    f: Vec<f32>,
    h: Vec<f32>,
    xin: Vec<f32>,
    ssum: Vec<f32>,
    input_channels: u16,
    output_channels: u16,
}

impl FdnReverbNode {
    /// Create a new FDN reverb. `input_channels` sets the [`AudioNode`] channel
    /// count: the input is folded to mono and output channel `c` is Hadamard row `c + 1`.
    pub fn new(input_channels: u16, sample_rate: f32) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 8000.0 { sample_rate } else { 48_000.0 };
        let lens = fdn_delay_lengths(sr);
        let mut off = [0usize; FDN_LINES];
        let mut total = 0;
        for i in 0..FDN_LINES {
            off[i] = total;
            total += lens[i];
        }
        let chunk = *lens.iter().min().unwrap_or(&8);
        let k = |fc: f32| (std::f32::consts::PI * fc / sr).tan();
        let (low_k, high_k) = (k(LOW_SHELF_HZ), k(HIGH_SHELF_HZ.min(0.45 * sr)));
        let max_pre = (0.1 * sr) as usize + 1;
        let mut out_delay = [0usize; FDN_MAX_BUS_OUTPUTS];
        for (k, d) in out_delay.iter_mut().enumerate() {
            let mut n = (OUT_DELAY_STEP * sr / 48_000.0 * k as f32).round() as usize;
            while n > 0 && !is_prime(n) {
                n += 1;
            }
            *d = n;
        }
        let hist_len = (out_delay[FDN_MAX_BUS_OUTPUTS - 1] + chunk + 1).next_power_of_two();
        let mut node = Self {
            sample_rate: sr,
            lens,
            ring: vec![0.0; total],
            off,
            pos: [0; FDN_LINES],
            chunk,
            low_state: [0.0; FDN_LINES],
            high_state: [0.0; FDN_LINES],
            gains_cur: [[1.0, 1.0, 1.0]; FDN_LINES],
            gains_tgt: [[1.0, 1.0, 1.0]; FDN_LINES],
            out_cur: 0.0,
            out_tgt: 0.0,
            total_len: lens.iter().sum::<usize>() as f32,
            wet: 1.0,
            t60: Band8::splat(2.0),
            low_k,
            low_inv: 1.0 / (1.0 + low_k),
            high_k,
            high_inv: 1.0 / (1.0 + high_k),
            pre: vec![0.0; max_pre],
            pre_pos: 0,
            pre_len: 0,
            max_pre,
            ramp_valid: false,
            out_delay,
            ohist: vec![0.0; FDN_MAX_BUS_OUTPUTS * hist_len],
            omask: hist_len - 1,
            owr: 0,
            f: vec![0.0; FDN_LINES * chunk],
            h: vec![0.0; FDN_LINES * chunk],
            xin: vec![0.0; chunk],
            ssum: vec![0.0; chunk],
            input_channels,
            output_channels: input_channels,
        };
        node.set_t60(&Band8::splat(2.0));
        node
    }

    /// The integer delay length (samples) of each line.
    pub fn delay_lengths(&self) -> &[usize; FDN_LINES] {
        &self.lens
    }

    /// Total delay `sum d_i / fs` in seconds: proportional to the modal density
    /// (modes per Hz), the same at every sample rate.
    pub fn total_delay_secs(&self) -> f32 {
        self.lens.iter().sum::<usize>() as f32 / self.sample_rate
    }

    /// Compute the Householder feedback matrix for 32 lines.
    pub fn feedback_matrix(input: &[f32; 32]) -> [f32; 32] {
        let n = 32.0;
        let sum: f32 = input.iter().sum();
        let scale = 2.0 / n;
        let mut out = [0.0_f32; 32];
        for i in 0..32 {
            out[i] = -input[i] + scale * sum;
        }
        out
    }

    /// Set the T60 target per band (seconds). The per-line loop gains and the level
    /// normalisation glide to it linearly per sample over the next block.
    pub fn set_t60(&mut self, t60: &Band8) {
        let mut t = t60.0;
        for v in t.iter_mut() {
            *v = if v.is_finite() { v.clamp(0.05, 100.0) } else { 2.0 };
        }
        self.t60 = Band8::new(t);
        let t_low = (t[0] + t[1] + t[2]) / 3.0;
        let t_mid = (t[3] + t[4] + t[5]) / 3.0;
        let t_high = (t[6] + t[7]) / 2.0;
        let fs = self.sample_rate;
        for i in 0..FDN_LINES {
            // Amplitude gain per pass for a 60 dB decay in `tt` seconds.
            let g = |tt: f32| 10.0_f32.powf(-3.0 * self.lens[i] as f32 / (fs * tt));
            let g_mid = g(t_mid);
            self.gains_tgt[i] = [g_mid, g(t_low) / g_mid, g(t_high) / g_mid];
        }
        self.update_out_target();
    }

    /// Set the wet gain: the RMS gain of ONE output channel for a white-noise input,
    /// whatever the T60 (linear, ramped per sample over the next block).
    pub fn set_wet(&mut self, wet: f32) {
        self.wet = if wet.is_finite() { wet.max(0.0) } else { 0.0 };
        self.update_out_target();
    }

    fn update_out_target(&mut self) {
        let t_mean = self.t60.mean();
        // Impulse-response energy of one output: `fs T60 / (13.8155 sum_i d_i)` = `T60 / (13.8155 * total_delay_secs)`
        // (once the lines have mixed, every line passes the same energy flux, so the stored
        // energy is proportional to `d_i`; the energy decays 60 dB in `T60`; a unit-norm output
        // vector sees `1/N` of the energy passing the ports).

        let e = self.sample_rate * t_mean / (13.8155 * self.total_len);
        self.out_tgt = self.wet * LEVEL_CAL / e.sqrt();
    }

    /// Set pre-delay in seconds (rounded to whole samples, at most 0.1 s).
    pub fn set_pre_delay(&mut self, delay_secs: f32) {
        let n = if delay_secs.is_finite() { (delay_secs.max(0.0) * self.sample_rate).round() as usize } else { 0 };
        self.pre_len = n.min(self.max_pre);
        self.pre_pos = 0;
        for v in self.pre.iter_mut() {
            *v = 0.0;
        }
    }

    /// Update the reverb from spatial coefficients: the T60 and the wet level
    /// (`late_gain_db`) are both taken from `params` (this is the [`AudioNode`] path;
    /// a bus owner calls [`Self::set_t60`] / [`Self::set_wet`] itself).
    pub fn update_from_coefficients(&mut self, params: &SpatialCoefficients) {
        self.wet = if params.late_gain_db.is_finite() { 10.0_f32.powf(params.late_gain_db.min(40.0) / 20.0) } else { 0.0 };
        self.set_t60(&params.late_t60);
    }

    /// Shared reverb BUS: a mono input in, `n_out` DIFFUSE output channels out.
    ///
    /// Output channel `k` is Hadamard row `k + 1` of the line outputs (see the module
    /// docs); the rows are orthogonal, so the channels are decorrelated copies of the
    /// same tail. Writes channels `0..n_out` of `out` (overwriting). Allocation-free.
    pub fn process_bus(&mut self, input: &[f32], out: &mut AudioBuffer, n_out: usize) {
        let n_out = n_out.min(FDN_MAX_BUS_OUTPUTS).min(out.channels() as usize);
        let n = input.len().min(out.samples() as usize);
        self.run(&input[..n], out, n_out);
    }

    /// Core: process `input` (mono) and write output rows `1..=n_out` to channels `0..n_out`.
    fn run(&mut self, input: &[f32], out: &mut AudioBuffer, n_out: usize) {
        let n = input.len();
        if n == 0 {
            return;
        }
        if !self.ramp_valid {
            self.gains_cur = self.gains_tgt;
            self.out_cur = self.out_tgt;
            self.ramp_valid = true;
        }
        let inv_n = 1.0 / n as f32;
        let norm = 1.0 / (FDN_LINES as f32).sqrt();
        let (lk, li) = (self.low_k, self.low_inv);
        let (hk, hi) = (self.high_k, self.high_inv);
        let (a_lo, a_hi) = ((lk - 1.0) * li, (hk - 1.0) * hi);
        let householder = 2.0 / FDN_LINES as f32;

        let mut j0 = 0;
        while j0 < n {
            let c = (n - j0).min(self.chunk);

            // Input (through the pre-delay).
            for j in 0..c {
                let x = input[j0 + j];
                self.xin[j] = if self.pre_len == 0 {
                    x
                } else {
                    let y = self.pre[self.pre_pos];
                    self.pre[self.pre_pos] = x;
                    self.pre_pos += 1;
                    if self.pre_pos >= self.pre_len {
                        self.pre_pos = 0;
                    }
                    y
                };
            }

            // 1-2. Read each line (one read, no interpolation) and apply its loop filter.
            for i in 0..FDN_LINES {
                let d = self.lens[i];
                let base = self.off[i];
                let (g0, g1) = (self.gains_cur[i], self.gains_tgt[i]);
                let (mut sl, mut sh) = (self.low_state[i], self.high_state[i]);
                let fi = &mut self.f[i * self.chunk..i * self.chunk + c];
                let mut idx = self.pos[i];
                for j in 0..c {
                    let t = (j0 + j + 1) as f32 * inv_n;
                    let gm = g0[0] + (g1[0] - g0[0]) * t;
                    let gl = g0[1] + (g1[1] - g0[1]) * t;
                    let gh = g0[2] + (g1[2] - g0[2]) * t;
                    let u = self.ring[base + idx] * gm;
                    idx += 1;
                    if idx >= d {
                        idx = 0;
                    }
                    // Low shelf (DC gain gl, 1 at HF), then high shelf (HF gain gh, 1 at DC).
                    let y1 = (1.0 + gl * lk) * li * u + sl;
                    sl = (gl * lk - 1.0) * li * u - a_lo * y1;
                    let y2 = (gh + hk) * hi * y1 + sh;
                    sh = (hk - gh) * hi * y1 - a_hi * y2;
                    fi[j] = if y2.abs() < 1e-24 { 0.0 } else { y2 };
                }
                self.low_state[i] = sl;
                self.high_state[i] = sh;
            }

            // 3. Outputs: Hadamard rows of the (pre-matrix) line outputs. For a zero-sum
            //    row the Householder term cancels, so the rows of `f` are the output.
            if n_out > 0 {
                let len = FDN_LINES * self.chunk;
                self.h[..len].copy_from_slice(&self.f[..len]);
                fwht_rows(&mut self.h[..len], self.chunk);
                let hist_len = self.omask + 1;
                for k in 0..n_out {
                    let row = &self.h[(k + 1) * self.chunk..(k + 1) * self.chunk + c];
                    let hist = &mut self.ohist[k * hist_len..(k + 1) * hist_len];
                    let dst = &mut out.channel_mut(k as u16)[j0..j0 + c];
                    let dly = self.out_delay[k];
                    for j in 0..c {
                        let t = (j0 + j + 1) as f32 * inv_n;
                        let w = (self.owr + j) & self.omask;
                        hist[w] = row[j] * norm * (self.out_cur + (self.out_tgt - self.out_cur) * t);
                        dst[j] = hist[(w + hist_len - dly) & self.omask];
                    }
                }
            }

            self.owr = (self.owr + c) & self.omask;

            // 4. Householder mix and write back with the input injection.
            for s in self.ssum[..c].iter_mut() {
                *s = 0.0;
            }
            for i in 0..FDN_LINES {
                let fi = &self.f[i * self.chunk..i * self.chunk + c];
                for j in 0..c {
                    self.ssum[j] += fi[j];
                }
            }
            for i in 0..FDN_LINES {
                let d = self.lens[i];
                let base = self.off[i];
                let b = hadamard(BUS_INPUT_ROW, i) * norm;
                let fi = &self.f[i * self.chunk..i * self.chunk + c];
                let mut idx = self.pos[i];
                for j in 0..c {
                    self.ring[base + idx] = b * self.xin[j] + (householder * self.ssum[j] - fi[j]);
                    idx += 1;
                    if idx >= d {
                        idx = 0;
                    }
                }
                self.pos[i] = idx;
            }
            j0 += c;
        }
        self.gains_cur = self.gains_tgt;
        self.out_cur = self.out_tgt;
    }
}

impl AudioNode for FdnReverbNode {
    /// Folds the input to mono and writes the WET signal only (output channel `c` =
    /// Hadamard row `c + 1`); T60 and wet level are taken from `params`.
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &SpatialCoefficients) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        self.update_from_coefficients(params);

        let n = (input.samples() as usize).min(output.samples() as usize).min(DEFAULT_BLOCK_SIZE);
        let mut mono = [0.0_f32; DEFAULT_BLOCK_SIZE];
        let ch = (input.channels() as usize).max(1);
        for i in 0..n {
            let mut s = 0.0;
            for c in 0..input.channels() as usize {
                s += input.channel(c as u16)[i];
            }
            mono[i] = s / ch as f32;
        }
        let n_out = (self.output_channels as usize).min(FDN_MAX_BUS_OUTPUTS);
        self.run(&mono[..n], output, n_out);
    }

    fn reset(&mut self) {
        for v in self.ring.iter_mut() {
            *v = 0.0;
        }
        self.pos = [0; FDN_LINES];
        self.low_state = [0.0; FDN_LINES];
        self.high_state = [0.0; FDN_LINES];
        for v in self.pre.iter_mut() {
            *v = 0.0;
        }
        self.pre_pos = 0;
        self.ramp_valid = false;
        for v in self.ohist.iter_mut() {
            *v = 0.0;
        }
        self.owr = 0;
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}

/// Empirical level trim of the calibration (measured +1.1 .. +1.9 dB over T60 0.3 .. 6 s and
/// 44.1 / 48 / 96 kHz without it; with it the residual is within +-0.5 dB).
const LEVEL_CAL: f32 = 0.84;

/// Spacing (samples at 48 kHz) of the per-output decorrelation delays: output `k` is
/// delayed by the first prime `>= 55 k` samples (0, 59, 113, 167, ... about 1.1 ms apart).
/// The Householder matrix is symmetric, so the second-pass echoes `i -> k` and `k -> i`
/// have the same total delay and show up in two output rows at the same instant (zero-lag
/// correlation of up to 0.3 - 0.6 between some Hadamard rows); shifting the rows against
/// each other turns those into correlations at a non-zero lag, which a white-ish tail
/// does not have. Costs one short ring per output.
const OUT_DELAY_STEP: f32 = 55.0;
