//! Band-limited polyphase windowed-sinc resampler (#76).
//!
//! Replaces 2-point linear interpolation (which loses high frequencies and aliases / images
//! heavily) for bringing a source at its file rate to the device rate, on the source side, so the
//! engine always receives device-rate buffers.
//!
//! # Design
//!
//! * **Kernel.** A Kaiser-windowed sinc with cutoff at the Nyquist frequency of the LOWER of the two
//!   rates, tabulated at `PHASES` fractional positions (default 1024) and linearly interpolated
//!   between adjacent phases. Every phase row is normalised to unit DC gain.
//! * **Quality.** The passband edge is `fp = 18 kHz * min(rate_in, rate_out) / 44.1 kHz` (18 kHz for
//!   44.1 kHz material, 19.6 kHz at 48 kHz); the stop band starts at `min(rate) - fp`, i.e. exactly
//!   where images of a tone at `fp` would start to fold back. The tap count is chosen from the
//!   Kaiser formula for the requested stop-band attenuation (default 105 dB), e.g. ~40 taps for
//!   44.1 -> 48 kHz and ~80 for 96 -> 48 kHz. Measured (see `tests/resampler_tests.rs`): passband
//!   flat within 0.1 dB up to `fp`, image / alias products below -90 dB.
//! * **Timeline.** Output frame `k` is the band-limited signal evaluated at input time
//!   `t_k = k * step` (input frames, time 0 = the first input frame, silence before it), with
//!   `step = rate_in / rate_out * (1 + trim)`. There is no filter delay in that timeline; instead
//!   the resampler needs [`lookahead_frames`](PolyphaseResampler::lookahead_frames) frames beyond
//!   `t_k` to produce output `k` (free when the source is random access, as in the streaming ring).
//!   The total output for `M` pushed input frames is `ceil((M - lookahead) / step)`.
//! * **1:1.** With equal rates and zero trim every output is an EXACT copy of the input frame
//!   (bit-identical, zero latency).
//! * **Streaming.** Input is pushed in planar blocks of ANY size and output pulled in blocks of any
//!   size; the result is bit-identical however the stream is chunked (each output depends only on
//!   its own position). The callback-facing flow is: `input_needed(n)` -> fetch that many source
//!   frames -> `process` -> advance the source by `consumed`.
//! * **Drift.** [`set_ratio_trim`](PolyphaseResampler::set_ratio_trim) nudges the consumption
//!   ratio by a small fraction (e.g. `+50e-6` = 50 ppm faster consumption). A controller that
//!   watches the source ring's fill level (or a device clock) steers it to keep the buffer level
//!   constant over hours. The trim is slewed per sample ([`TRIM_SLEW_PER_SAMPLE`]), so changing it
//!   never produces a pitch step or click, and is clamped to +-2%. The kernel is designed for the
//!   nominal ratio; such small trims leave its response unchanged for practical purposes.
//! * **Real-time safety.** All memory (kernel table, history) is allocated in [`new`](
//!   PolyphaseResampler::new); `process` / `input_needed` never allocate, lock or panic.

use std::f64::consts::PI;

/// Fractional positions per input frame in the kernel table.
pub const DEFAULT_PHASES: usize = 1024;
/// Default stop-band attenuation target of the kernel (dB).
pub const DEFAULT_STOPBAND_DB: f64 = 105.0;
/// Largest |trim| accepted by [`PolyphaseResampler::set_ratio_trim`] (2%).
pub const MAX_RATIO_TRIM: f64 = 0.02;
/// Largest change of the trim per output sample (1000 ppm takes ~4 s at 48 kHz).
pub const TRIM_SLEW_PER_SAMPLE: f64 = 5e-9;
/// Spare input frames the history holds beyond the kernel window.
const HISTORY_SLACK: usize = 8192;

/// Result of one [`PolyphaseResampler::process`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResampleResult {
    /// Input frames taken from the input slices.
    pub consumed: usize,
    /// Output frames written to the output slices.
    pub produced: usize,
}

/// Streaming multi-channel polyphase resampler. See the module docs.
pub struct PolyphaseResampler {
    channels: usize,
    in_rate: f64,
    out_rate: f64,
    base_step: f64,
    trim_target: f64,
    trim: f64,
    /// Input time of the next output frame (input frames; 0 = first input frame).
    pos: f64,
    taps: usize,
    half: usize,
    phases: usize,
    /// `(phases + 1) x taps` kernel rows; row `p` is the kernel for fraction `p / phases`.
    table: Vec<f32>,
    /// Planar history, `channels x cap`; frames `[h0, h0 + len)` of the input timeline.
    hist: Vec<f32>,
    cap: usize,
    h0: i64,
    len: usize,
}

impl PolyphaseResampler {
    /// Resampler from `in_rate` to `out_rate` (Hz) for `channels` channels, with the default
    /// quality (105 dB stop band, 1024 phases). Allocates.
    pub fn new(channels: usize, in_rate: f64, out_rate: f64) -> Self {
        Self::with_quality(channels, in_rate, out_rate, DEFAULT_STOPBAND_DB, DEFAULT_PHASES)
    }

    /// Like [`new`](Self::new) with an explicit stop-band attenuation (dB, 60..=140) and phase
    /// count (64..=4096).
    pub fn with_quality(channels: usize, in_rate: f64, out_rate: f64, stopband_db: f64, phases: usize) -> Self {
        let channels = channels.max(1);
        let in_rate = if in_rate.is_finite() && in_rate > 0.0 { in_rate } else { 48_000.0 };
        let out_rate = if out_rate.is_finite() && out_rate > 0.0 { out_rate } else { 48_000.0 };
        let a = stopband_db.clamp(60.0, 140.0);
        let phases = phases.clamp(64, 4096);

        // Kernel design (input-sample units). Cutoff = Nyquist of the lower rate.
        let m = in_rate.min(out_rate);
        let fp = 18_000.0 * m / 44_100.0;
        let transition = (m - 2.0 * fp).max(0.02 * m) / in_rate; // (stop - fp) in cycles/input sample
        let n_est = ((a - 7.95) / (2.285 * 2.0 * PI * transition)).ceil() as usize + 4;
        let taps = (n_est + (n_est & 1)).clamp(16, 512);
        let half = taps / 2;
        let fc = 0.5 * (m / in_rate); // cycles per input sample
        let beta = if a > 50.0 { 0.1102 * (a - 8.7) } else { 0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0) };
        let inv_i0_beta = 1.0 / bessel_i0(beta);

        let mut table = vec![0.0_f32; (phases + 1) * taps];
        for p in 0..=phases {
            let frac = p as f64 / phases as f64;
            let row = &mut table[p * taps..(p + 1) * taps];
            let mut sum = 0.0_f64;
            let mut tmp = vec![0.0_f64; taps];
            for (j, v) in tmp.iter_mut().enumerate() {
                // Tap j reads frame (c - (half - 1) + j); its distance from t = c + frac:
                let x = j as f64 - (half as f64 - 1.0) - frac;
                let w = if x.abs() < half as f64 {
                    let r = x / half as f64;
                    bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) * inv_i0_beta
                } else {
                    0.0
                };
                let s = if x.abs() < 1e-12 { 2.0 * fc } else { (2.0 * PI * fc * x).sin() / (PI * x) };
                *v = s * w;
                sum += *v;
            }
            // Unit DC gain at every phase (no zipper at DC, exact for 1.0 inputs).
            for (j, v) in tmp.iter().enumerate() {
                row[j] = (*v / sum) as f32;
            }
        }

        let cap = taps + HISTORY_SLACK;
        let mut r = Self {
            channels,
            in_rate,
            out_rate,
            base_step: in_rate / out_rate,
            trim_target: 0.0,
            trim: 0.0,
            pos: 0.0,
            taps,
            half,
            phases,
            table,
            hist: vec![0.0; channels * cap],
            cap,
            h0: 0,
            len: 0,
        };
        r.reset();
        r
    }

    /// Restart at input time 0 with silent history and zero trim.
    pub fn reset(&mut self) {
        self.pos = 0.0;
        self.trim = 0.0;
        self.trim_target = 0.0;
        self.hist.fill(0.0);
        // Silence before the first frame: the window of output 0 reaches `half - 1` frames back.
        self.h0 = -(self.half as i64 - 1);
        self.len = self.half - 1;
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Kernel length in taps.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Input frames needed beyond an output's time `t` to compute it (`taps / 2`). With a 1:1 ratio
    /// and zero trim the resampler is an exact copy and needs none.
    pub fn lookahead_frames(&self) -> usize {
        self.half
    }

    /// Nominal consumption ratio `rate_in / rate_out` (without trim).
    pub fn ratio(&self) -> f64 {
        self.base_step
    }

    /// Input rate in Hz.
    pub fn in_rate(&self) -> f64 {
        self.in_rate
    }

    /// Output rate in Hz.
    pub fn out_rate(&self) -> f64 {
        self.out_rate
    }

    /// Set the drift-compensation trim: input frames are consumed at `ratio * (1 + trim)`.
    /// `trim` is clamped to [`-MAX_RATIO_TRIM`, `MAX_RATIO_TRIM`] and slewed to at most
    /// [`TRIM_SLEW_PER_SAMPLE`] per output sample, so the change is click-free. Positive values
    /// consume input faster (use it when the source ring is filling up).
    pub fn set_ratio_trim(&mut self, trim: f64) {
        self.trim_target = if trim.is_finite() { trim.clamp(-MAX_RATIO_TRIM, MAX_RATIO_TRIM) } else { 0.0 };
    }

    /// The trim currently applied (after slewing).
    pub fn ratio_trim(&self) -> f64 {
        self.trim
    }

    /// Input time (input frames) of the next output frame.
    pub fn position(&self) -> f64 {
        self.pos
    }

    #[inline]
    fn step(&self) -> f64 {
        self.base_step * (1.0 + self.trim)
    }

    /// Whether the next output is an exact copy of an input frame (1:1, zero trim, on a frame).
    #[inline]
    fn exact_copy(&self) -> bool {
        self.base_step == 1.0 && self.trim == 0.0 && self.trim_target == 0.0 && self.pos.fract() == 0.0
    }

    /// Input frames the caller must supply (on top of what the resampler already holds) so that
    /// `out_frames` outputs can be produced. Conservative by at most one frame; frames supplied
    /// beyond need are simply not consumed by [`process`](Self::process).
    pub fn input_needed(&self, out_frames: usize) -> usize {
        if out_frames == 0 {
            return 0;
        }
        let last_pos = self.pos + (out_frames - 1) as f64 * self.step().max(self.base_step * (1.0 + self.trim_target));
        let lookahead = if self.exact_copy() { 0 } else { self.half as i64 };
        let need_last = last_pos.floor() as i64 + lookahead; // inclusive absolute frame
        let have_last = self.h0 + self.len as i64 - 1;
        (need_last - have_last + 1).max(0) as usize
    }

    /// Resample. `input`: one slice per channel (equal lengths), `output`: one slice per channel
    /// (equal lengths). Pushes input into the history as needed and writes as many output frames
    /// as the input and the output capacity allow. Input is only taken while more output is
    /// wanted, so `consumed` can be less than the input length; present the remainder again on
    /// the next call. Never allocates.
    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]]) -> ResampleResult {
        let ch = self.channels.min(input.len()).min(output.len());
        if ch == 0 {
            return ResampleResult { consumed: 0, produced: 0 };
        }
        let n_in = input.iter().take(ch).map(|s| s.len()).min().unwrap_or(0);
        let n_out = output.iter().take(ch).map(|s| s.len()).min().unwrap_or(0);
        let (mut consumed, mut produced) = (0usize, 0usize);
        let taps = self.taps;

        loop {
            // 1. Produce everything the history allows.
            while produced < n_out {
                let c = self.pos.floor();
                let frac = self.pos - c;
                let c = c as i64;
                let exact = frac == 0.0 && self.base_step == 1.0 && self.trim == 0.0 && self.trim_target == 0.0;
                let last_needed = if exact { c } else { c + self.half as i64 };
                if last_needed >= self.h0 + self.len as i64 {
                    break;
                }
                if exact {
                    let idx = (c - self.h0) as usize;
                    for k in 0..ch {
                        output[k][produced] = self.hist[k * self.cap + idx];
                    }
                } else {
                    let start = (c - (self.half as i64 - 1) - self.h0) as usize;
                    let pf = frac * self.phases as f64;
                    let p = (pf as usize).min(self.phases - 1);
                    let a = (pf - p as f64) as f32;
                    let row0 = &self.table[p * taps..(p + 1) * taps];
                    let row1 = &self.table[(p + 1) * taps..(p + 2) * taps];
                    for k in 0..ch {
                        let x = &self.hist[k * self.cap + start..k * self.cap + start + taps];
                        let (s0, s1) = dot2(row0, row1, x);
                        output[k][produced] = s0 + (s1 - s0) * a;
                    }
                }
                self.pos += self.step();
                // Slew the trim toward its target, once per produced frame (chunking independent).
                if self.trim != self.trim_target {
                    let d = (self.trim_target - self.trim).clamp(-TRIM_SLEW_PER_SAMPLE, TRIM_SLEW_PER_SAMPLE);
                    self.trim += d;
                }
                produced += 1;
            }
            if produced == n_out || consumed == n_in {
                break;
            }
            // 2. Need more input: drop frames the window no longer reaches, then append.
            let keep_from = self.pos.floor() as i64 - (self.half as i64 - 1);
            let drop = (keep_from - self.h0).clamp(0, self.len as i64) as usize;
            if drop > 0 {
                for k in 0..self.channels {
                    let base = k * self.cap;
                    self.hist.copy_within(base + drop..base + self.len, base);
                }
                self.h0 += drop as i64;
                self.len -= drop;
            }
            let take = (self.cap - self.len).min(n_in - consumed);
            if take == 0 {
                break; // history cannot grow (cannot happen with a sane window); never spin
            }
            for k in 0..ch {
                let base = k * self.cap + self.len;
                self.hist[base..base + take].copy_from_slice(&input[k][consumed..consumed + take]);
            }
            self.len += take;
            consumed += take;
        }
        ResampleResult { consumed, produced }
    }
}

/// Two dot products of `x` with `a` and `b` (equal lengths), 4 independent accumulators each.
#[inline]
fn dot2(a: &[f32], b: &[f32], x: &[f32]) -> (f32, f32) {
    let n = x.len().min(a.len()).min(b.len());
    let (mut a0, mut a1, mut a2, mut a3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let (mut b0, mut b1, mut b2, mut b3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut i = 0;
    while i + 4 <= n {
        a0 += a[i] * x[i];
        a1 += a[i + 1] * x[i + 1];
        a2 += a[i + 2] * x[i + 2];
        a3 += a[i + 3] * x[i + 3];
        b0 += b[i] * x[i];
        b1 += b[i + 1] * x[i + 1];
        b2 += b[i + 2] * x[i + 2];
        b3 += b[i + 3] * x[i + 3];
        i += 4;
    }
    while i < n {
        a0 += a[i] * x[i];
        b0 += b[i] * x[i];
        i += 1;
    }
    ((a0 + a1) + (a2 + a3), (b0 + b1) + (b2 + b3))
}

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..200 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}
