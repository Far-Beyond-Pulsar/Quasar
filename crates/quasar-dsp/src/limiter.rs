//! Output safety stage (#80): gain staging, look-ahead peak limiter, NaN / inf scrub, denormal
//! control and metering for one listener's output bus.
//!
//! ```text
//! in -> x pre-gain (headroom) -> scrub non-finite -> [look-ahead delay] -> x limiter gain -> hard clamp -> out
//!                                      |                                        ^
//!                                      +--> peak detect (linked over channels) -+  (min over the look-ahead
//!                                                                                 window, smoothed attack,
//!                                                                                 one-pole release)
//! ```
//!
//! * **Ceiling.** The output never exceeds `ceiling_db` (default -1 dBFS) in sample-peak terms: the
//!   gain for each sample is the minimum, over the look-ahead window, of the gain that would bring
//!   that sample under the ceiling, smoothed by a moving average of the same length (so the gain
//!   reaches the dip before the peak arrives) and released with a one-pole of `release_ms`. A final
//!   hard clamp backs this up against rounding and is counted in
//!   [`OutputMeter::hard_clipped_samples`] (it should stay 0 in normal operation).
//!   **This is a SAMPLE-peak limiter, not a true-peak one**: it does not oversample, so inter-sample
//!   peaks of the reconstructed signal can exceed the ceiling by up to a few dB in the worst case.
//!   The default -1 dBFS ceiling leaves ~1 dB of margin for that; lower it (e.g. -2 .. -3 dBFS)
//!   when the output feeds a codec or sample-rate converter.
//! * **Gain is linked across channels** (the loudest channel drives it) so the stereo / surround
//!   image does not move while limiting.
//! * **Latency.** A look-ahead of `L` samples delays the signal by exactly `L` samples
//!   ([`OutputSafetyConfig::latency_samples`]; `OutputSafety::latency_samples`). With
//!   `lookahead_ms = 0` (the engine default) the stage has ZERO latency and an instant attack: the
//!   ceiling is still never exceeded, but the gain steps down within one sample at an overshoot
//!   (some distortion on the peak). Use ~1 ms look-ahead for a smooth, transparent-sounding limiter.
//! * **Transparent below the ceiling.** While every sample is under the ceiling the gain is exactly
//!   1.0, so the output equals the (delayed) input bit for bit (with the default 0 dB headroom).
//! * **NaN / inf scrub.** A non-finite input sample becomes 0.0 and increments
//!   [`OutputMeter::nonfinite_samples`]; no NaN or infinity ever reaches the output or the limiter state.
//! * **Denormals.** [`set_flush_to_zero`] sets FTZ / DAZ on the calling thread (x86_64 and aarch64);
//!   the limiter's own state is also snapped to exact values (gain snaps back to 1.0), so it never
//!   produces denormals itself. Other stages with feedback (reverb) rely on FTZ.
//! * **Metering.** [`OutputMeter`] holds lock-free atomics (peak, current gain reduction,
//!   limited-sample count, hard-clip count, non-finite count) readable from any thread.
//!
//! Memory is allocated in [`OutputSafety::new`] (all [`MAX_AUDIO_CHANNELS`] channels x
//! [`MAX_LOOKAHEAD_SAMPLES`]); `process` never allocates.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::audio_buffer::{AudioBuffer, MAX_AUDIO_CHANNELS};

/// Largest supported look-ahead (samples); `lookahead_ms` is clamped to it.
pub const MAX_LOOKAHEAD_SAMPLES: usize = 512;

/// Configuration of the output safety stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputSafetyConfig {
    /// Limiter on/off. When off the stage still scrubs non-finite samples and meters, with zero
    /// latency.
    pub enabled: bool,
    /// Pre-limiter gain in dB (negative = headroom). Default 0.
    pub headroom_db: f32,
    /// Output ceiling in dBFS (sample peak). Default -1.
    pub ceiling_db: f32,
    /// Look-ahead in milliseconds (= attack time); 0 = zero latency / instant attack (default).
    pub lookahead_ms: f32,
    /// Release time constant in milliseconds. Default 80.
    pub release_ms: f32,
}

impl Default for OutputSafetyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            headroom_db: 0.0,
            ceiling_db: -1.0,
            lookahead_ms: 0.0,
            release_ms: 80.0,
        }
    }
}

impl OutputSafetyConfig {
    /// Look-ahead in samples at `sample_rate` (0 when the limiter is disabled).
    pub fn lookahead_samples(&self, sample_rate: f32) -> usize {
        if !self.enabled || !self.lookahead_ms.is_finite() || !sample_rate.is_finite() {
            return 0;
        }
        ((self.lookahead_ms.max(0.0) * 0.001 * sample_rate).round() as usize).min(MAX_LOOKAHEAD_SAMPLES)
    }

    /// Latency this configuration adds to the output, in samples at `sample_rate`: exactly the
    /// look-ahead.
    pub fn latency_samples(&self, sample_rate: f32) -> usize {
        self.lookahead_samples(sample_rate)
    }
}

/// Lock-free meters of one output stage. Share the `Arc` with any reader thread.
#[derive(Debug)]
pub struct OutputMeter {
    /// Peak |sample| at the output since the last [`reset`](Self::reset) (f32 bits; positive
    /// floats order like their bits, so `fetch_max` is exact).
    peak_bits: AtomicU32,
    /// Gain applied by the limiter at the end of the latest block (f32 bits; 1.0 = no reduction).
    gain_bits: AtomicU32,
    /// Samples at which the limiter was reducing gain.
    limited: AtomicU64,
    /// Samples the final hard clamp had to touch (should stay 0).
    hard_clipped: AtomicU64,
    /// Non-finite input samples replaced by 0.
    nonfinite: AtomicU64,
}

impl OutputMeter {
    pub fn new() -> Self {
        Self {
            peak_bits: AtomicU32::new(0),
            gain_bits: AtomicU32::new(1.0_f32.to_bits()),
            limited: AtomicU64::new(0),
            hard_clipped: AtomicU64::new(0),
            nonfinite: AtomicU64::new(0),
        }
    }

    /// Peak output level (linear) since the last reset.
    pub fn peak(&self) -> f32 {
        f32::from_bits(self.peak_bits.load(Ordering::Relaxed))
    }

    /// Peak output level in dBFS (`-inf` for silence).
    pub fn peak_db(&self) -> f32 {
        20.0 * self.peak().log10()
    }

    /// Limiter gain at the end of the latest block (linear, 1.0 = not limiting).
    pub fn gain(&self) -> f32 {
        f32::from_bits(self.gain_bits.load(Ordering::Relaxed))
    }

    /// Current gain reduction in dB (>= 0).
    pub fn gain_reduction_db(&self) -> f32 {
        -20.0 * self.gain().max(1e-9).log10()
    }

    /// Samples during which the limiter reduced the gain.
    pub fn limited_samples(&self) -> u64 {
        self.limited.load(Ordering::Relaxed)
    }

    /// Samples the final hard clamp had to touch (clip meter; stays 0 while the limiter holds).
    pub fn hard_clipped_samples(&self) -> u64 {
        self.hard_clipped.load(Ordering::Relaxed)
    }

    /// Non-finite (NaN / inf) input samples that were replaced by silence (error counter).
    pub fn nonfinite_samples(&self) -> u64 {
        self.nonfinite.load(Ordering::Relaxed)
    }

    /// Reset the peak and the counters.
    pub fn reset(&self) {
        self.peak_bits.store(0, Ordering::Relaxed);
        self.limited.store(0, Ordering::Relaxed);
        self.hard_clipped.store(0, Ordering::Relaxed);
        self.nonfinite.store(0, Ordering::Relaxed);
    }
}

impl Default for OutputMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// The output safety stage of one listener bus.
pub struct OutputSafety {
    cfg: OutputSafetyConfig,
    sample_rate: f32,
    meter: Arc<OutputMeter>,
    pre_gain: f32,
    ceiling: f32,
    /// Look-ahead in samples (0 = none).
    n: usize,
    release_coef: f32,
    /// Per channel look-ahead delay ring, `MAX_LOOKAHEAD_SAMPLES` each; `delay_pos` is shared.
    delay: Vec<f32>,
    delay_pos: usize,
    /// Monotonic deque (indices, values) for the sliding minimum of the required gain.
    dq_idx: Vec<u64>,
    dq_val: Vec<f32>,
    dq_head: usize,
    dq_len: usize,
    /// Ring of the sliding-min values with their running sum (moving average, `n + 1` taps).
    ma_ring: Vec<f32>,
    ma_sum: f64,
    ma_pos: usize,
    /// Sample counter (monotonic) for the deque.
    t: u64,
    /// Smoothed gain state kept as the gap to unity (`gain = 1 - gap`): the release recursion on the
    /// gap decays all the way to exactly 0, where a recursion on the gain itself would stall a
    /// few ulps below 1.0.
    gap: f32,
}

const DQ_CAP: usize = MAX_LOOKAHEAD_SAMPLES + 2;

impl OutputSafety {
    /// Create the stage (allocates all state).
    pub fn new(sample_rate: f32, cfg: OutputSafetyConfig) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 { sample_rate } else { 48_000.0 };
        let mut s = Self {
            cfg,
            sample_rate: sr,
            meter: Arc::new(OutputMeter::new()),
            pre_gain: 1.0,
            ceiling: 0.891_25,
            n: 0,
            release_coef: 0.0,
            delay: vec![0.0; MAX_AUDIO_CHANNELS * MAX_LOOKAHEAD_SAMPLES],
            delay_pos: 0,
            dq_idx: vec![0; DQ_CAP],
            dq_val: vec![1.0; DQ_CAP],
            dq_head: 0,
            dq_len: 0,
            ma_ring: vec![1.0; MAX_LOOKAHEAD_SAMPLES + 1],
            ma_sum: 1.0,
            ma_pos: 0,
            t: 0,
            gap: 0.0,
        };
        s.set_config(cfg);
        s
    }

    /// The shared meters.
    pub fn meter(&self) -> &Arc<OutputMeter> {
        &self.meter
    }

    /// Replace the meters `Arc` (used to hand the renderer's meter to the compute side).
    pub fn set_meter(&mut self, meter: Arc<OutputMeter>) {
        self.meter = meter;
    }

    /// Current configuration.
    pub fn config(&self) -> OutputSafetyConfig {
        self.cfg
    }

    /// Latency (samples) this stage adds: the look-ahead.
    pub fn latency_samples(&self) -> usize {
        self.n
    }

    /// Reconfigure in place (no allocation). The limiter state is reset if the look-ahead
    /// changed (a click is possible at that moment).
    pub fn set_config(&mut self, cfg: OutputSafetyConfig) {
        let db = |v: f32, default: f32| if v.is_finite() { v } else { default };
        let cfg = OutputSafetyConfig {
            headroom_db: db(cfg.headroom_db, 0.0).clamp(-60.0, 24.0),
            ceiling_db: db(cfg.ceiling_db, -1.0).clamp(-60.0, 0.0),
            lookahead_ms: db(cfg.lookahead_ms, 0.0).clamp(0.0, 1000.0),
            release_ms: db(cfg.release_ms, 80.0).clamp(0.1, 10_000.0),
            ..cfg
        };
        let n = cfg.lookahead_samples(self.sample_rate);
        let changed = n != self.n;
        self.cfg = cfg;
        self.n = n;
        self.pre_gain = 10f32.powf(cfg.headroom_db / 20.0);
        self.ceiling = 10f32.powf(cfg.ceiling_db / 20.0);
        // One-pole release: the gap to unity shrinks by this fraction per sample.
        self.release_coef = 1.0 - (-1.0 / (cfg.release_ms * 0.001 * self.sample_rate)).exp();
        if changed {
            self.reset();
        }
    }

    /// Clear the delay / limiter state (not the meters).
    pub fn reset(&mut self) {
        self.delay.fill(0.0);
        self.delay_pos = 0;
        self.dq_head = 0;
        self.dq_len = 0;
        self.ma_ring.fill(1.0);
        self.ma_sum = (self.n + 1) as f64;
        self.ma_pos = 0;
        self.t = 0;
        self.gap = 0.0;
    }

    /// Process `buf` in place (all its channels, linked). Never allocates, locks or panics.
    pub fn process(&mut self, buf: &mut AudioBuffer) {
        let channels = (buf.channels() as usize).min(MAX_AUDIO_CHANNELS);
        let samples = buf.samples() as usize;
        let n = self.n;
        let ceiling = self.ceiling;
        let pre = self.pre_gain;
        let enabled = self.cfg.enabled;
        let mut nonfinite = 0u64;
        let mut limited = 0u64;
        let mut hard = 0u64;
        let mut block_peak = 0.0_f32;

        for i in 0..samples {
            // 1. Gain staging + scrub, per channel; linked peak.
            let mut peak = 0.0_f32;
            for c in 0..channels {
                let mut x = buf.channel(c as u16)[i];
                if pre != 1.0 {
                    x *= pre;
                }
                if !x.is_finite() {
                    x = 0.0;
                    nonfinite += 1;
                }
                buf.channel_mut(c as u16)[i] = x;
                peak = peak.max(x.abs());
            }
            if !enabled {
                block_peak = block_peak.max(peak);
                continue;
            }

            // 2. Required gain -> sliding min over n+1 -> moving average (attack) -> release.
            let g_req = if peak > ceiling { ceiling / peak } else { 1.0 };
            let a = if n == 0 {
                g_req
            } else {
                let w = self.sliding_min(g_req, n);
                // Moving average over n + 1 taps via a running sum.
                let old = self.ma_ring[self.ma_pos];
                self.ma_ring[self.ma_pos] = w;
                self.ma_pos = if self.ma_pos == n { 0 } else { self.ma_pos + 1 };
                self.ma_sum += w as f64 - old as f64;
                ((self.ma_sum / (n + 1) as f64) as f32).min(1.0)
            };
            let gap_a = 1.0 - a;
            if gap_a > self.gap {
                self.gap = gap_a; // attack: follow the (already smoothed) requirement at once
            } else {
                self.gap = gap_a + (self.gap - gap_a) * (1.0 - self.release_coef);
                if self.gap < 1e-7 && gap_a == 0.0 {
                    self.gap = 0.0; // snap: exact transparency, no denormal tail
                }
            }
            let g = 1.0 - self.gap;
            if g < 1.0 {
                limited += 1;
            }

            // 3. Delay line (look-ahead) and apply.
            let mut out_peak = 0.0_f32;
            for c in 0..channels {
                let x = buf.channel(c as u16)[i];
                let y = if n == 0 {
                    x
                } else {
                    let slot = &mut self.delay[c * MAX_LOOKAHEAD_SAMPLES + self.delay_pos];
                    let y = *slot;
                    *slot = x;
                    y
                };
                let mut o = if g == 1.0 { y } else { y * g };
                if o.abs() > ceiling {
                    // Rounding (or a reconfiguration): clamp, and count real overshoots.
                    if o.abs() > ceiling * 1.0001 {
                        hard += 1;
                    }
                    o = o.clamp(-ceiling, ceiling);
                }
                buf.channel_mut(c as u16)[i] = o;
                out_peak = out_peak.max(o.abs());
            }
            if n > 0 {
                self.delay_pos = if self.delay_pos + 1 == n { 0 } else { self.delay_pos + 1 };
            }
            block_peak = block_peak.max(out_peak);
        }

        // Meters.
        let m = &*self.meter;
        if block_peak > 0.0 {
            m.peak_bits.fetch_max(block_peak.to_bits(), Ordering::Relaxed);
        }
        m.gain_bits.store((1.0 - self.gap).to_bits(), Ordering::Relaxed);
        if limited > 0 {
            m.limited.fetch_add(limited, Ordering::Relaxed);
        }
        if hard > 0 {
            m.hard_clipped.fetch_add(hard, Ordering::Relaxed);
        }
        if nonfinite > 0 {
            m.nonfinite.fetch_add(nonfinite, Ordering::Relaxed);
        }
    }

    /// Push `v` and return the minimum of the last `n + 1` pushed values (monotonic deque).
    fn sliding_min(&mut self, v: f32, n: usize) -> f32 {
        let t = self.t;
        self.t += 1;
        // Drop values that can never be the minimum again.
        while self.dq_len > 0 {
            let last = (self.dq_head + self.dq_len - 1) % DQ_CAP;
            if self.dq_val[last] >= v {
                self.dq_len -= 1;
            } else {
                break;
            }
        }
        let tail = (self.dq_head + self.dq_len) % DQ_CAP;
        self.dq_idx[tail] = t;
        self.dq_val[tail] = v;
        self.dq_len += 1;
        // Drop values that left the window.
        while self.dq_len > 0 && self.dq_idx[self.dq_head] + n as u64 + 1 <= t {
            self.dq_head = (self.dq_head + 1) % DQ_CAP;
            self.dq_len -= 1;
        }
        self.dq_val[self.dq_head]
    }
}

/// Enable flush-to-zero / denormals-are-zero on the CALLING thread, so denormal floats (slow on
/// many CPUs, e.g. decaying reverb tails) are treated as 0. Per-thread: call it at the start of the
/// audio thread (the renderer does so on its first block). Returns `true` if it could be set
/// (x86_64 via MXCSR FTZ|DAZ, aarch64 via FPCR.FZ); `false` where unsupported.
#[allow(unused_unsafe)]
pub fn set_flush_to_zero() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let mut csr: u32 = 0;
        // SAFETY: reads / writes the SSE control register of this thread only; setting the
        // FTZ (bit 15) and DAZ (bit 6) flags is always valid.
        unsafe {
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags));
            csr |= 0x8040;
            core::arch::asm!("ldmxcsr [{}]", in(reg) &csr, options(nostack, readonly));
        }
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: sets FPCR.FZ (bit 24) of this thread only.
        unsafe {
            let mut fpcr: u64;
            core::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack));
            fpcr |= 1 << 24;
            core::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nomem, nostack));
        }
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// Whether FTZ is currently set on the calling thread (x86_64 / aarch64); `None` if unsupported.
pub fn flush_to_zero_enabled() -> Option<bool> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut csr: u32 = 0;
        // SAFETY: read-only access to this thread's MXCSR.
        unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags)) };
        return Some(csr & 0x8040 == 0x8040);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let fpcr: u64;
        // SAFETY: read-only access to this thread's FPCR.
        unsafe { core::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack)) };
        return Some(fpcr & (1 << 24) != 0);
    }
    #[allow(unreachable_code)]
    None
}
