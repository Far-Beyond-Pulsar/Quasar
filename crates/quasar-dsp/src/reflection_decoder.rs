//! Spatialised rendering of early-reflection taps (#58).
//!
//! One [`ReflectionDecoder`] serves one (listener, scene output) pair. It reads
//! its taps from the output's shared dry delay line
//! ([`EarlyReflectionDelayNode`](crate::early_reflections::EarlyReflectionDelayNode),
//! fed once per block with the un-attenuated dry signal) and renders **every tap at
//! its own listener-space arrival direction**, either through the listener's
//! [`VbapPanner`] (speaker layouts) or through one [`BinauralRenderer`] per tap
//! (HRTF listeners), instead of folding the reflections to a mono signal panned from
//! the direct path's azimuth.
//!
//! Per tap and block:
//!
//! * **Identity.** Targets are matched to persistent slots by nearest delay (a tap
//!   that moved more than [`MAX_TAP_SLEW`] samples per sample is a different
//!   reflection), so a tap's gain / delay / pan state follows the same reflection
//!   from block to block. A new reflection fades in from zero at its own delay, a
//!   vanished one fades out to zero at its last delay and direction; with
//!   [`REFLECTION_SLOTS`] slots busy, further new taps are dropped.
//! * **Delay and gain.** Delay and both band gains are ramped linearly per sample
//!   from the previous block's value to the target.
//! * **Per-band gain.** The 8-band tap gain is folded to two bands: the mean of
//!   bands 62.5..500 Hz (`gain_lo`) and of 1..8 kHz (`gain_hi`), applied around a
//!   one-pole crossover at [`CROSSOVER_HZ`]: `y = g_lo lp(x) + g_hi (x - lp(x))`.
//!   Equal gains reduce to a plain gain (no filtering). Coarser than the direct
//!   path's 8-band EQ, but one one-pole per tap instead of eight biquads; a wall
//!   with a lot of HF absorption therefore still darkens its reflection.
//! * **Pan.** Speaker layouts: constant-power VBAP gains of the tap's direction,
//!   ramped linearly per sample from the previous block's gains. HRTF: the tap's
//!   own parametric binaural renderer (which ramps its own delays and filters).
//!
//! Allocation-free after construction. Cost: one Hermite read, one one-pole and a
//! `channels`-wide gain ramp per tap per sample (about 1.5 k flops per tap-block
//! for 5.1); binaural slots cost about 10 x that.

use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use crate::binaural::{BinauralConfig, BinauralRenderer, ParametricBinauralRenderer};
use crate::crossfader::MAX_CROSSFADE_REFLECTIONS;
use crate::early_reflections::{EarlyReflectionDelayNode, MAX_TAP_SLEW};
use crate::vbap::VbapPanner;

/// Simultaneously active (rendering or fading) taps per decoder. Two full
/// 16-reflection sets (the union the crossfader holds during a fade) fit.
pub const REFLECTION_SLOTS: usize = 32;
/// Crossover (Hz) between the low and high tap gain: between the 500 Hz and 1 kHz bands.
pub const CROSSOVER_HZ: f32 = 707.0;
/// Gains below this (linear) count as silent (-180 dB).
const SILENT: f32 = 1e-9;

/// Listener-space target of one reflection tap, at the end of the block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TapTarget {
    /// Total path delay in samples.
    pub delay_samples: f32,
    /// Linear gain below the crossover (mean of the 62.5..500 Hz band gains).
    pub gain_lo: f32,
    /// Linear gain above the crossover (mean of the 1..8 kHz band gains).
    pub gain_hi: f32,
    /// Arrival azimuth in the LISTENER frame (radians, 0 = ahead, + = right).
    pub azimuth: f32,
    /// Arrival elevation in the listener frame (radians, + = up).
    pub elevation: f32,
}

struct Slot {
    active: bool,
    delay: f32,
    g_lo: f32,
    g_hi: f32,
    az: f32,
    el: f32,
    /// One-pole low-pass memory of the crossover.
    lp: f32,
    /// Speaker gains of the previous block (per-sample ramp start).
    spk: [f32; MAX_AUDIO_CHANNELS],
    spk_valid: bool,
    bin: Option<Box<dyn BinauralRenderer>>,
}

impl Slot {
    fn new(hrtf: bool, sample_rate: f32) -> Self {
        Self {
            active: false,
            delay: 0.0,
            g_lo: 0.0,
            g_hi: 0.0,
            az: 0.0,
            el: 0.0,
            lp: 0.0,
            spk: [0.0; MAX_AUDIO_CHANNELS],
            spk_valid: false,
            bin: if hrtf {
                Some(Box::new(ParametricBinauralRenderer::new(BinauralConfig::new(sample_rate))))
            } else {
                None
            },
        }
    }
}

/// Renders the early-reflection taps of one (listener, output) pair.
pub struct ReflectionDecoder {
    slots: Vec<Slot>,
    scratch: Vec<f32>,
    /// Per-block scratch: tap delays, raw tap reads, low-passed reads.
    dl: Vec<f32>,
    xs: Vec<f32>,
    lps: Vec<f32>,
    /// One-pole coefficient `1 - exp(-2 pi f / fs)` of the crossover.
    xover_a: f32,
}

impl ReflectionDecoder {
    /// Create a decoder. `hrtf` selects binaural rendering (the output buffer is
    /// then rendered onto channels 0 / 1) instead of the VBAP panner.
    /// Allocates (API thread only).
    pub fn new(sample_rate: f32, hrtf: bool) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 8000.0 { sample_rate } else { 48_000.0 };
        Self {
            slots: (0..REFLECTION_SLOTS).map(|_| Slot::new(hrtf, sr)).collect(),
            scratch: vec![0.0; DEFAULT_BLOCK_SIZE],
            dl: vec![0.0; DEFAULT_BLOCK_SIZE],
            xs: vec![0.0; DEFAULT_BLOCK_SIZE],
            lps: vec![0.0; DEFAULT_BLOCK_SIZE],
            xover_a: 1.0 - (-2.0 * std::f32::consts::PI * CROSSOVER_HZ / sr).exp(),
        }
    }

    /// Number of slots currently rendering or fading out.
    pub fn active_taps(&self) -> usize {
        self.slots.iter().filter(|s| s.active).count()
    }

    /// Forget all taps and filter / pan history (next taps fade in from silence).
    pub fn reset(&mut self) {
        for s in self.slots.iter_mut() {
            s.active = false;
            s.lp = 0.0;
            s.spk_valid = false;
            if let Some(b) = s.bin.as_mut() {
                b.reset();
            }
        }
    }

    /// Render this block's taps and ADD them onto `out`.
    ///
    /// `line` must already hold this block (call
    /// [`EarlyReflectionDelayNode::push_block`] first). `panner` is the listener's
    /// VBAP panner (`None` for HRTF decoders). `block` is the number of samples of
    /// this block (clamped to the buffer and [`DEFAULT_BLOCK_SIZE`]). Targets
    /// beyond [`MAX_CROSSFADE_REFLECTIONS`] are ignored.
    pub fn render_add(
        &mut self,
        line: &EarlyReflectionDelayNode,
        targets: &[TapTarget],
        panner: Option<&VbapPanner>,
        out: &mut AudioBuffer,
        block: usize,
    ) {
        let block = block.min(DEFAULT_BLOCK_SIZE).min(out.samples() as usize);
        if block == 0 {
            return;
        }
        let max_d = (line.max_tap_delay() - block as f32).max(0.0);
        let max_glide = MAX_TAP_SLEW * block as f32;
        let n_speakers = out.channels() as usize;

        // 1. Match targets to slots (nearest delay among unmatched active slots).
        let mut target_of = [usize::MAX; REFLECTION_SLOTS];
        let mut fresh = [false; REFLECTION_SLOTS];
        let nt = targets.len().min(MAX_CROSSFADE_REFLECTIONS);
        for (i, t) in targets.iter().take(nt).enumerate() {
            let td = if t.delay_samples.is_finite() { t.delay_samples.clamp(0.0, max_d) } else { 0.0 };
            let mut best = usize::MAX;
            let mut best_d = f32::INFINITY;
            for (s, slot) in self.slots.iter().enumerate() {
                if slot.active && target_of[s] == usize::MAX {
                    let d = (slot.delay - td).abs();
                    if d < best_d {
                        best_d = d;
                        best = s;
                    }
                }
            }
            if best != usize::MAX && best_d <= max_glide {
                target_of[best] = i;
                continue;
            }
            if !(t.gain_lo.abs() > SILENT || t.gain_hi.abs() > SILENT) {
                continue; // a silent new tap does not take a slot
            }
            // New reflection: first free slot.
            if let Some(s) = (0..REFLECTION_SLOTS).find(|&s| !self.slots[s].active && target_of[s] == usize::MAX) {
                target_of[s] = i;
                fresh[s] = true;
            }
        }

        // 2. Render every active slot.
        let xover_a = self.xover_a;
        let inv_n = 1.0 / block as f32;
        let mut spk_target = [0.0_f32; MAX_AUDIO_CHANNELS];
        for s in 0..REFLECTION_SLOTS {
            let slot = &mut self.slots[s];
            let ti = target_of[s];
            if ti != usize::MAX {
                if fresh[s] {
                    let t = &targets[ti];
                    slot.active = true;
                    slot.delay = if t.delay_samples.is_finite() { t.delay_samples.clamp(0.0, max_d) } else { 0.0 };
                    slot.g_lo = 0.0;
                    slot.g_hi = 0.0;
                    slot.az = t.azimuth;
                    slot.el = t.elevation;
                    slot.lp = 0.0;
                    slot.spk_valid = false;
                    if let Some(b) = slot.bin.as_mut() {
                        b.reset();
                    }
                }
            } else if !slot.active {
                continue;
            }

            // Ramp end values.
            let (d1, lo1, hi1, az, el) = if ti != usize::MAX {
                let t = &targets[ti];
                let d = if t.delay_samples.is_finite() { t.delay_samples.clamp(0.0, max_d) } else { slot.delay };
                let lo = if t.gain_lo.is_finite() { t.gain_lo } else { 0.0 };
                let hi = if t.gain_hi.is_finite() { t.gain_hi } else { 0.0 };
                let (a, e) = if t.azimuth.is_finite() && t.elevation.is_finite() {
                    (t.azimuth, t.elevation)
                } else {
                    (slot.az, slot.el)
                };
                (d, lo, hi, a, e)
            } else {
                (slot.delay, 0.0, 0.0, slot.az, slot.el) // fade out in place
            };
            let (d0, lo0, hi0) = (slot.delay, slot.g_lo, slot.g_hi);

            let silent = lo0.abs() <= SILENT && hi0.abs() <= SILENT && lo1.abs() <= SILENT && hi1.abs() <= SILENT;
            if !silent {
                // Tap read with per-sample delay / gain ramps and the two-band split, as three
                // passes over the block (delays + vectorised Hermite read, the one-pole recursion,
                // the band mix) so each pass is a tight loop of one kind.
                let line_dl = &mut self.dl[..block];
                for (j, d) in line_dl.iter_mut().enumerate() {
                    let t = (j + 1) as f32 * inv_n;
                    *d = (d0 + (d1 - d0) * t).clamp(0.0, max_d) + (block - 1 - j) as f32;
                }
                let xs = &mut self.xs[..block];
                line.tap_many_at(line_dl, xs);
                let mut lp = slot.lp;
                let lps = &mut self.lps[..block];
                for (l, &x) in lps.iter_mut().zip(xs.iter()) {
                    lp += xover_a * (x - lp);
                    if lp.abs() < 1e-24 {
                        lp = 0.0; // flush denormals
                    }
                    *l = lp;
                }
                slot.lp = lp;
                for (j, ((s, &x), &l)) in self.scratch[..block].iter_mut().zip(xs.iter()).zip(lps.iter()).enumerate() {
                    let t = (j + 1) as f32 * inv_n;
                    let g_lo = lo0 + (lo1 - lo0) * t;
                    let g_hi = hi0 + (hi1 - hi0) * t;
                    *s = g_lo * l + g_hi * (x - l);
                }

                // Decode at the tap's own direction.
                if let Some(bin) = slot.bin.as_mut() {
                    let (l, r) = out.stereo_mut();
                    let nb = block.min(l.len()).min(r.len());
                    bin.render_add(&self.scratch[..nb], az, el, &mut l[..nb], &mut r[..nb]);
                } else if let Some(p) = panner {
                    let n = p.num_outputs().min(MAX_AUDIO_CHANNELS);
                    p.gains(az, el, &mut spk_target[..n]);
                    if !slot.spk_valid {
                        slot.spk[..n].copy_from_slice(&spk_target[..n]);
                        slot.spk_valid = true;
                    }
                    for sp in 0..n {
                        let (g0, g1) = (slot.spk[sp], spk_target[sp]);
                        slot.spk[sp] = g1;
                        if sp >= n_speakers || (g0 == 0.0 && g1 == 0.0) {
                            continue;
                        }
                        let ch = out.channel_mut(sp as u16);
                        crate::vbap::ramp_add(&mut ch[..block], &self.scratch[..block], g0, g1);
                    }
                }
            }

            slot.delay = d1;
            slot.g_lo = lo1;
            slot.g_hi = hi1;
            slot.az = az;
            slot.el = el;
            if ti == usize::MAX {
                slot.active = false; // faded out this block
            }
        }
    }
}
