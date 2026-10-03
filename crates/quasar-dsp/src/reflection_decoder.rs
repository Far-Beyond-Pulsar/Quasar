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
//! * **Per-band gain.** All eight material bands remain independent. Seven one-pole
//!   split filters at [`CROSSOVER_HZ`] render the octave-band gains independently. Equal gains
//!   reduce to a plain gain.
//! * **Pan.** Speaker layouts: constant-power VBAP gains of the tap's direction,
//!   ramped linearly per sample from the previous block's gains. HRTF: the tap's
//!   own parametric binaural renderer (which ramps its own delays and filters).
//!
//! Allocation-free after construction. Cost: one Hermite read, seven one-pole
//! updates and an eight-band mix per tap per sample; binaural slots cost about 10 x
//! that.

use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use crate::binaural::{BinauralConfig, BinauralRenderer, ParametricBinauralRenderer};
use crate::crossfader::MAX_CROSSFADE_REFLECTIONS;
use crate::early_reflections::{EarlyReflectionDelayNode, MAX_TAP_SLEW};
use crate::vbap::VbapPanner;
use quasar_core::bands::FREQ_BAND_COUNT;

/// Simultaneously active (rendering or fading) taps per decoder. Two full
/// 16-reflection sets (the union the crossfader holds during a fade) fit.
pub const REFLECTION_SLOTS: usize = 32;
/// Crossover frequencies between four adjacent two-band gain ranges.
/// Each frequency is the geometric mean of the neighboring octave-band centers.
pub const CROSSOVER_HZ: [f32; 7] = [88.388, 176.777, 353.553, 707.107, 1414.214, 2828.427, 5656.854];
/// Number of reflection gain ranges, matching the direct path's octave bands.
pub const REFLECTION_GAIN_BANDS: usize = FREQ_BAND_COUNT;
/// Gains below this (linear) count as silent (-180 dB).
const SILENT: f32 = 1e-9;

/// Listener-space target of one reflection tap, at the end of the block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TapTarget {
    /// Total path delay in samples.
    pub delay_samples: f32,
    /// Linear gains for the eight octave bands, matching the direct path.
    pub gains: [f32; REFLECTION_GAIN_BANDS],
    /// Arrival azimuth in the LISTENER frame (radians, 0 = ahead, + = right).
    pub azimuth: f32,
    /// Arrival elevation in the listener frame (radians, + = up).
    pub elevation: f32,
}

#[inline]
fn split_eight_bands(x: f32, state: &mut [f32; 7], coeffs: [f32; 7]) -> [f32; 8] {
    for i in 0..7 {
        state[i] += coeffs[i] * (x - state[i]);
    }
    for s in state.iter_mut() {
        if s.abs() < 1e-24 {
            *s = 0.0;
        }
    }
    [state[0], state[1] - state[0], state[2] - state[1], state[3] - state[2], state[4] - state[3], state[5] - state[4], state[6] - state[5], x - state[6]]
}

struct Slot {
    active: bool,
    delay: f32,
    gains: [f32; REFLECTION_GAIN_BANDS],
    az: f32,
    el: f32,
    /// Low-pass memories at the seven crossover frequencies.
    lp: [f32; 7],
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
            gains: [0.0; REFLECTION_GAIN_BANDS],
            az: 0.0,
            el: 0.0,
            lp: [0.0; 7],
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
    /// Per-block scratch: tap delays, raw reads and seven low-passed splits.
    dl: Vec<f32>,
    xs: Vec<f32>,
    lps: [Vec<f32>; 7],
    /// One-pole coefficients `1 - exp(-2 pi f / fs)` at the seven splits.
    xover_a: [f32; 7],
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
            lps: std::array::from_fn(|_| vec![0.0; DEFAULT_BLOCK_SIZE]),
            xover_a: CROSSOVER_HZ.map(|f| 1.0 - (-2.0 * std::f32::consts::PI * f / sr).exp()),
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
            s.lp = [0.0; 7];
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
    /// this block (clamped to the buffer and [`DEFAULT_BLOCK_SIZE`]). If more than
    /// [`MAX_CROSSFADE_REFLECTIONS`] targets arrive, the strongest by eight-band
    /// squared gain are selected; ties preserve input order. Replaced taps fade
    /// out through the normal slot lifecycle.
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
        // Select the strongest targets without allocating on the render thread.
        // Keep input order for equal energies so the cap remains deterministic.
        let mut selected = [usize::MAX; MAX_CROSSFADE_REFLECTIONS];
        let mut selected_energy = [f32::NEG_INFINITY; MAX_CROSSFADE_REFLECTIONS];
        let mut nt = 0;
        for (i, t) in targets.iter().enumerate() {
            let energy = t.gains.iter().filter(|g| g.is_finite()).map(|g| g * g).sum::<f32>();
            let pos = (0..nt).find(|&j| energy > selected_energy[j]).unwrap_or(nt);
            if pos >= MAX_CROSSFADE_REFLECTIONS {
                continue;
            }
            if pos < nt {
                let end = nt.min(MAX_CROSSFADE_REFLECTIONS - 1);
                for j in (pos + 1..=end).rev() {
                    selected[j] = selected[j - 1];
                    selected_energy[j] = selected_energy[j - 1];
                }
            }
            selected[pos] = i;
            selected_energy[pos] = energy;
            nt = (nt + 1).min(MAX_CROSSFADE_REFLECTIONS);
        }
        for (i, &target_index) in selected[..nt].iter().enumerate() {
            let t = &targets[target_index];
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
            if !t.gains.iter().any(|g| g.is_finite() && g.abs() > SILENT) {
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
                    let t = &targets[selected[ti]];
                    slot.active = true;
                    slot.delay = if t.delay_samples.is_finite() { t.delay_samples.clamp(0.0, max_d) } else { 0.0 };
                    slot.gains = [0.0; REFLECTION_GAIN_BANDS];
                    slot.az = t.azimuth;
                    slot.el = t.elevation;
                    slot.lp = [0.0; 7];
                    slot.spk_valid = false;
                    if let Some(b) = slot.bin.as_mut() {
                        b.reset();
                    }
                }
            } else if !slot.active {
                continue;
            }

            // Ramp end values.
            let (d1, gains1, az, el) = if ti != usize::MAX {
                let t = &targets[selected[ti]];
                let d = if t.delay_samples.is_finite() { t.delay_samples.clamp(0.0, max_d) } else { slot.delay };
                let gains = t.gains.map(|g| if g.is_finite() { g } else { 0.0 });
                let (a, e) = if t.azimuth.is_finite() && t.elevation.is_finite() {
                    (t.azimuth, t.elevation)
                } else {
                    (slot.az, slot.el)
                };
                (d, gains, a, e)
            } else {
                (slot.delay, [0.0; REFLECTION_GAIN_BANDS], slot.az, slot.el) // fade out in place
            };
            let d0 = slot.delay;
            let gains0 = slot.gains;

            let silent = gains0.iter().chain(gains1.iter()).all(|g| g.abs() <= SILENT);
            if !silent {
                // Tap read with per-sample delay / gain ramps and the eight-band split, as three
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
                for j in 0..block {
                    split_eight_bands(xs[j], &mut lp, xover_a);
                    for band in 0..7 { self.lps[band][j] = lp[band]; }
                }
                slot.lp = lp;
                for (j, s) in self.scratch[..block].iter_mut().enumerate() {
                    let t = (j + 1) as f32 * inv_n;
                    let g = std::array::from_fn::<_, REFLECTION_GAIN_BANDS, _>(|b| gains0[b] + (gains1[b] - gains0[b]) * t);
                    let mut previous = 0.0;
                    let mut y = 0.0;
                    for band in 0..7 {
                        let cumulative = self.lps[band][j];
                        y += g[band] * (cumulative - previous);
                        previous = cumulative;
                    }
                    y += g[7] * (xs[j] - previous);
                    *s = y;
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
            slot.gains = gains1;
            slot.az = az;
            slot.el = el;
            if ti == usize::MAX {
                slot.active = false; // faded out this block
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shaped_rms(freq: f32, gains: [f32; 8]) -> f32 {
        let sr = 48_000.0;
        let coeffs = CROSSOVER_HZ.map(|f| 1.0 - (-2.0 * std::f32::consts::PI * f / sr).exp());
        let mut state = [0.0; 7];
        let mut sum = 0.0;
        let mut count = 0;
        for n in 0..48_000 {
            let x = (2.0 * std::f32::consts::PI * freq * n as f32 / sr).sin();
            let b = split_eight_bands(x, &mut state, coeffs);
            let y = b.iter().zip(gains).map(|(x, g)| x * g).sum::<f32>();
            if n > 24_000 {
                sum += y * y;
                count += 1;
            }
        }
        (sum / count as f32).sqrt()
    }

    #[test]
    fn eight_band_split_preserves_material_gains_at_octave_centres() {
        let gains = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9];
        for (band, freq) in quasar_core::bands::FREQ_BAND_CENTRES.into_iter().enumerate() {
            let measured = shaped_rms(freq, gains);
            let expected = gains[band] / std::f32::consts::SQRT_2;
            assert!((measured - expected).abs() < 0.12, "{freq} Hz band {band}: RMS {measured}, expected near {expected}");
        }
    }
}
