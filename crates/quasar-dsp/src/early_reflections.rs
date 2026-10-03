use quasar_core::param_exchange::{SpatialCoefficients, EarlyReflectionCoeffs};
use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use crate::fractional_delay::HermiteInterpolatingDelayLine;
use crate::node_graph::AudioNode;

/// A single early reflection tap.
#[derive(Clone, Copy)]
struct ReflectionTap {
    delay_samples: f32,
    gain: f32,
}

/// One tap's per-sample trajectory across the current block (`*0` = value at the
/// end of the previous block, `*1` = target at the end of this block).
#[derive(Clone, Copy)]
struct RampTap {
    d0: f32,
    d1: f32,
    g0: f32,
    g1: f32,
}

/// Largest per-sample delay change (samples per sample) a tap may glide at when it
/// is matched to the same tap of the previous block. A tap that moved further than
/// this over one block is treated as a different reflection (the old one fades out
/// at its old delay, the new one fades in at its new delay): a crossfade instead of
/// a pitch glide.
pub const MAX_TAP_SLEW: f32 = 0.25;

/// Multi-tap early reflection processor.
///
/// Owns a shared dry delay line with multiple tapped outputs, each representing
/// one specular early reflection path with its own delay.
///
/// Two ways to use it:
///
/// * **Engine path (spatialised).** [`Self::push_block`] feeds the line once per
///   block with the un-attenuated dry signal; one
///   [`ReflectionDecoder`](crate::reflection_decoder::ReflectionDecoder) per
///   (listener, output) then reads its taps with [`Self::tap_at`] and renders each
///   at its own arrival direction (#58).
/// * **Mono fold ([`AudioNode::process`]).** The legacy path, kept for standalone
///   use and tests: the taps are summed into one mono channel, each tap's gain being
///   the average per-band reflection gain; the pan angle is ignored.
///
/// Tap gains and delays are ramped linearly per sample from their value at the end
/// of the previous block to the new target (taps are matched between blocks by
/// nearest delay), so block-rate parameter updates neither zipper nor step the
/// read position. Taps that appear fade in from zero, taps that disappear fade
/// out; the first block after construction / reset starts at its target.
pub struct EarlyReflectionDelayNode {
    delay_line: HermiteInterpolatingDelayLine,
    /// Targets set by [`Self::update_reflections`].
    taps: Vec<ReflectionTap>,
    /// Targets rendered by the previous block (ramp start values).
    prev: Vec<ReflectionTap>,
    /// Scratch: per-block ramp list (matched + fading in + fading out).
    ramps: Vec<RampTap>,
    primed: bool,
    input_channels: u16,
    output_channels: u16,
    _sample_rate: f32,
    _max_delay_secs: f32,
}

impl EarlyReflectionDelayNode {
    /// Create a new early reflection delay node.
    ///
    /// `max_reflections`: maximum number of early reflection taps to support.
    pub fn new(input_channels: u16, sample_rate: f32, max_delay_secs: f32, max_reflections: usize) -> Self {
        let delay_line = HermiteInterpolatingDelayLine::new(max_delay_secs, sample_rate);
        let cap = max_reflections.max(crate::crossfader::MAX_CROSSFADE_REFLECTIONS);
        Self {
            delay_line,
            taps: Vec::with_capacity(cap),
            prev: Vec::with_capacity(cap),
            ramps: Vec::with_capacity(2 * cap),
            primed: false,
            input_channels,
            output_channels: 1,
            _sample_rate: sample_rate,
            _max_delay_secs: max_delay_secs,
        }
    }

    /// Push one block of the dry signal (channels averaged to mono) into the
    /// shared line, WITHOUT reading any tap. Used with [`Self::tap_at`] by the
    /// per-listener decoders; do not mix with [`AudioNode::process`] on the same
    /// block (both push). Allocation-free.
    pub fn push_block(&mut self, input: &AudioBuffer) {
        let n = input.samples() as usize;
        let ch = input.channels() as usize;
        if ch == 0 {
            for _ in 0..n {
                self.delay_line.push(0.0);
            }
            return;
        }
        let inv = 1.0 / ch as f32;
        for i in 0..n {
            let mut mono = 0.0_f32;
            for c in 0..ch {
                mono += input.channel(c as u16)[i];
            }
            self.delay_line.push(mono * inv);
        }
    }

    /// Sample `delay_from_newest` samples (fractional, Hermite) before the newest
    /// pushed sample: 0 = the last sample of the last [`Self::push_block`].
    /// Clamped to `[0, max_tap_delay()]`.
    #[inline]
    pub fn tap_at(&self, delay_from_newest: f32) -> f32 {
        self.delay_line.tap(delay_from_newest)
    }

    /// Read a block of taps: `out[j] = tap_at(delays[j])` (same arithmetic, vectorisable layout).
    #[inline]
    pub fn tap_many_at(&self, delays: &[f32], out: &mut [f32]) {
        self.delay_line.tap_many(delays, 0, false, out);
    }

    /// Largest delay (samples) [`Self::tap_at`] can read.
    pub fn max_tap_delay(&self) -> f32 {
        (self.delay_line.max_samples() - 3) as f32
    }

    /// Update reflection taps from spatial coefficients.
    pub fn update_reflections(&mut self, reflections: &[EarlyReflectionCoeffs]) {
        // Called every block from the audio thread: never grow the Vec (the
        // crossfader's in-flight set can hold the union of two reflection
        // lists, up to `MAX_CROSSFADE_REFLECTIONS`), and never read past the
        // delay line (release builds wrap into garbage instead of asserting).
        self.taps.clear();
        let max_delay = (self.delay_line.max_samples() - 3) as f32;
        for r in reflections.iter().take(self.taps.capacity()) {
            // Mono early-reflection contribution; pan (azimuth) is spatialized in P3.
            let avg_gain = r.gain.0.iter().sum::<f32>() / 8.0;
            let gain = if avg_gain.is_finite() { avg_gain } else { 0.0 };
            let delay = if r.delay_samples.is_finite() { r.delay_samples.clamp(0.0, max_delay) } else { 0.0 };
            self.taps.push(ReflectionTap { delay_samples: delay, gain });
        }
    }

    /// Build this block's ramp list by matching the new targets to the previous
    /// block's taps (greedy, nearest delay), then remember the targets.
    /// Allocation-free: only pushes within the preallocated capacity.
    fn build_ramps(&mut self, block: usize) {
        self.ramps.clear();
        let cap = self.ramps.capacity();
        if !self.primed {
            for t in &self.taps {
                if self.ramps.len() < cap {
                    self.ramps.push(RampTap { d0: t.delay_samples, d1: t.delay_samples, g0: t.gain, g1: t.gain });
                }
            }
            self.primed = true;
        } else {
            let max_glide = MAX_TAP_SLEW * block as f32;
            let mut used = [false; crate::crossfader::MAX_CROSSFADE_REFLECTIONS];
            for t in &self.taps {
                let mut best: Option<usize> = None;
                let mut best_d = f32::INFINITY;
                for (i, p) in self.prev.iter().enumerate().take(used.len()) {
                    if used[i] {
                        continue;
                    }
                    let d = (p.delay_samples - t.delay_samples).abs();
                    if d < best_d {
                        best_d = d;
                        best = Some(i);
                    }
                }
                let ramp = match best {
                    Some(i) if best_d <= max_glide => {
                        used[i] = true;
                        let p = self.prev[i];
                        RampTap { d0: p.delay_samples, d1: t.delay_samples, g0: p.gain, g1: t.gain }
                    }
                    // New reflection: fade in at its own delay.
                    _ => RampTap { d0: t.delay_samples, d1: t.delay_samples, g0: 0.0, g1: t.gain },
                };
                if self.ramps.len() < cap {
                    self.ramps.push(ramp);
                }
            }
            // Vanished reflections: fade out at their last delay.
            for (i, p) in self.prev.iter().enumerate() {
                if i < used.len() && used[i] {
                    continue;
                }
                if p.gain != 0.0 && self.ramps.len() < cap {
                    self.ramps.push(RampTap { d0: p.delay_samples, d1: p.delay_samples, g0: p.gain, g1: 0.0 });
                }
            }
        }
        self.prev.clear();
        self.prev.extend_from_slice(&self.taps);
    }
}

impl AudioNode for EarlyReflectionDelayNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, _params: &SpatialCoefficients) {
        debug_assert!(input.channels() >= 1);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;
        let inv_n = 1.0 / num_samples.max(1) as f32;
        self.build_ramps(num_samples);

        // Fold to mono and push the whole block first; each tap is then read per sample
        // counted back from the end of the block (`tap_back`), which is exactly what a read right
        // after the push of sample `i` returns, and lets the loop run tap-major (constants of
        // one tap hoisted, the line read in a tight loop).
        let mut mono = [0.0_f32; DEFAULT_BLOCK_SIZE];
        let n = num_samples.min(DEFAULT_BLOCK_SIZE);
        let chs = input.channels() as usize;
        for c in 0..chs {
            let ch = &input.channel(c as u16)[..n];
            for (m, x) in mono[..n].iter_mut().zip(ch) {
                *m += *x;
            }
        }
        let chf = input.channels() as f32;
        for m in mono[..n].iter_mut() {
            *m /= chf;
        }
        self.delay_line.push_slice(&mono[..n]);

        output.channel_mut(0).fill(0.0);
        let out = &mut output.channel_mut(0)[..n];
        let mut dly = [0.0_f32; DEFAULT_BLOCK_SIZE];
        let mut rd = [0.0_f32; DEFAULT_BLOCK_SIZE];
        for r in &self.ramps {
            let (d0, dd, g0, dg) = (r.d0, r.d1 - r.d0, r.g0, r.g1 - r.g0);
            for (i, d) in dly[..n].iter_mut().enumerate() {
                let t = (i + 1) as f32 * inv_n;
                *d = d0 + dd * t;
            }
            self.delay_line.tap_many(&dly[..n], n - 1, true, &mut rd[..n]);
            for (i, (o, x)) in out.iter_mut().zip(&rd[..n]).enumerate() {
                let t = (i + 1) as f32 * inv_n;
                *o += *x * (g0 + dg * t);
            }
        }
    }

    fn reset(&mut self) {
        self.delay_line.clear();
        self.prev.clear();
        self.ramps.clear();
        self.primed = false;
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}
