//! Audio-thread render state of the scene pipeline.
//!
//! The state is split by what each piece depends on, so structural edits (adding / removing an
//! output or a listener) can be made IN PLACE without touching the DSP state of anything that
//! survives (#73):
//!
//! * [`OutputRender`] - per scene output (emitter): dry delay line, direct-chain scratch, LFE send.
//! * [`ListenerRender`] - per listener: decoder (VBAP / HRTF), shared reverb bus, LFE bus, and the
//!   list of that listener's [`PairRender`]s (one per scene output, same order as the outputs).
//! * [`PairRender`] - per (listener, output) pair: parameter triple buffer reader, smoothing
//!   crossfader, "ready" flag (#119), pan / send ramps, reflection decoder, binaural renderer.
//!
//! Everything is allocated by the `build_*` functions at config time. The `SceneRenderState`
//! structural methods only `Vec::insert` / `remove` within capacity reserved at construction, so
//! they too are allocation-free; removed boxes are handed back to the caller so no `Drop` of big
//! DSP state has to happen where the audio thread is.

use std::sync::Arc;

use quasar_core::bands::Band8;
use quasar_core::param_exchange::{ParameterTripleBuffer, SpatialCoefficients};
use quasar_core::scene_output::{ListenerConfig, PhysicalOutputLayout};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use quasar_dsp::binaural::{BinauralConfig, BinauralRenderer, ParametricBinauralRenderer};
use quasar_dsp::biquad::BiquadFilter;
use quasar_dsp::crossfader::{EqualPowerCrossfader, MAX_CROSSFADE_REFLECTIONS};
use quasar_dsp::early_reflections::EarlyReflectionDelayNode;
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::limiter::{OutputMeter, OutputSafety, OutputSafetyConfig};
use quasar_dsp::master_decoder::{layout_lfe, layout_panner, SpeakerLayout};
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;
use quasar_dsp::patch_bay::{PatchBayBus, PatchBayNode, PatchEntry};
use quasar_dsp::reflection_decoder::{ReflectionDecoder, TapTarget};
use quasar_dsp::vbap::VbapPanner;

/// Maximum number of scene outputs (emitters). Capacity of the audio-side vectors, reserved at
/// construction so structural edits never reallocate.
pub const MAX_SCENE_OUTPUTS: usize = quasar_dsp::patch_bay::MAX_BAY_OUTPUTS;
/// Maximum number of listeners.
pub const MAX_LISTENERS: usize = 16;

/// Corner frequency of the per-listener LFE low-pass (4th-order Butterworth, two biquads).
pub const LFE_CUTOFF_HZ: f32 = 120.0;

/// Capacity of each scene output's propagation-delay line, in seconds (0.5 s =
/// 171 m at 343 m/s). Longer direct paths are clamped to this delay.
pub const MAX_PROPAGATION_DELAY_SECS: f32 = 0.5;

/// Per scene output: the LISTENER-INDEPENDENT stage (patch-bay mix + the dry delay line). Boxed.
pub(crate) struct OutputRender {
    /// The output's un-attenuated dry delay line (early-reflection taps and reverb sends read it).
    early: EarlyReflectionDelayNode,
    /// Patch-bay output.
    mixed: AudioBuffer,
    /// LFE send (linear).
    pub(crate) lfe_send: f32,
}

impl OutputRender {
    pub(crate) fn new(sample_rate: f32, lfe_send: f32) -> Box<Self> {
        Box::new(Self {
            early: EarlyReflectionDelayNode::new(1, sample_rate, 0.2, 16),
            mixed: AudioBuffer::new(1, DEFAULT_BLOCK_SIZE as u16),
            lfe_send,
        })
    }
}

/// Per (listener, output) pair.
pub(crate) struct PairRender {
    /// Single-slot triple buffer shared with the compute side (which writes it).
    params: Arc<ParameterTripleBuffer>,
    /// Per-pair direct chain (#77): propagation delay + band EQ (occlusion / air absorption) +
    /// gain, driven by THIS listener's coefficients. Memory: one `MAX_PROPAGATION_DELAY_SECS`
    /// (0.5 s) delay line per (listener x output) pair.
    occ: AirAbsorptionOcclusionNode,
    /// Output of the direct chain: this pair's direct mono (reflections / reverb are added later).
    direct: AudioBuffer,
    crossfader: EqualPowerCrossfader,
    last_version: u64,
    /// True once the pair has received its first REAL published coefficients (#119): until then it
    /// renders silence instead of gliding / sounding from the default coefficients.
    ready: bool,
    /// Speaker gains at the end of the previous block (per-sample ramps).
    prev_gains: [f32; MAX_AUDIO_CHANNELS],
    /// False until the pair has rendered once (first block uses the target directly).
    prev_valid: bool,
    /// `Some` only for HRTF listeners.
    binaural: Option<Box<dyn BinauralRenderer + Send>>,
    refl_dec: ReflectionDecoder,
    /// Reverb send gain / propagation delay at the end of the previous block; NaN until rendered.
    rev_send_prev: f32,
    rev_delay_prev: f32,
    /// LFE send applied at the end of the previous block.
    lfe_send_prev: f32,
}

impl PairRender {
    pub(crate) fn new(
        sample_rate: f32,
        fade_ms: f32,
        hrtf: bool,
        lfe_send: f32,
        params: Arc<ParameterTripleBuffer>,
    ) -> Box<Self> {
        Box::new(Self {
            params,
            occ: AirAbsorptionOcclusionNode::new(1, sample_rate, MAX_PROPAGATION_DELAY_SECS),
            direct: AudioBuffer::new(1, DEFAULT_BLOCK_SIZE as u16),
            crossfader: EqualPowerCrossfader::new(fade_ms, sample_rate, initial_scene_coeffs()),
            last_version: 0,
            ready: false,
            prev_gains: [0.0; MAX_AUDIO_CHANNELS],
            prev_valid: false,
            binaural: if hrtf {
                Some(Box::new(ParametricBinauralRenderer::new(BinauralConfig::new(sample_rate))))
            } else {
                None
            },
            refl_dec: ReflectionDecoder::new(sample_rate, hrtf),
            rev_send_prev: f32::NAN,
            rev_delay_prev: f32::NAN,
            lfe_send_prev: lfe_send,
        })
    }
}

/// Per listener.
pub(crate) struct ListenerRender {
    heading: [f32; 3],
    /// VBAP panner of the listener's layout (LFE slots excluded).
    panner: VbapPanner,
    /// LFE slots of the layout (empty = no LFE channel).
    lfe_slots: &'static [usize],
    /// LFE low-pass (4th order = two biquads) fed by the per-output sends.
    lfe_filters: [BiquadFilter; 2],
    /// True while the LFE filters may still hold signal (tail decay after sends stop).
    lfe_hot: bool,
    /// Shared reverb bus (FDN).
    rev_bus: FdnReverbNode,
    /// FDN output `k` on channel `k`.
    rev_out: AudioBuffer,
    /// Listener output channel of each FDN output.
    rev_slots: Vec<usize>,
    /// Layout normalisation of the diffuse level (constant total power).
    rev_gain: f32,
    /// One pair per scene output, same order as `SceneRenderState::outputs`.
    pairs: Vec<Box<PairRender>>,
    /// Output safety stage (limiter, scrub, meters) of this listener's bus (#80).
    safety: OutputSafety,
}

impl ListenerRender {
    /// Install the per-output pairs (compute side, before the listener is shipped).
    pub(crate) fn set_pairs(&mut self, pairs: Vec<Box<PairRender>>) {
        self.pairs.extend(pairs);
    }

    /// Shared meters of this listener's output stage.
    pub(crate) fn meter(&self) -> Arc<OutputMeter> {
        Arc::clone(self.safety.meter())
    }

    pub(crate) fn new(sample_rate: f32, cfg: &ListenerConfig) -> Box<Self> {
        let layout = physical_to_speaker_layout(&cfg.physical_layout);
        let hrtf = cfg.physical_layout == PhysicalOutputLayout::Hrtf;
        let panner = layout_panner(&layout);
        let lfe_slots = layout_lfe(&layout);
        let (mut a, mut b) = (BiquadFilter::new(), BiquadFilter::new());
        a.set_lowpass_q(LFE_CUTOFF_HZ, 0.5412, sample_rate);
        b.set_lowpass_q(LFE_CUTOFF_HZ, 1.3066, sample_rate);
        // FDN output k -> listener output channel: every non-LFE slot of a speaker layout
        // (LFE never gets reverb), the two ears for an HRTF listener.
        let rev_slots: Vec<usize> = if hrtf {
            vec![0, 1]
        } else {
            (0..panner.num_outputs())
                .filter(|s| !lfe_slots.contains(s))
                .take(quasar_dsp::late_reverb::FDN_MAX_BUS_OUTPUTS)
                .collect()
        };
        // Constant TOTAL diffuse power across layouts, referenced to a stereo pair.
        let rev_gain = (2.0 / rev_slots.len().max(2) as f32).sqrt();
        let rev_out = AudioBuffer::new(rev_slots.len().max(1) as u16, DEFAULT_BLOCK_SIZE as u16);
        Box::new(Self {
            heading: cfg.heading,
            panner,
            lfe_slots,
            lfe_filters: [a, b],
            lfe_hot: false,
            rev_bus: FdnReverbNode::new(1, sample_rate),
            rev_out,
            rev_slots,
            rev_gain,
            pairs: Vec::with_capacity(MAX_SCENE_OUTPUTS),
            safety: OutputSafety::new(sample_rate, OutputSafetyConfig::default()),
        })
    }
}

/// Everything the audio thread renders with. See the module docs.
pub(crate) struct SceneRenderState {
    patch_bay: PatchBayNode,
    outputs: Vec<Box<OutputRender>>,
    listeners: Vec<Box<ListenerRender>>,
    /// Mono send-sum scratch (one block) of the listener being rendered.
    rev_in: Vec<f32>,
    /// Mono scratch for the summed LFE send of one listener (one block).
    lfe_scratch: Vec<f32>,
    /// Preallocated (capacity MAX_CROSSFADE_REFLECTIONS) listener-space tap list for one pair.
    tap_targets: Vec<TapTarget>,
}

impl SceneRenderState {
    /// Empty state (no outputs, no listeners).
    pub(crate) fn new(sample_rate: f32) -> Self {
        let mut patch_bay = PatchBayNode::new(0);
        patch_bay.set_ramp_ms(quasar_dsp::patch_bay::DEFAULT_PULL_RAMP_MS, sample_rate);
        Self {
            patch_bay,
            outputs: Vec::with_capacity(MAX_SCENE_OUTPUTS),
            listeners: Vec::with_capacity(MAX_LISTENERS),
            rev_in: vec![0.0; DEFAULT_BLOCK_SIZE],
            lfe_scratch: vec![0.0; DEFAULT_BLOCK_SIZE],
            tap_targets: Vec::with_capacity(MAX_CROSSFADE_REFLECTIONS),
        }
    }

    pub(crate) fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    pub(crate) fn num_listeners(&self) -> usize {
        self.listeners.len()
    }

    // ── structural edits (allocation-free; state of survivors untouched) ──────────────────

    /// Append a scene output (taken out of `add`) with its patch-bay bus and one prebuilt pair
    /// per listener (same order as the listeners). Returns `false` and leaves `add` untouched
    /// if a capacity is exhausted or the pair count does not match.
    pub(crate) fn add_output(&mut self, add: &mut OutputAdd) -> bool {
        if self.outputs.len() >= MAX_SCENE_OUTPUTS
            || add.pairs.len() != self.listeners.len()
            || add.output.is_none()
            || add.bus.is_none()
            || self.patch_bay.num_outputs() >= self.patch_bay.capacity()
        {
            return false;
        }
        let (Some(output), Some(bus)) = (add.output.take(), add.bus.take()) else { return false };
        if let Err(bus) = self.patch_bay.insert_output(self.outputs.len(), bus) {
            add.bus = Some(bus);
            add.output = Some(output);
            return false;
        }
        self.outputs.push(output);
        for (lis, pair) in self.listeners.iter_mut().zip(add.pairs.drain(..)) {
            lis.pairs.push(pair);
        }
        true
    }

    /// Remove scene output `idx` (later outputs shift down); the removed boxes go to `sink`.
    pub(crate) fn remove_output(&mut self, idx: usize, sink: &mut impl FnMut(Garbage)) {
        if idx >= self.outputs.len() {
            return;
        }
        sink(Garbage::Output(self.outputs.remove(idx)));
        if let Some(bus) = self.patch_bay.remove_output(idx) {
            sink(Garbage::Bus(bus));
        }
        for lis in self.listeners.iter_mut() {
            if idx < lis.pairs.len() {
                sink(Garbage::Pair(lis.pairs.remove(idx)));
            }
        }
    }

    /// Append a listener whose `pairs` already has one entry per existing output. A listener
    /// that does not fit (capacity / pair count) goes to `sink` instead.
    pub(crate) fn add_listener(&mut self, listener: Box<ListenerRender>, sink: &mut impl FnMut(Garbage)) {
        if self.listeners.len() >= MAX_LISTENERS || listener.pairs.len() != self.outputs.len() {
            sink(Garbage::Listener(listener));
            return;
        }
        self.listeners.push(listener);
    }

    /// Remove listener `idx` (later listeners shift down); the removed box goes to `sink`.
    pub(crate) fn remove_listener(&mut self, idx: usize, sink: &mut impl FnMut(Garbage)) {
        if idx < self.listeners.len() {
            sink(Garbage::Listener(self.listeners.remove(idx)));
        }
    }

    pub(crate) fn set_output_lfe_send(&mut self, idx: usize, g: f32) {
        if let Some(o) = self.outputs.get_mut(idx) {
            o.lfe_send = g;
        }
    }

    pub(crate) fn set_safety(&mut self, idx: usize, cfg: OutputSafetyConfig) {
        if let Some(l) = self.listeners.get_mut(idx) {
            l.safety.set_config(cfg);
        }
    }

    pub(crate) fn meter(&self, idx: usize) -> Option<&Arc<OutputMeter>> {
        self.listeners.get(idx).map(|l| l.safety.meter())
    }

    pub(crate) fn set_listener_pose(&mut self, idx: usize, heading: [f32; 3]) {
        if let Some(l) = self.listeners.get_mut(idx) {
            l.heading = heading;
        }
    }

    // ── per-block render ─────────────────────────────────────────────────────────────────

    /// Render one block. ZERO ALLOCATION. Pre-conditions (checked by the caller): one
    /// `listener_outputs` buffer per listener, every buffer `<= DEFAULT_BLOCK_SIZE` samples.
    ///
    /// Pipeline per block:
    ///   1. publish latest triple-buffer data and smooth per-pair coefficients through the
    ///      crossfaders (a pair snaps on its first real update, #119);
    ///   2. the patch bay sums the configured pulls into one mono buffer per scene output;
    ///   3. per scene output (once, reference listener 0): direct chain, dry delay line push;
    ///   4. per listener: VBAP / HRTF decode of every output, early reflections, the shared
    ///      reverb bus, the LFE bus;
    ///   5. advance all crossfaders.
    pub(crate) fn process(
        &mut self,
        sources: &[&AudioBuffer],
        listener_outputs: &mut [AudioBuffer],
        stage: u8,
    ) {
        let n_out = self.outputs.len();
        let n_lis = self.listeners.len();
        if n_out == 0 || n_lis == 0 || listener_outputs.is_empty() {
            for l in listener_outputs.iter_mut() {
                l.clear();
            }
            return;
        }
        let block = listener_outputs[0].samples() as usize;

        // 1. Publish + retarget. Re-target ONLY on a strictly newer version (the triple buffer
        //    rotates its read slot, so an old snapshot is re-read with its old version).
        for lis in self.listeners.iter_mut() {
            for pair in lis.pairs.iter_mut() {
                pair.params.update();
                let ver = pair.params.read_version(0);
                if ver > pair.last_version {
                    pair.last_version = ver;
                    let latest = unsafe { pair.params.read(0) };
                    if pair.ready {
                        pair.crossfader.set_target(latest);
                    } else {
                        // First real update of this pair: snap (no glide from the defaults).
                        pair.crossfader.snap_to_ref(latest);
                        pair.ready = true;
                    }
                }
            }
        }

        // 2. Patch bay: sum pulls into one mono buffer per scene output.
        self.patch_bay_process(sources);

        // 3. Listener-independent stage, once per scene output: the un-attenuated, undelayed dry
        //    signal (post patch bay) feeds the output's delay line that the early-reflection taps
        //    and the reverb sends read (#59, #125). It starts with the first block in which any
        //    listener's pair for the output has real coefficients (#119). `stage` gates the
        //    spatial stages: 0 = silence, 1 = raw mixed only, 2 = +occlusion, 3 = +early, 4 = full.
        let stage = stage.min(4);
        if stage >= 3 {
            for o in 0..n_out {
                let live = self.listeners.iter().any(|l| l.pairs.get(o).map_or(false, |p| p.ready));
                if live {
                    let out = &mut *self.outputs[o];
                    out.early.push_block(&out.mixed);
                }
            }
        }

        // 4. Per-listener decode; the direct chain runs per (listener, output) pair.
        let n_lis_proc = n_lis.min(listener_outputs.len());
        let mut target = [0.0_f32; MAX_AUDIO_CHANNELS];
        for l in 0..n_lis_proc {
            let lis = &mut *self.listeners[l];
            let out = &mut listener_outputs[l];
            let basis = ListenerBasis::from_heading(lis.heading);
            out.clear();
            let n_speakers = out.channels() as usize;
            let n_o = n_out.min(lis.pairs.len());
            let ListenerRender { panner, pairs, rev_bus, rev_out, rev_slots, rev_gain, lfe_slots, lfe_filters, lfe_hot, safety, .. } = lis;
            let n = panner.num_outputs().min(MAX_AUDIO_CHANNELS);

            for o in 0..n_o {
                let pair = &mut *pairs[o];
                if !pair.ready {
                    continue; // no real coefficients yet: silence (#119)
                }
                let coeff = pair.crossfader.current_coefficients();

                // Direct chain of THIS pair (#77): delay, band EQ (occlusion / air absorption) and
                // gain from this listener's own coefficients, so each listener hears the emitter at
                // its own distance and occlusion. The node ramps gains, filter coefficients and
                // delay per sample.
                match stage {
                    0 => pair.direct.clear(),
                    1 => pair.direct.copy_from(&self.outputs[o].mixed),
                    _ => {
                        // Emitter directivity toward THIS listener is a separate per-band factor.
                        let gains = coeff.direct_gain.mul(&coeff.directivity_gain);
                        pair.occ.process_with_gains(
                            &self.outputs[o].mixed,
                            &mut pair.direct,
                            &gains,
                            coeff.direct_delay_samples,
                        );
                    }
                }
                let combined = &pair.direct;
                let (az, el) = basis.to_listener_angles(coeff.direct_azimuth, coeff.direct_elevation);

                if let Some(bin) = pair.binaural.as_mut() {
                    let (left, right) = out.stereo_mut();
                    let nb = block.min(left.len()).min(right.len());
                    bin.render_add(&combined.channel(0)[..nb], az, el, &mut left[..nb], &mut right[..nb]);
                } else {
                    panner.gains(az, el, &mut target[..n]);
                    let prev = &mut pair.prev_gains;
                    if !pair.prev_valid {
                        // First block: start at the target, don't ramp from garbage.
                        prev[..n].copy_from_slice(&target[..n]);
                        pair.prev_valid = true;
                    }
                    let combined_ch = combined.channel(0);
                    for sp in 0..n {
                        let g0 = prev[sp];
                        let g1 = target[sp];
                        prev[sp] = g1;
                        if sp >= n_speakers || (g0 == 0.0 && g1 == 0.0) {
                            continue;
                        }
                        let ch = out.channel_mut(sp as u16);
                        if g0 == g1 {
                            for i in 0..block {
                                ch[i] += combined_ch[i] * g1;
                            }
                        } else {
                            let step = (g1 - g0) / block.max(1) as f32;
                            for i in 0..block {
                                ch[i] += combined_ch[i] * (g0 + step * (i + 1) as f32);
                            }
                        }
                    }
                }

                // Early reflections of this pair: every tap at its OWN arrival direction, rotated
                // into the listener frame, through the same decoder, with per-tap ramps (#58).
                if stage >= 3 {
                    self.tap_targets.clear();
                    for er in coeff.early_reflections.iter().take(MAX_CROSSFADE_REFLECTIONS) {
                        let (taz, tel) = basis.to_listener_angles(er.azimuth, er.elevation);
                        let lo = er.gain.0[..4].iter().sum::<f32>() * 0.25;
                        let hi = er.gain.0[4..].iter().sum::<f32>() * 0.25;
                        self.tap_targets.push(TapTarget {
                            delay_samples: er.delay_samples,
                            gain_lo: lo,
                            gain_hi: hi,
                            azimuth: taz,
                            elevation: tel,
                        });
                    }
                    pair.refl_dec.render_add(&self.outputs[o].early, &self.tap_targets, Some(&*panner), out, block);
                }
            }

            // Late reverb: ONE shared FDN bus per listener (#62). Each output contributes a SEND
            // of its un-attenuated dry signal read at its propagation delay, scaled by
            // `late_gain_db` (the room's diffuse level re the direct sound at 1 m; independent of
            // the direct path's distance gain / occlusion). Sends are ramped per sample. The bus
            // is decoded DIFFUSELY (FDN output k -> k-th non-LFE speaker / an ear); T60 is the
            // mean over the listener's ready outputs.
            if stage >= 4 && !rev_slots.is_empty() {
                let inv_n = 1.0 / block.max(1) as f32;
                let max_d = (self.outputs[0].early.max_tap_delay() - block as f32).max(0.0);
                self.rev_in[..block].fill(0.0);
                let mut t60_sum = Band8::zeros();
                let mut n_ready = 0usize;
                for o in 0..n_o {
                    let pair = &mut *pairs[o];
                    if !pair.ready {
                        continue;
                    }
                    n_ready += 1;
                    let coeff = pair.crossfader.current_coefficients();
                    t60_sum = t60_sum.add(&coeff.late_t60);
                    let s1 = if coeff.late_gain_db.is_finite() { db_to_linear(coeff.late_gain_db.min(40.0)) } else { 0.0 };
                    let d1 = if coeff.direct_delay_samples.is_finite() {
                        coeff.direct_delay_samples.clamp(0.0, max_d)
                    } else {
                        0.0
                    };
                    let (mut s0, mut d0) = (pair.rev_send_prev, pair.rev_delay_prev);
                    if !s0.is_finite() || !d0.is_finite() {
                        s0 = s1; // first block of this pair: start at the target
                        d0 = d1;
                    }
                    pair.rev_send_prev = s1;
                    pair.rev_delay_prev = d1;
                    if s0 <= 0.0 && s1 <= 0.0 {
                        continue;
                    }
                    let line = &self.outputs[o].early;
                    let acc = &mut self.rev_in[..block];
                    for j in 0..block {
                        let t = (j + 1) as f32 * inv_n;
                        let d = d0 + (d1 - d0) * t + (block - 1 - j) as f32;
                        acc[j] += line.tap_at(d) * (s0 + (s1 - s0) * t);
                    }
                }
                rev_bus.set_t60(&t60_sum.scale(1.0 / n_ready.max(1) as f32));
                rev_bus.set_wet(*rev_gain);
                let n_fdn = rev_slots.len();
                rev_bus.process_bus(&self.rev_in[..block], rev_out, n_fdn);
                for (k, &slot) in rev_slots.iter().enumerate() {
                    if slot < n_speakers {
                        let src = rev_out.channel(k as u16);
                        let dst = out.channel_mut(slot as u16);
                        for i in 0..block {
                            dst[i] += src[i];
                        }
                    }
                }
            }

            // LFE bus: sum the per-output sends (ramped per sample), low-pass once (the sum is
            // linear), add to the LFE slot(s).
            if !lfe_slots.is_empty() {
                let scratch = &mut self.lfe_scratch[..block];
                scratch.fill(0.0);
                let mut any = false;
                for o in 0..n_o {
                    let pair = &mut *pairs[o];
                    let g1 = self.outputs[o].lfe_send;
                    let g0 = pair.lfe_send_prev;
                    pair.lfe_send_prev = g1;
                    if g0 == 0.0 && g1 == 0.0 {
                        continue;
                    }
                    any = true;
                    let combined_ch = pair.direct.channel(0);
                    let step = (g1 - g0) / block.max(1) as f32;
                    for i in 0..block {
                        scratch[i] += combined_ch[i] * (g0 + step * (i + 1) as f32);
                    }
                }
                if any {
                    *lfe_hot = true;
                }
                if *lfe_hot {
                    let [f1, f2] = lfe_filters;
                    let mut peak = 0.0_f32;
                    for i in 0..block {
                        let y = f2.process(f1.process(scratch[i]));
                        scratch[i] = y;
                        peak = peak.max(y.abs());
                    }
                    for &slot in lfe_slots.iter() {
                        if slot < n_speakers {
                            let ch = out.channel_mut(slot as u16);
                            for i in 0..block {
                                ch[i] += scratch[i];
                            }
                        }
                    }
                    if !any && peak < 1e-9 {
                        // Tail has decayed: stop filtering and drop the (denormal) state.
                        *lfe_hot = false;
                        f1.reset();
                        f2.reset();
                    }
                }
            }

            // Output safety stage (#80): gain staging, NaN / inf scrub, look-ahead limiter, meters.
            safety.process(out);
        }

        // 5. Advance all crossfaders by the block size (fades complete in ~fade_ms of real time).
        for lis in self.listeners.iter_mut() {
            for pair in lis.pairs.iter_mut() {
                pair.crossfader.advance(block);
            }
        }
    }

    /// Run the patch bay into every output's `mixed` buffer.
    fn patch_bay_process(&mut self, sources: &[&AudioBuffer]) {
        self.patch_bay.process_into(sources, self.outputs.iter_mut().map(|o| &mut o.mixed));
    }
}

/// Everything needed to add one scene output, prepared on the compute side. The shell travels
/// to the audio thread and back (as [`Garbage::Shell`]) so no `Vec` is freed there.
pub(crate) struct OutputAdd {
    pub(crate) output: Option<Box<OutputRender>>,
    pub(crate) bus: Option<PatchBayBus>,
    /// One pair per listener, in listener order (drained by the audio side).
    pub(crate) pairs: Vec<Box<PairRender>>,
}

/// Boxes retired on the audio side, to be dropped elsewhere (no big `Drop` or `free` on the
/// audio thread).
#[allow(dead_code)] // held only so that they are dropped by the compute side
pub(crate) enum Garbage {
    Output(Box<OutputRender>),
    Pair(Box<PairRender>),
    Listener(Box<ListenerRender>),
    Bus(PatchBayBus),
    Shell(Box<OutputAdd>),
}

/// A configuration change for the audio thread, applied at the start of a block. Small and
/// `Send`; heavy payloads are boxed on the compute side.
pub(crate) enum Command {
    AddOutput(Box<OutputAdd>),
    RemoveOutput(usize),
    AddListener(Box<ListenerRender>),
    RemoveListener(usize),
    SetPull { output: usize, entry: PatchEntry },
    RemovePull { output: usize, source_idx: usize, channel: usize },
    SetPullGain { output: usize, source_idx: usize, channel: usize, gain: f32 },
    RemoveSource(usize),
    SetLfeSend { output: usize, gain: f32 },
    SetListenerHeading { listener: usize, heading: [f32; 3] },
    SetSafety { listener: usize, cfg: OutputSafetyConfig },
    SetPullRampSamples(u32),
}

impl SceneRenderState {
    /// Apply one command. Allocation-free; retired boxes go to `sink`.
    pub(crate) fn apply(&mut self, cmd: Command, sink: &mut impl FnMut(Garbage)) {
        match cmd {
            Command::AddOutput(mut add) => {
                self.add_output(&mut add);
                // Whether it fitted or not, the shell (and anything left in it) goes back.
                sink(Garbage::Shell(add));
            }
            Command::RemoveOutput(i) => self.remove_output(i, sink),
            Command::AddListener(l) => self.add_listener(l, sink),
            Command::RemoveListener(i) => self.remove_listener(i, sink),
            Command::SetPull { output, entry } => {
                self.patch_bay.set_pull(output, entry);
            }
            Command::RemovePull { output, source_idx, channel } => {
                self.patch_bay.remove_pull(output, source_idx, channel)
            }
            Command::SetPullGain { output, source_idx, channel, gain } => {
                self.patch_bay.set_pull_gain(output, source_idx, channel, gain)
            }
            Command::RemoveSource(i) => self.patch_bay.remove_source(i),
            Command::SetLfeSend { output, gain } => self.set_output_lfe_send(output, gain),
            Command::SetListenerHeading { listener, heading } => self.set_listener_pose(listener, heading),
            Command::SetSafety { listener, cfg } => self.set_safety(listener, cfg),
            Command::SetPullRampSamples(n) => self.patch_bay.set_ramp_samples(n),
        }
    }
}

/// Default `SpatialCoefficients` used to seed scene-pipeline crossfaders.
pub(crate) fn initial_scene_coeffs() -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: 0.0,
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version: 0,
    }
}

/// Convert a dB gain to linear amplitude.
pub(crate) fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Map a listener's physical output layout to a VBAP [`SpeakerLayout`].
///
/// `Hrtf` listeners are rendered by the binaural path, never by this panner; the
/// Stereo mapping only gives them a placeholder (2-slot) panner so the per-listener
/// vectors stay uniform.
fn physical_to_speaker_layout(layout: &PhysicalOutputLayout) -> SpeakerLayout {
    match layout {
        PhysicalOutputLayout::Stereo => SpeakerLayout::Stereo,
        PhysicalOutputLayout::Surround51 => SpeakerLayout::Surround51,
        PhysicalOutputLayout::Surround714 => SpeakerLayout::Surround714,
        PhysicalOutputLayout::Quad => SpeakerLayout::Quad,
        PhysicalOutputLayout::Custom { positions } => SpeakerLayout::Custom { positions: positions.clone() },
        PhysicalOutputLayout::Hrtf => SpeakerLayout::Stereo,
    }
}

/// Listener-space basis built from a heading vector (forward), world up = +Y.
///
/// `right = forward x up`, `up' = right x forward`. When looking (almost)
/// straight up/down the right axis falls back to world +X.
pub(crate) struct ListenerBasis {
    fwd: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
}

impl ListenerBasis {
    pub(crate) fn from_heading(heading: [f32; 3]) -> Self {
        let norm = |v: [f32; 3]| -> Option<[f32; 3]> {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 1e-4 && l.is_finite() { Some([v[0] / l, v[1] / l, v[2] / l]) } else { None }
        };
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let fwd = norm(heading).unwrap_or([0.0, 0.0, -1.0]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0])).unwrap_or([1.0, 0.0, 0.0]);
        let up = cross(right, fwd);
        Self { fwd, right, up }
    }

    /// Convert world-space (azimuth, elevation) from the listener to listener-space angles.
    pub(crate) fn to_listener_angles(&self, az: f32, el: f32) -> (f32, f32) {
        let s = [az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()];
        let d = |b: &[f32; 3]| b[0] * s[0] + b[1] * s[1] + b[2] * s[2];
        let (x, y, f) = (d(&self.right), d(&self.up), d(&self.fwd));
        (x.atan2(f), y.atan2(x.hypot(f)))
    }
}
