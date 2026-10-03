//! Zero-alloc patch-bay mixer with click-free (ramped) edits.
//!
//! The patch bay is the only place audio enters the scene mix. Each scene
//! output owns an explicit list of taps (`PatchEntry`); its audible content is
//! the sum over those taps of `source[source_idx].channel[channel] × gain`.
//!
//! # Ramped edits (#73)
//!
//! Every tap carries a *current* gain that moves linearly to its *target* over a ramp of
//! [`PatchBayNode::ramp_samples`] samples (default [`DEFAULT_PULL_RAMP_MS`] ms) once the bay has
//! rendered its first block:
//!
//! * a new tap fades in from 0 (attack),
//! * [`set_pull_gain`](PatchBayNode::set_pull_gain) glides from the current to the new gain
//!   (a retarget mid-ramp continues from where the ramp is, so it is always continuous),
//! * [`remove_pull`](PatchBayNode::remove_pull) fades the tap out (release) and drops it once
//!   it reaches 0; re-adding the same `(source, channel)` while it fades revives it.
//!
//! The largest per-sample gain step is `|Δgain| / ramp_samples`. Before the first block has been
//! rendered there is nothing to click against, so edits take effect immediately.
//!
//! This node deliberately does **not** implement the [`crate::node_graph::AudioNode`]
//! trait: it has many inputs (one buffer per loaded source, each with its own
//! channel count) and many outputs (one mono buffer per scene output).
//!
//! All memory is allocated at construction / config time (every bus reserves
//! [`MAX_PULLS_PER_OUTPUT`] taps, the bay [`MAX_BAY_OUTPUTS`] buses). The structural edit methods
//! ([`insert_output`](PatchBayNode::insert_output), [`remove_output`](PatchBayNode::remove_output),
//! [`remove_source`](PatchBayNode::remove_source), the pull setters) and
//! [`PatchBayNode::process`] never allocate, lock, or panic, so they may run on the audio thread.

use crate::audio_buffer::AudioBuffer;

/// Default attack/release ramp of a pull edit, in milliseconds.
pub const DEFAULT_PULL_RAMP_MS: f32 = 15.0;
/// Taps (including ones still fading out) one bus can hold without allocating.
pub const MAX_PULLS_PER_OUTPUT: usize = 64;
/// Output buses the bay can hold without allocating.
pub const MAX_BAY_OUTPUTS: usize = 256;

/// One tap in the patch bay: read `channel` of `source_idx` at `gain_linear`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchEntry {
    /// Index into the engine's source buffer array.
    pub source_idx: usize,
    /// Channel of that source to read (`0 .. source.channels()`).
    pub channel: usize,
    /// Linear amplitude gain applied to the tapped samples (the ramp TARGET).
    pub gain_linear: f32,
}

/// A tap plus its ramp state.
#[derive(Clone, Copy, Debug)]
struct Tap {
    source_idx: usize,
    channel: usize,
    /// Gain the ramp is heading to.
    target: f32,
    /// Gain at the start of the next block.
    gain: f32,
    /// Per-sample gain increment while `remaining > 0`.
    step: f32,
    /// Samples of ramp left.
    remaining: u32,
    /// Fading out: dropped once the ramp ends.
    removing: bool,
}

/// The taps of ONE scene output (opaque; build with [`PatchBayBus::new`], insert with
/// [`PatchBayNode::insert_output`]).
pub struct PatchBayBus {
    taps: Vec<Tap>,
}

impl PatchBayBus {
    /// An empty bus with room for [`MAX_PULLS_PER_OUTPUT`] taps (allocates).
    pub fn new() -> Self {
        Self { taps: Vec::with_capacity(MAX_PULLS_PER_OUTPUT) }
    }
}

impl Default for PatchBayBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Standalone multi-input / multi-output patch-bay mixer.
///
/// `outputs[i]` holds the list of taps that mix into scene output `i`.
pub struct PatchBayNode {
    /// Per scene output: the taps that mix into it.
    outputs: Vec<PatchBayBus>,
    /// Length of an attack/release ramp in samples (0 = edits are immediate).
    ramp_samples: u32,
    /// True once a block has been rendered: only then are edits ramped.
    rendered: bool,
}

impl PatchBayNode {
    /// Create a patch bay with `num_outputs` (initially empty) outputs and the default ramp
    /// ([`DEFAULT_PULL_RAMP_MS`] at 48 kHz; use [`set_ramp_ms`](Self::set_ramp_ms) for other rates).
    pub fn new(num_outputs: usize) -> Self {
        let mut outputs = Vec::with_capacity(num_outputs.max(MAX_BAY_OUTPUTS));
        for _ in 0..num_outputs {
            outputs.push(PatchBayBus::new());
        }
        Self {
            outputs,
            ramp_samples: (DEFAULT_PULL_RAMP_MS * 48.0) as u32,
            rendered: false,
        }
    }

    /// Number of scene-output mix busses.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Number of buses the bay can hold without allocating.
    pub fn capacity(&self) -> usize {
        self.outputs.capacity()
    }

    /// Set the attack/release ramp in samples (0 = immediate edits).
    pub fn set_ramp_samples(&mut self, samples: u32) {
        self.ramp_samples = samples;
    }

    /// Set the attack/release ramp in milliseconds at `sample_rate`.
    pub fn set_ramp_ms(&mut self, ms: f32, sample_rate: f32) {
        let s = if ms.is_finite() && sample_rate.is_finite() { (ms.max(0.0) * 0.001 * sample_rate).round() } else { 0.0 };
        self.ramp_samples = s as u32;
    }

    /// Current ramp length in samples.
    pub fn ramp_samples(&self) -> u32 {
        self.ramp_samples
    }

    /// Resize the number of outputs (API thread only; allocates when growing).
    ///
    /// Truncates when shrinking; appends empty buses when growing.
    pub fn resize(&mut self, num_outputs: usize) {
        if num_outputs < self.outputs.len() {
            self.outputs.truncate(num_outputs);
        } else {
            while self.outputs.len() < num_outputs {
                self.outputs.push(PatchBayBus::new());
            }
        }
    }

    /// Insert a prebuilt bus at `index` (clamped to the end), shifting later buses up. No
    /// allocation while the bay holds fewer than [`MAX_BAY_OUTPUTS`] buses. Returns the bus back
    /// if the bay is full.
    pub fn insert_output(&mut self, index: usize, bus: PatchBayBus) -> Result<(), PatchBayBus> {
        if self.outputs.len() >= self.outputs.capacity() {
            return Err(bus);
        }
        let at = index.min(self.outputs.len());
        self.outputs.insert(at, bus);
        Ok(())
    }

    /// Remove bus `index` (later buses shift down) and hand it back so the caller decides where
    /// it is dropped. `None` if out of range. Never allocates.
    pub fn remove_output(&mut self, index: usize) -> Option<PatchBayBus> {
        if index < self.outputs.len() {
            Some(self.outputs.remove(index))
        } else {
            None
        }
    }

    /// A source was unloaded: drop every tap on `source_idx` (the audio is gone, so there is
    /// nothing to fade) and renumber taps on later sources down by one. Never allocates.
    pub fn remove_source(&mut self, source_idx: usize) {
        for bus in &mut self.outputs {
            bus.taps.retain(|t| t.source_idx != source_idx);
            for t in bus.taps.iter_mut() {
                if t.source_idx > source_idx {
                    t.source_idx -= 1;
                }
            }
        }
    }

    /// Start (or restart) a ramp of `t` toward `target`.
    fn retarget(ramp: u32, rendered: bool, t: &mut Tap, target: f32) {
        t.target = target;
        if !rendered || ramp == 0 {
            t.gain = target;
            t.remaining = 0;
            t.step = 0.0;
        } else {
            t.step = (target - t.gain) / ramp as f32;
            t.remaining = ramp;
        }
    }

    /// Add (or replace) a pull on an output.
    ///
    /// A tap for the same `(source_idx, channel)` glides to the new gain (reviving it if it was
    /// fading out); otherwise a new tap fades in from 0 over one ramp. Allocation-free; returns
    /// `false` (and does nothing) only if the bus is full of live taps.
    pub fn set_pull(&mut self, output: usize, entry: PatchEntry) -> bool {
        let (ramp, rendered) = (self.ramp_samples, self.rendered);
        let Some(bus) = self.outputs.get_mut(output) else { return false };
        if let Some(t) = bus
            .taps
            .iter_mut()
            .find(|t| t.source_idx == entry.source_idx && t.channel == entry.channel)
        {
            t.removing = false;
            Self::retarget(ramp, rendered, t, entry.gain_linear);
            return true;
        }
        if bus.taps.len() >= bus.taps.capacity() {
            // Full: evict the fading tap closest to done (never allocate).
            let victim = bus
                .taps
                .iter()
                .enumerate()
                .filter(|(_, t)| t.removing)
                .min_by_key(|(_, t)| t.remaining)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    bus.taps.swap_remove(i);
                }
                None => return false,
            }
        }
        let mut t = Tap {
            source_idx: entry.source_idx,
            channel: entry.channel,
            target: 0.0,
            gain: 0.0,
            step: 0.0,
            remaining: 0,
            removing: false,
        };
        Self::retarget(ramp, rendered, &mut t, entry.gain_linear);
        bus.taps.push(t);
        true
    }

    /// Remove every pull tapping `(source_idx, channel)` from an output: fades out over one
    /// ramp and is dropped at 0 (immediately if the bay has not rendered yet). No-op if none
    /// match.
    pub fn remove_pull(&mut self, output: usize, source_idx: usize, channel: usize) {
        let (ramp, rendered) = (self.ramp_samples, self.rendered);
        let Some(bus) = self.outputs.get_mut(output) else { return };
        for t in bus
            .taps
            .iter_mut()
            .filter(|t| t.source_idx == source_idx && t.channel == channel)
        {
            t.removing = true;
            Self::retarget(ramp, rendered, t, 0.0);
        }
        bus.taps.retain(|t| !(t.removing && t.remaining == 0));
    }

    /// Glide the linear gain of an existing (not fading-out) pull. No-op if absent.
    pub fn set_pull_gain(&mut self, output: usize, source_idx: usize, channel: usize, gain_linear: f32) {
        let (ramp, rendered) = (self.ramp_samples, self.rendered);
        let Some(bus) = self.outputs.get_mut(output) else { return };
        if let Some(t) = bus
            .taps
            .iter_mut()
            .find(|t| t.source_idx == source_idx && t.channel == channel && !t.removing)
        {
            Self::retarget(ramp, rendered, t, gain_linear);
        }
    }

    /// The live (not fading-out) taps of an output with their TARGET gains (config-side view;
    /// allocates).
    pub fn pulls(&self, output: usize) -> Vec<PatchEntry> {
        self.outputs
            .get(output)
            .map(|bus| {
                bus.taps
                    .iter()
                    .filter(|t| !t.removing)
                    .map(|t| PatchEntry { source_idx: t.source_idx, channel: t.channel, gain_linear: t.target })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Number of taps on an output still fading out.
    pub fn fading_pulls(&self, output: usize) -> usize {
        self.outputs.get(output).map_or(0, |b| b.taps.iter().filter(|t| t.removing).count())
    }

    /// Mix all taps into the output busses. ZERO ALLOCATION.
    ///
    /// `sources`: one buffer per Source (each with its own channel count).
    /// `outputs`: one MONO buffer per SceneOutput (already sized `num_outputs`).
    ///
    /// Each output is cleared to zero, then every tap accumulates
    /// `sources[source_idx].channel[channel][i] × gain(i)` where `gain(i)` follows the tap's
    /// ramp. Out-of-range `source_idx`/`channel` are silently skipped (defensive; no panic) but
    /// their ramps still advance. Samples beyond the shorter of the source/output channel are
    /// left untouched.
    pub fn process(&mut self, sources: &[&AudioBuffer], outputs: &mut [AudioBuffer]) {
        self.process_into(sources, outputs.iter_mut());
    }

    /// [`process`](Self::process) for outputs that are not one contiguous slice (bus `i` renders
    /// into the `i`-th buffer yielded). ZERO ALLOCATION.
    pub fn process_into<'a>(
        &mut self,
        sources: &[&AudioBuffer],
        outputs: impl Iterator<Item = &'a mut AudioBuffer>,
    ) {
        for (bus, output) in self.outputs.iter_mut().zip(outputs) {
            output.clear();
            let out_ch = output.channel_mut(0);
            let block = out_ch.len();
            for t in bus.taps.iter_mut() {
                let src = sources
                    .get(t.source_idx)
                    .filter(|s| t.channel < s.channels() as usize)
                    .map(|s| s.channel(t.channel as u16));
                let n = src.map_or(block, |s| block.min(s.len()));
                let r = (t.remaining as usize).min(n);
                if let Some(s) = src {
                    // Ramped part, then constant (target) part.
                    for i in 0..r {
                        out_ch[i] += s[i] * (t.gain + t.step * (i + 1) as f32);
                    }
                    let g = if r as u32 == t.remaining { t.target } else { t.gain + t.step * r as f32 };
                    if g != 0.0 {
                        for i in r..n {
                            out_ch[i] += s[i] * g;
                        }
                    }
                }
                if t.remaining > 0 {
                    if r as u32 == t.remaining {
                        t.gain = t.target;
                        t.remaining = 0;
                        t.step = 0.0;
                    } else {
                        t.gain += t.step * r as f32;
                        t.remaining -= r as u32;
                    }
                }
            }
            bus.taps.retain(|t| !(t.removing && t.remaining == 0));
        }
        self.rendered = true;
    }
}
