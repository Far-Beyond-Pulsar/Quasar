//! Zero-alloc patch-bay mixer.
//!
//! The patch bay is the only place audio enters the scene mix. Each scene
//! output owns an explicit list of taps (`PatchEntry`); its audible content is
//! the sum over those taps of `source[source_idx].channel[channel] × gain_linear`.
//!
//! This node deliberately does **not** implement the [`crate::node_graph::AudioNode`]
//! trait: it has many inputs (one buffer per loaded source, each with its own
//! channel count) and many outputs (one mono buffer per scene output).
//!
//! All memory is allocated at construction / config time. [`PatchBayNode::process`]
//! never allocates, locks, or panics.

use crate::audio_buffer::AudioBuffer;

/// One tap in the patch bay: read `channel` of `source_idx` at `gain_linear`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchEntry {
    /// Index into the engine's source buffer array.
    pub source_idx: usize,
    /// Channel of that source to read (`0 .. source.channels()`).
    pub channel: usize,
    /// Linear amplitude gain applied to the tapped samples.
    pub gain_linear: f32,
}

/// Standalone multi-input / multi-output patch-bay mixer.
///
/// `outputs[i]` holds the list of taps that mix into scene output `i`.
pub struct PatchBayNode {
    /// Per scene output: the list of taps that mix into it.
    outputs: Vec<Vec<PatchEntry>>,
}

impl PatchBayNode {
    /// Create a patch bay with `num_outputs` (initially empty) outputs.
    pub fn new(num_outputs: usize) -> Self {
        Self {
            outputs: (0..num_outputs).map(|_| Vec::new()).collect(),
        }
    }

    /// Number of scene-output mix busses.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Resize the number of outputs (API thread only).
    ///
    /// Truncates when shrinking; appends empty tap lists when growing.
    pub fn resize(&mut self, num_outputs: usize) {
        if num_outputs < self.outputs.len() {
            self.outputs.truncate(num_outputs);
        } else {
            while self.outputs.len() < num_outputs {
                self.outputs.push(Vec::new());
            }
        }
    }

    /// Add (or replace) a pull on an output.
    ///
    /// A tap for the same `(source_idx, channel)` replaces the existing gain;
    /// otherwise the tap is appended. API thread only.
    pub fn set_pull(&mut self, output: usize, entry: PatchEntry) {
        if output >= self.outputs.len() {
            return;
        }
        let taps = &mut self.outputs[output];
        match taps
            .iter_mut()
            .find(|e| e.source_idx == entry.source_idx && e.channel == entry.channel)
        {
            Some(existing) => existing.gain_linear = entry.gain_linear,
            None => taps.push(entry),
        }
    }

    /// Remove every pull tapping `(source_idx, channel)` from an output.
    /// No-op if none match. API thread only.
    pub fn remove_pull(&mut self, output: usize, source_idx: usize, channel: usize) {
        if output >= self.outputs.len() {
            return;
        }
        self.outputs[output].retain(|e| e.source_idx != source_idx || e.channel != channel);
    }

    /// Update the linear gain of an existing pull. No-op if absent. API thread only.
    pub fn set_pull_gain(&mut self, output: usize, source_idx: usize, channel: usize, gain_linear: f32) {
        if output >= self.outputs.len() {
            return;
        }
        if let Some(existing) = self.outputs[output]
            .iter_mut()
            .find(|e| e.source_idx == source_idx && e.channel == channel)
        {
            existing.gain_linear = gain_linear;
        }
    }

    /// The current taps for an output (read-only).
    pub fn pulls(&self, output: usize) -> &[PatchEntry] {
        self.outputs
            .get(output)
            .map(|taps| taps.as_slice())
            .unwrap_or(&[])
    }

    /// Mix all taps into the output busses. ZERO ALLOCATION.
    ///
    /// `sources`: one buffer per Source (each with its own channel count).
    /// `outputs`: one MONO buffer per SceneOutput (already sized `num_outputs`).
    ///
    /// Each output is cleared to zero, then every tap accumulates
    /// `sources[source_idx].channel[channel][i] × gain_linear`.
    /// Out-of-range `source_idx`/`channel` are silently skipped (defensive;
    /// no panic). Samples beyond the shorter of the source/output channel are
    /// left untouched.
    pub fn process(&mut self, sources: &[&AudioBuffer], outputs: &mut [AudioBuffer]) {
        let n_out = self.outputs.len().min(outputs.len());
        for o in 0..n_out {
            outputs[o].clear();
            for e in &self.outputs[o] {
                if e.source_idx >= sources.len() {
                    continue;
                }
                let src = sources[e.source_idx];
                if e.channel >= src.channels() as usize {
                    continue;
                }
                let src_ch = src.channel(e.channel as u16);
                let out_ch = outputs[o].channel_mut(0);
                let n = out_ch.len().min(src_ch.len());
                let gain = e.gain_linear;
                for i in 0..n {
                    out_ch[i] += src_ch[i] * gain;
                }
            }
        }
    }
}
