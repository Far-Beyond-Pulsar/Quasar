use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use crate::binaural::{BinauralRenderer, ParametricBinauralRenderer};
use crate::node_graph::AudioNode;
use crate::vbap::VbapPanner;

/// Speaker layout for VBAP panning.
#[derive(Clone, Debug)]
pub enum SpeakerLayout {
    Stereo,
    Surround51,
    Surround714,
    Quad,
    Custom { positions: Vec<[f32; 3]> },
}

/// Decoding mode for the master output.
#[derive(Clone, Debug)]
pub enum DecoderMode {
    /// Binaural rendering via the parametric model in [`crate::binaural`] (not a measured HRTF).
    BinauralHrtf,
    /// Vector-Based Amplitude Panning for speaker arrays.
    Vbap { layout: SpeakerLayout },
    /// Higher-Order Ambisonics decoding.
    AmbisonicDecode { order: u32 },
}

/// Master spatial decoder node.
///
/// Converts the internal spatial audio representation to the final output format.
pub struct MasterSpatialDecoderNode {
    mode: DecoderMode,
    input_channels: u16,
    output_channels: u16,
    #[allow(dead_code)]
    sample_rate: f32,
    /// Shared constant-power VBAP panner (`Some` only for `Vbap` mode).
    panner: Option<VbapPanner>,
    /// Speaker gains at the end of the previous block (for per-sample ramps).
    prev_gains: [f32; MAX_AUDIO_CHANNELS],
    prev_valid: bool,
    /// Parametric binaural renderer (`Some` only for `BinauralHrtf` mode).
    binaural: Option<ParametricBinauralRenderer>,
    /// Mono downmix scratch for the binaural path (preallocated, one block).
    mono_scratch: Vec<f32>,
}

impl MasterSpatialDecoderNode {
    /// Create a new master decoder node.
    pub fn new(mode: DecoderMode, sample_rate: f32) -> Self {
        let output_channels = Self::output_channels_for_mode(&mode);
        Self {
            input_channels: 2,
            output_channels,
            sample_rate,
            panner: match &mode {
                DecoderMode::Vbap { layout } => Some(layout_panner(layout)),
                _ => None,
            },
            prev_gains: [0.0; MAX_AUDIO_CHANNELS],
            prev_valid: false,
            binaural: match &mode {
                DecoderMode::BinauralHrtf => Some(ParametricBinauralRenderer::with_sample_rate(sample_rate)),
                _ => None,
            },
            mono_scratch: vec![0.0; DEFAULT_BLOCK_SIZE],
            mode,
        }
    }

    /// Get the expected number of output channels for a given mode.
    pub fn output_channels_for_mode(mode: &DecoderMode) -> u16 {
        match mode {
            DecoderMode::BinauralHrtf => 2,
            DecoderMode::Vbap { layout } => match layout {
                SpeakerLayout::Stereo => 2,
                SpeakerLayout::Surround51 => 6,
                SpeakerLayout::Surround714 => 8,
                SpeakerLayout::Quad => 4,
                SpeakerLayout::Custom { positions } => positions.len() as u16,
            },
            DecoderMode::AmbisonicDecode { order } => ((order + 1) * (order + 1)) as u16,
        }
    }
}

impl AudioNode for MasterSpatialDecoderNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &SpatialCoefficients) {
        debug_assert!(input.channels() >= 1);
        debug_assert_eq!(output.channels(), self.output_channels);

        let num_samples = input.samples() as usize;
        output.clear();

        match &self.mode {
            DecoderMode::BinauralHrtf => {
                // Mix input to mono (preallocated scratch), then render it through
                // the parametric binaural model (see `crate::binaural`) into L/R.
                let in_chs = input.channels() as usize;
                let inv_in = 1.0 / in_chs.max(1) as f32;
                let (left, right) = output.stereo_mut();
                if let Some(bin) = self.binaural.as_mut() {
                    let mut done = 0;
                    while done < num_samples {
                        let n = (num_samples - done).min(self.mono_scratch.len());
                        for i in 0..n {
                            let mut mono = 0.0;
                            for c in 0..in_chs {
                                mono += input.channel(c as u16)[done + i];
                            }
                            self.mono_scratch[i] = mono * inv_in;
                        }
                        bin.render_add(
                            &self.mono_scratch[..n],
                            params.direct_azimuth,
                            params.direct_elevation,
                            &mut left[done..done + n],
                            &mut right[done..done + n],
                        );
                        done += n;
                    }
                }
            }
            DecoderMode::Vbap { .. } => {
                // Constant-power VBAP via the shared panner (LFE slots stay silent),
                // with a per-sample linear gain ramp from the previous block's gains
                // so block-rate pan changes do not zipper.
                let mut target = [0.0_f32; MAX_AUDIO_CHANNELS];
                let out_chs = (output.channels() as usize).min(MAX_AUDIO_CHANNELS);
                if let Some(panner) = &self.panner {
                    panner.gains(params.direct_azimuth, params.direct_elevation, &mut target[..out_chs]);
                }
                if !self.prev_valid {
                    self.prev_gains = target;
                    self.prev_valid = true;
                }
                let in_chs = input.channels() as usize;
                let inv_in = 1.0 / in_chs.max(1) as f32;
                for ch in 0..out_chs {
                    let g0 = self.prev_gains[ch];
                    let g1 = target[ch];
                    self.prev_gains[ch] = g1;
                    if g0 == 0.0 && g1 == 0.0 {
                        continue;
                    }
                    let step = (g1 - g0) / num_samples.max(1) as f32;
                    let dst = output.channel_mut(ch as u16);
                    for i in 0..num_samples {
                        let mut mono = 0.0;
                        for c in 0..in_chs {
                            mono += input.channel(c as u16)[i];
                        }
                        dst[i] = mono * inv_in * (g0 + step * (i + 1) as f32);
                    }
                }
            }
            DecoderMode::AmbisonicDecode { order: _order } => {
                // Simple pass-through for ambisonics (first 2 channels)
                for i in 0..num_samples {
                    if output.channels() > 0 && input.channels() > 0 {
                        output.channel_mut(0)[i] = input.channel(0)[i];
                    }
                    if output.channels() > 1 && input.channels() > 1 {
                        output.channel_mut(1)[i] = input.channel(1)[i];
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.prev_valid = false;
        if let Some(b) = self.binaural.as_mut() {
            b.reset();
        }
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}

// ── VBAP layout helpers (scene pipeline) ─────────────────────────────────

/// Unit-vector speaker directions for the named layouts (azimuth 0 = -Z,
/// +azimuth toward +X). Slot order follows the WASAPI / SMPTE channel order.
const STEREO_POSITIONS: [[f32; 3]; 2] = [
    [-0.5, 0.0, -0.866], // FL (-30°)
    [ 0.5, 0.0, -0.866], // FR (+30°)
];

/// 5.1: FL FR C LFE BL BR (surrounds at ±110°). LFE (slot 3) is never panned to.
const SURROUND51_POSITIONS: [[f32; 3]; 6] = [
    [-0.5, 0.0, -0.866],      // FL (-30°)
    [ 0.5, 0.0, -0.866],      // FR (+30°)
    [ 0.0, 0.0, -1.0],        // C
    [ 0.0, 0.0, -1.0],        // LFE (position unused)
    [-0.94, 0.0, 0.342],      // BL (-110°)
    [ 0.94, 0.0, 0.342],      // BR (+110°)
];

/// 7.1: FL FR C LFE BL BR SL SR (standard order). LFE (slot 3) is never panned to.
const SURROUND714_POSITIONS: [[f32; 3]; 8] = [
    [-0.5, 0.0, -0.866],      // FL (-30°)
    [ 0.5, 0.0, -0.866],      // FR (+30°)
    [ 0.0, 0.0, -1.0],        // C
    [ 0.0, 0.0, -1.0],        // LFE (position unused)
    [-0.5, 0.0, 0.866],       // BL (-150°)
    [ 0.5, 0.0, 0.866],       // BR (+150°)
    [-1.0, 0.0, 0.0],         // SL (-90°)
    [ 1.0, 0.0, 0.0],         // SR (+90°)
];

const QUAD_POSITIONS: [[f32; 3]; 4] = [
    [-0.707, 0.0, -0.707],   // FL (-45°)
    [ 0.707, 0.0, -0.707],   // FR (+45°)
    [-0.707, 0.0, 0.707],    // BL (-135°)
    [ 0.707, 0.0, 0.707],    // BR (+135°)
];

/// Resolve a speaker layout to explicit speaker directions (unit vectors).
///
/// Named layouts use fixed unit-vector positions; Custom uses the caller's
/// positions. API thread only (allocates).
pub fn layout_positions(layout: &SpeakerLayout) -> Vec<[f32; 3]> {
    match layout {
        SpeakerLayout::Stereo => STEREO_POSITIONS.to_vec(),
        SpeakerLayout::Surround51 => SURROUND51_POSITIONS.to_vec(),
        SpeakerLayout::Surround714 => SURROUND714_POSITIONS.to_vec(),
        SpeakerLayout::Quad => QUAD_POSITIONS.to_vec(),
        SpeakerLayout::Custom { positions } => positions.clone(),
    }
}

/// Output slots that carry the LFE channel (never receive panned signal).
///
/// Named 5.1 and 7.1 layouts have LFE at index 3; every other layout
/// (including Custom) has none.
pub fn layout_lfe(layout: &SpeakerLayout) -> &'static [usize] {
    match layout {
        SpeakerLayout::Surround51 | SpeakerLayout::Surround714 => &[3],
        _ => &[],
    }
}

/// Build the constant-power VBAP panner for a layout (positions + LFE slots).
/// API thread only (allocates).
pub fn layout_panner(layout: &SpeakerLayout) -> VbapPanner {
    VbapPanner::new(&layout_positions(layout), layout_lfe(layout))
}
