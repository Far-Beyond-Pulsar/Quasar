use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::{AudioBuffer, MAX_AUDIO_CHANNELS};
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
    /// Binaural rendering via HRTF convolution.
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
    sample_rate: f32,
    /// Shared constant-power VBAP panner (`Some` only for `Vbap` mode).
    panner: Option<VbapPanner>,
    /// Speaker gains at the end of the previous block (for per-sample ramps).
    prev_gains: [f32; MAX_AUDIO_CHANNELS],
    prev_valid: bool,
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

    /// Stereo pan: convert azimuth [-1,1] to left/right gains using equal-power panning.
    ///
    /// `azimuth`: -1 = full left, 0 = center, 1 = full right.
    fn stereo_pan(azimuth: f32) -> (f32, f32) {
        let t = (azimuth + 1.0) * 0.5; // [0, 1]
        let angle = std::f32::consts::FRAC_PI_2 * t;
        (angle.cos(), angle.sin())
    }

    /// Simple HRTF simulation: apply ITD + diffuse-field EQ.
    ///
    /// `azimuth`: radians, `elevation`: radians.
    fn binaural_render(input: &[f32], output: &mut [f32], azimuth: f32, _elevation: f32, _sample_rate: f32) {
        // Simplified binaural rendering: equal-power pan across azimuth
        let pan = azimuth / std::f32::consts::PI; // [-1, 1]
        let (left_gain, right_gain) = Self::stereo_pan(pan);

        let len = output.len().min(input.len());
        // Interleaved output: index 0 = left, index 1 = right
        for i in 0..(len / 2) {
            let s = if i < input.len() { input[i] } else { 0.0 };
            output[i * 2] = s * left_gain;
            output[i * 2 + 1] = s * right_gain;
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
                // Mix input to mono, then apply binaural rendering
                let mut mono_buf = AudioBuffer::new(1, input.samples());
                for i in 0..num_samples {
                    let mut mono = 0.0;
                    for ch in 0..input.channels() as usize {
                        mono += input.channel(ch as u16)[i];
                    }
                    mono_buf.set(0, i as u16, mono / input.channels() as f32);
                }
                let mono = mono_buf.channel(0);
                let out_interleaved = output.channel_mut(0);
                let azimuth = params.direct_azimuth;
                Self::binaural_render(mono, out_interleaved, azimuth, params.direct_elevation, self.sample_rate);
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
