use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::{AudioBuffer, MAX_AUDIO_CHANNELS};
use crate::node_graph::AudioNode;

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
}

impl MasterSpatialDecoderNode {
    /// Create a new master decoder node.
    pub fn new(mode: DecoderMode, sample_rate: f32) -> Self {
        let output_channels = Self::output_channels_for_mode(&mode);
        Self {
            mode,
            input_channels: 2,
            output_channels,
            sample_rate,
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
            DecoderMode::Vbap { layout } => {
                // Resolve every layout to an explicit speaker-position set so all
                // VBAP layouts share one panner. Named layouts use unit-vector
                // directions; Custom layouts use the user's world-space positions.
                let positions: Vec<[f32; 3]> = match layout {
                    SpeakerLayout::Stereo => vec![
                        [-0.5, 0.0, -0.866], // FL (-30°)
                        [ 0.5, 0.0, -0.866], // FR (+30°)
                    ],
                    SpeakerLayout::Surround51 => vec![
                        [-0.5, 0.0, -0.866],      // FL
                        [ 0.5, 0.0, -0.866],      // FR
                        [ 0.0, 0.0, -1.0],        // C
                        [ 0.0, -0.707, -0.707],   // LFE (below center)
                        [-0.94, 0.0, 0.342],      // SL (-110°)
                        [ 0.94, 0.0, 0.342],      // SR (+110°)
                    ],
                    SpeakerLayout::Surround714 => vec![
                        [-0.5, 0.0, -0.866],      // FL
                        [ 0.5, 0.0, -0.866],      // FR
                        [ 0.0, 0.0, -1.0],        // C
                        [ 0.0, -0.707, -0.707],   // LFE (below center)
                        [-0.94, 0.0, 0.342],      // SL (-110°)
                        [ 0.94, 0.0, 0.342],      // SR (+110°)
                        [-0.5, 0.0, 0.866],       // BL (-150°)
                        [ 0.5, 0.0, 0.866],       // BR (+150°)
                    ],
                    SpeakerLayout::Quad => vec![
                        [-0.707, 0.0, -0.707],   // FL (-45°)
                        [ 0.707, 0.0, -0.707],   // FR (+45°)
                        [-0.707, 0.0, 0.707],    // BL (-135°)
                        [ 0.707, 0.0, 0.707],    // BR (+135°)
                    ],
                    SpeakerLayout::Custom { positions } => positions.clone(),
                };

                // Reconstruct the source direction from the spatial azimuth/elevation
                // (azimuth 0 = straight ahead / -Z, +X = right), then give each speaker
                // a gain proportional to how well its position matches that direction
                // (measured from the listener origin), with cosine falloff.
                let src_dir = [
                    params.direct_azimuth.sin() * params.direct_elevation.cos(),
                    params.direct_elevation.sin(),
                    -params.direct_azimuth.cos() * params.direct_elevation.cos(),
                ];
                let mut gains = vec![0.0f32; positions.len()];
                for (i, p) in positions.iter().enumerate() {
                    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    if len <= 1e-6 { continue; }
                    let dot = (src_dir[0] * p[0] + src_dir[1] * p[1] + src_dir[2] * p[2]) / len;
                    let d = dot.clamp(-1.0, 1.0);
                    gains[i] = d.max(0.0).powi(2);
                }
                let sum: f32 = gains.iter().sum();
                if sum > 1e-6 {
                    for g in gains.iter_mut() { *g /= sum; }
                }
                let out_chs = output.channels() as usize;
                for i in 0..num_samples {
                    let mut mono = 0.0;
                    for ch in 0..input.channels() as usize {
                        mono += input.channel(ch as u16)[i];
                    }
                    mono /= input.channels() as f32;
                    for (ch, &g) in gains.iter().enumerate().take(out_chs) {
                        output.channel_mut(ch as u16)[i] = mono * g;
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
        // No state to reset
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}

// ── Zero-alloc VBAP helpers (scene pipeline) ─────────────────────────────

/// Unit-vector speaker directions for the named layouts.
///
/// Must stay in lock-step with the inline tables in
/// [`MasterSpatialDecoderNode`]'s VBAP branch (see `process`).
const STEREO_POSITIONS: [[f32; 3]; 2] = [
    [-0.5, 0.0, -0.866], // FL (-30°)
    [ 0.5, 0.0, -0.866], // FR (+30°)
];

const SURROUND51_POSITIONS: [[f32; 3]; 6] = [
    [-0.5, 0.0, -0.866],      // FL
    [ 0.5, 0.0, -0.866],      // FR
    [ 0.0, 0.0, -1.0],        // C
    [ 0.0, -0.707, -0.707],   // LFE (below center)
    [-0.94, 0.0, 0.342],      // SL (-110°)
    [ 0.94, 0.0, 0.342],      // SR (+110°)
];

const SURROUND714_POSITIONS: [[f32; 3]; 8] = [
    [-0.5, 0.0, -0.866],      // FL
    [ 0.5, 0.0, -0.866],      // FR
    [ 0.0, 0.0, -1.0],        // C
    [ 0.0, -0.707, -0.707],   // LFE (below center)
    [-0.94, 0.0, 0.342],      // SL (-110°)
    [ 0.94, 0.0, 0.342],      // SR (+110°)
    [-0.5, 0.0, 0.866],       // BL (-150°)
    [ 0.5, 0.0, 0.866],       // BR (+150°)
];

const QUAD_POSITIONS: [[f32; 3]; 4] = [
    [-0.707, 0.0, -0.707],   // FL (-45°)
    [ 0.707, 0.0, -0.707],   // FR (+45°)
    [-0.707, 0.0, 0.707],    // BL (-135°)
    [ 0.707, 0.0, 0.707],    // BR (+135°)
];

/// Resolve a speaker layout to explicit speaker directions (unit vectors).
///
/// Named layouts use the exact unit-vector positions already in
/// [`MasterSpatialDecoderNode`] (Stereo ±30°, 5.1, 7.1, Quad); Custom uses the
/// caller's positions. API thread only (allocates).
pub fn layout_positions(layout: &SpeakerLayout) -> Vec<[f32; 3]> {
    match layout {
        SpeakerLayout::Stereo => STEREO_POSITIONS.to_vec(),
        SpeakerLayout::Surround51 => SURROUND51_POSITIONS.to_vec(),
        SpeakerLayout::Surround714 => SURROUND714_POSITIONS.to_vec(),
        SpeakerLayout::Quad => QUAD_POSITIONS.to_vec(),
        SpeakerLayout::Custom { positions } => positions.clone(),
    }
}

/// Fill `out[0..n]` with per-speaker VBAP gains for the given resolved speaker
/// directions. Returns the number of speakers written (`n`).
///
/// Zero allocation. Reuses the same math as [`MasterSpatialDecoderNode`]'s VBAP
/// branch: `src_dir = [sin(az)·cos(el), sin(el), -cos(az)·cos(el)]`; per-speaker
/// gain = `(dot(src_dir, pos)/len(pos)).max(0)²`, normalized to sum 1.
///
/// `positions` comes from [`layout_positions`] (precomputed per listener on the
/// API thread); `out` is caller-owned stack scratch.
pub fn vbap_gains(
    positions: &[[f32; 3]],
    azimuth: f32,
    elevation: f32,
    out: &mut [f32; MAX_AUDIO_CHANNELS],
) -> usize {
    let n = positions.len().min(out.len());

    let src_dir = [
        azimuth.sin() * elevation.cos(),
        elevation.sin(),
        -azimuth.cos() * elevation.cos(),
    ];

    for (i, p) in positions.iter().enumerate().take(n) {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if len <= 1e-6 {
            out[i] = 0.0;
            continue;
        }
        let dot = (src_dir[0] * p[0] + src_dir[1] * p[1] + src_dir[2] * p[2]) / len;
        let d = dot.clamp(-1.0, 1.0);
        out[i] = d.max(0.0).powi(2);
    }

    let sum: f32 = out.iter().take(n).sum();
    if sum > 1e-6 {
        for g in out.iter_mut().take(n) {
            *g /= sum;
        }
    }

    n
}
