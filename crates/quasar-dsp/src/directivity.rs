use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::AudioBuffer;
use crate::node_graph::AudioNode;

/// Source directivity pattern (broadband, amplitude).
///
/// This is the standalone, broadband pattern used by [`DirectivityDspNode`]. The scene engine
/// itself evaluates the per-band cardioid family of `quasar_core::source_directivity` on the
/// compute side (per listener / emitter pair) and carries it in
/// `SpatialCoefficients::directivity_gain`; this enum is for graphs that want a directivity
/// stage driven by an explicit emission angle.
#[derive(Clone, Debug)]
pub enum DirectivityPattern {
    /// Uniform radiation in all directions.
    Omnidirectional,
    /// Cardioid pattern (maximum at 0°, null at 180°).
    Cardioid,
    /// Figure-8 / bidirectional pattern.
    Figure8,
    /// First-order spherical harmonic weights `[Y00, Y1-1, Y10, Y11]` (extra entries ignored;
    /// only order 1 is evaluated).
    SphericalHarmonics { weights: Vec<f32>, order: u32 },
}

impl DirectivityPattern {
    /// Amplitude gain (>= 0, <= 1) of the pattern toward `(azimuth, elevation)` measured from
    /// the emitter's forward axis: `azimuth` radians (0 = on axis, positive = right) and
    /// `elevation` radians (0 = horizon, positive = up). The off-axis angle `theta` follows
    /// `cos(theta) = cos(azimuth) cos(elevation)`.
    pub fn gain(&self, azimuth: f32, elevation: f32) -> f32 {
        let (az, el) = (azimuth, elevation);
        let cos_theta = az.cos() * el.cos();
        match self {
            DirectivityPattern::Omnidirectional => 1.0,
            DirectivityPattern::Cardioid => 0.5 * (1.0 + cos_theta),
            DirectivityPattern::Figure8 => cos_theta.abs(),
            DirectivityPattern::SphericalHarmonics { weights, order } => {
                if *order >= 1 && weights.len() >= 4 {
                    // Y00 + Y1-1 sin(az) cos(el) + Y10 sin(el) + Y11 cos(az) cos(el)
                    let val = weights[0]
                        + weights[1] * az.sin() * el.cos()
                        + weights[2] * el.sin()
                        + weights[3] * cos_theta;
                    val.clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
        }
    }
}

/// Applies a broadband source-directivity gain to its input.
///
/// The emission angle (direction to the listener relative to the emitter's forward axis) is an
/// explicit input: set it with [`set_angle`](Self::set_angle) (the caller derives it from the
/// emitter orientation and the emitter-to-listener vector; see
/// `quasar_core::source_directivity::emission_cos`). Until set it is `(0, 0)`, i.e. on axis.
pub struct DirectivityDspNode {
    pattern: DirectivityPattern,
    azimuth: f32,
    elevation: f32,
    input_channels: u16,
    output_channels: u16,
}

impl DirectivityDspNode {
    /// Create a new directivity node (on axis).
    pub fn new(pattern: DirectivityPattern, channels: u16) -> Self {
        let output_channels = channels;
        Self {
            pattern,
            azimuth: 0.0,
            elevation: 0.0,
            input_channels: channels,
            output_channels,
        }
    }

    /// Update the directivity pattern at runtime.
    pub fn set_pattern(&mut self, pattern: DirectivityPattern) {
        self.pattern = pattern;
    }

    /// Set the emission angle toward the listener, relative to the emitter's forward axis
    /// (`azimuth` radians, 0 = on axis, positive = right; `elevation` radians, positive = up).
    pub fn set_angle(&mut self, azimuth: f32, elevation: f32) {
        self.azimuth = if azimuth.is_finite() { azimuth } else { 0.0 };
        self.elevation = if elevation.is_finite() { elevation } else { 0.0 };
    }

    /// Gain of the current pattern at the current emission angle.
    pub fn current_gain(&self) -> f32 {
        self.pattern.gain(self.azimuth, self.elevation)
    }

    /// Gain of `pattern` toward `(azimuth, elevation)` (see [`DirectivityPattern::gain`]).
    pub fn compute_gain(pattern: &DirectivityPattern, azimuth: f32, elevation: f32) -> f32 {
        pattern.gain(azimuth, elevation)
    }
}

impl AudioNode for DirectivityDspNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, _params: &SpatialCoefficients) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;
        let gain = self.current_gain();

        for ch in 0..output.channels() as usize {
            let in_ch = input.channel(ch as u16);
            let out_ch = output.channel_mut(ch as u16);
            for i in 0..num_samples {
                out_ch[i] = in_ch[i] * gain;
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
