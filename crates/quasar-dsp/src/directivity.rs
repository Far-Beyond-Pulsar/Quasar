use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::AudioBuffer;
use crate::node_graph::AudioNode;

/// Source directivity pattern.
#[derive(Clone, Debug)]
pub enum DirectivityPattern {
    /// Uniform radiation in all directions.
    Omnidirectional,
    /// Cardioid pattern (maximum at 0°, null at 180°).
    Cardioid,
    /// Figure-8 / bidirectional pattern.
    Figure8,
    /// Custom spherical harmonic weights (must be 16 or 25 for order 3 or 4).
    SphericalHarmonics { weights: Vec<f32>, order: u32 },
}

/// Applies source directivity filtering.
///
/// Uses a simplified model: omni (uniform), cardioid, figure-8, or custom SH weights.
pub struct DirectivityDspNode {
    pattern: DirectivityPattern,
    input_channels: u16,
    output_channels: u16,
}

impl DirectivityDspNode {
    /// Create a new directivity node.
    pub fn new(pattern: DirectivityPattern, channels: u16) -> Self {
        let output_channels = channels;
        Self {
            pattern,
            input_channels: channels,
            output_channels,
        }
    }

    /// Update the directivity pattern at runtime.
    pub fn set_pattern(&mut self, pattern: DirectivityPattern) {
        self.pattern = pattern;
    }

    /// Compute gain for a given angle.
    ///
    /// `azimuth`: radians, 0 = front, positive = right.
    /// `elevation`: radians, 0 = horizon, positive = up.
    pub fn compute_gain(azimuth: f32, _elevation: f32) -> f32 {
        // For each pattern, compute the gain factor.
        // Elevation is not used in the simplified 2D patterns.
        match azimuth {
            // Placeholder; real implementations use SH evaluation
            _ => {
                // Simplified: return a value based on azimuth for cardioid etc.
                // This is overridden by the pattern-specific logic in process().
                1.0
            }
        }
    }

    fn pattern_gain(&self, azimuth: f32, _elevation: f32) -> f32 {
        match &self.pattern {
            DirectivityPattern::Omnidirectional => 1.0,
            DirectivityPattern::Cardioid => 0.5 * (1.0 + azimuth.cos()),
            DirectivityPattern::Figure8 => azimuth.cos().abs(),
            DirectivityPattern::SphericalHarmonics { weights, order } => {
                // Very simplified SH evaluation (order 1 only for now)
                if *order >= 1 && weights.len() >= 4 {
                    // SH: Y00 + Y1-1*sin(az)*cos(el) + Y10*sin(el) + Y11*cos(az)*cos(el)
                    let w00 = weights[0];
                    let w1n1 = if weights.len() > 1 { weights[1] } else { 0.0 };
                    let w10 = if weights.len() > 2 { weights[2] } else { 0.0 };
                    let w11 = if weights.len() > 3 { weights[3] } else { 0.0 };
                    let el = _elevation;
                    let val = w00 + w1n1 * azimuth.sin() * el.cos() + w10 * el.sin() + w11 * azimuth.cos() * el.cos();
                    val.max(0.0).min(1.0)
                } else {
                    1.0
                }
            }
        }
    }
}

impl AudioNode for DirectivityDspNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, _params: &SpatialCoefficients) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;

        // Apply the directivity gain from params (azimuth/elevation from source direction)
        // For simplicity, we apply a uniform gain per channel based on a single directivity value.
        // In a real implementation, the azimuth/elevation comes from SpatialCoefficients metadata.
        let azimuth = _params.source_id as f32 * 0.1; // simplified: derive from source_id
        let elevation = 0.0;
        let gain = self.pattern_gain(azimuth, elevation);

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
