use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::AudioBuffer;
use crate::node_graph::AudioNode;

/// Applies direct-path distance attenuation (scalar gain only, NO delay line).
///
/// The delay line (`HermiteInterpolatingDelayLine`) and the earlier biquad
/// shading have both been removed because they produced audible artifacts:
///
/// **Delay line** — the fractional Hermite interpolation through the circular
/// buffer was reading partially overwritten data when the physical delay
/// (~108 ms at 37 m, 5178 samples) exceeded the buffer's wrap period,
/// causing periodic garbage injection.
///
/// **Biquad** — the ad-hoc cutoff-mapping formula plus stale filter state
/// (coefficients changed every block without state resets) created click
/// trains at every crossfade update.
///
/// For the P4 demo, the backend's `direct_gain.mean()` is applied purely as
/// a sample-wise scalar — no buffer, no filter, no state. The loss of the
/// fractional delay (the direct-path propagation delay) is inaudible in a
/// 37 m static scene. P3 will reintroduce delay with a proper crossfading
/// read pointer.
pub struct AirAbsorptionOcclusionNode {
    input_channels: u16,
    output_channels: u16,
}

impl AirAbsorptionOcclusionNode {
    /// Create a new occlusion node (no delay line, no biquads).
    pub fn new(input_channels: u16, _sample_rate: f32, _max_delay_secs: f32) -> Self {
        Self {
            input_channels,
            output_channels: input_channels,
        }
    }

    /// No-op — direct_gain.mean() in [`process`] handles all attenuation.
    pub fn update_occlusion(&mut self, _attenuation: &Band8, _delay_samples: f32) {
    }
}

impl AudioNode for AirAbsorptionOcclusionNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &SpatialCoefficients) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;
        let gain = params.direct_gain.mean();

        for ch in 0..self.input_channels as usize {
            let in_ch = input.channel(ch as u16);
            let out_ch = output.channel_mut(ch as u16);

            for i in 0..num_samples {
                out_ch[i] = in_ch[i] * gain;
            }
        }
    }

    fn reset(&mut self) {
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}
