use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::AudioBuffer;
use crate::node_graph::AudioNode;

/// Direct-path attenuation (scalar gain only, NO delay line).
///
/// The delay line was removed because every time the delay value changes
/// (snapped, crossfaded, or otherwise) the circular buffer's read pointer
/// jumps, creating a discontinuity in the output waveform.  In the P4
/// cathedral demo the propagation delay (~108 ms at 37 m) is inaudible.
/// P3 will reintroduce delay with a proper per-read-pointer crossfader
/// that smoothly morphs between the old and new delay positions.
pub struct AirAbsorptionOcclusionNode {
    input_channels: u16,
    output_channels: u16,
}

impl AirAbsorptionOcclusionNode {
    pub fn new(input_channels: u16, _sample_rate: f32, _max_delay_secs: f32) -> Self {
        Self {
            input_channels,
            output_channels: input_channels,
        }
    }

    pub fn update_occlusion(&mut self, _attenuation: &Band8, _delay_samples: f32) {
    }

    /// Fallback for tests that use the `AudioNode` trait (delay is constant,
    /// no crossfade issues; delay value is ignored).
    pub fn process_raw(
        &mut self,
        input: &AudioBuffer,
        output: &mut AudioBuffer,
        params: &SpatialCoefficients,
        _delay_samples: f32,
    ) {
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
}

impl AudioNode for AirAbsorptionOcclusionNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &SpatialCoefficients) {
        self.process_raw(input, output, params, 0.0);
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
