use quasar_core::param_exchange::{SpatialCoefficients, EarlyReflectionCoeffs};
use crate::audio_buffer::AudioBuffer;
use crate::fractional_delay::HermiteInterpolatingDelayLine;
use crate::node_graph::AudioNode;

/// A single early reflection tap.
struct ReflectionTap {
    delay_samples: f32,
    gain: f32,
}

/// Multi-tap early reflection processor (MONO contribution).
///
/// Feeds a shared delay line with multiple tapped outputs, each representing one
/// specular early reflection path with its own delay. The contribution is mono:
/// each tap's gain is the average per-band reflection gain and the pan from
/// `EarlyReflectionCoeffs.azimuth` is ignored. Spatialized (per-direction)
/// reflections land in P3; this stage feeds the listener's mono mix.
pub struct EarlyReflectionDelayNode {
    delay_line: HermiteInterpolatingDelayLine,
    taps: Vec<ReflectionTap>,
    input_channels: u16,
    output_channels: u16,
    _sample_rate: f32,
    _max_delay_secs: f32,
}

impl EarlyReflectionDelayNode {
    /// Create a new early reflection delay node.
    ///
    /// `max_reflections`: maximum number of early reflection taps to support.
    pub fn new(input_channels: u16, sample_rate: f32, max_delay_secs: f32, max_reflections: usize) -> Self {
        let delay_line = HermiteInterpolatingDelayLine::new(max_delay_secs, sample_rate);
        let taps = Vec::with_capacity(max_reflections);
        Self {
            delay_line,
            taps,
            input_channels,
            output_channels: 1,
            _sample_rate: sample_rate,
            _max_delay_secs: max_delay_secs,
        }
    }

    /// Update reflection taps from spatial coefficients.
    pub fn update_reflections(&mut self, reflections: &[EarlyReflectionCoeffs]) {
        self.taps.clear();
        for r in reflections {
            // Mono early-reflection contribution; pan (azimuth) is spatialized in P3.
            let avg_gain = r.gain.0.iter().sum::<f32>() / 8.0;
            self.taps.push(ReflectionTap {
                delay_samples: r.delay_samples,
                gain: avg_gain,
            });
        }
    }
}

impl AudioNode for EarlyReflectionDelayNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, _params: &SpatialCoefficients) {
        debug_assert!(input.channels() >= 1);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;

        output.channel_mut(0).fill(0.0);

        for i in 0..num_samples {
            let mut mono = 0.0_f32;
            for ch in 0..input.channels() as usize {
                mono += input.channel(ch as u16)[i];
            }
            mono /= input.channels() as f32;

            self.delay_line.push(mono);

            for tap in &self.taps {
                let delayed = self.delay_line.tap(tap.delay_samples);
                output.channel_mut(0)[i] += delayed * tap.gain;
            }
        }
    }

    fn reset(&mut self) {
        self.delay_line.clear();
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}
