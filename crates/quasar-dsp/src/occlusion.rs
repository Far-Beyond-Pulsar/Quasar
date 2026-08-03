use quasar_core::param_exchange::SpatialCoefficients;
use quasar_core::bands::Band8;
use crate::audio_buffer::AudioBuffer;
use crate::fractional_delay::HermiteInterpolatingDelayLine;
use crate::node_graph::AudioNode;

/// A single biquad IIR filter section (Direct Form 1).
#[derive(Clone, Debug)]
pub struct BiquadFilter {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    x1: f32, x2: f32,
    y1: f32, y2: f32,
}

impl BiquadFilter {
    /// Create a new biquad filter with unity gain (all-pass).
    pub fn new() -> Self {
        Self {
            b0: 1.0, b1: 0.0, b2: 0.0,
            a1: 0.0, a2: 0.0,
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
        }
    }

    /// Set filter coefficients directly.
    pub fn set_coefficients(&mut self, b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) {
        self.b0 = b0;
        self.b1 = b1;
        self.b2 = b2;
        self.a1 = a1;
        self.a2 = a2;
    }

    /// Process a single sample through the filter.
    pub fn process(&mut self, sample: f32) -> f32 {
        let y = self.b0 * sample + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = sample;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Reset filter state to zero.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Configure as a lowpass filter with given cutoff frequency.
    pub fn set_lowpass(&mut self, cutoff_hz: f32, sample_rate: f32) {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let q = 0.7071; // Butterworth Q
        let alpha = sin_w0 / (2.0 * q);

        let inv_a0 = 1.0 / (1.0 + alpha);
        let b0 = (1.0 - cos_w0) * 0.5 * inv_a0;
        let b1 = (1.0 - cos_w0) * inv_a0;
        let b2 = (1.0 - cos_w0) * 0.5 * inv_a0;
        let a1 = (-2.0 * cos_w0) * inv_a0;
        let a2 = (1.0 - alpha) * inv_a0;

        self.set_coefficients(b0, b1, b2, a1, a2);
    }

    /// Configure as a highshelf filter with given gain dB and frequency.
    pub fn set_highshelf(&mut self, gain_db: f32, freq_hz: f32, sample_rate: f32) {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let s = 1.0; // shelf slope parameter
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).sqrt();

        let inv_a0 = 1.0 / ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha) * inv_a0;
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) * inv_a0;
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha) * inv_a0;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) * inv_a0;
        let a2 = ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha) * inv_a0;

        self.set_coefficients(b0, b1, b2, a1, a2);
    }
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies air absorption, occlusion filtering, and delay for the direct path.
pub struct AirAbsorptionOcclusionNode {
    /// One filter per band per output channel
    filters: Vec<BiquadFilter>,
    /// Fractional delay line for the direct path
    delay_line: HermiteInterpolatingDelayLine,
    input_channels: u16,
    output_channels: u16,
    sample_rate: f32,
}

impl AirAbsorptionOcclusionNode {
    /// Create a new occlusion node.
    pub fn new(input_channels: u16, sample_rate: f32, max_delay_secs: f32) -> Self {
        let num_filters = 8 * input_channels as usize; // 8 bands per channel
        let filters = (0..num_filters).map(|_| BiquadFilter::new()).collect();
        let delay_line = HermiteInterpolatingDelayLine::new(max_delay_secs, sample_rate);
        Self {
            filters,
            delay_line,
            input_channels,
            output_channels: input_channels,
            sample_rate,
        }
    }

    /// Apply 8-band occlusion gain by adjusting biquad coefficients.
    ///
    /// Called when spatial coefficients are updated.
    pub fn update_occlusion(&mut self, attenuation: &Band8, _delay_samples: f32) {
        // Convert per-band attenuation to lowpass filter coefficients
        for ch in 0..self.input_channels as usize {
            for band in 0..8 {
                let idx = ch * 8 + band;
                if idx < self.filters.len() {
                    let atten = attenuation.0[band].max(0.001).min(1.0);
                    // Map attenuation to cutoff frequency: lower atten = lower cutoff
                    let bandwidth = FREQ_BAND_WIDTHS[band];
                    let centre = quasar_core::bands::FREQ_BAND_CENTRES[band];
                    let cutoff = centre * atten.sqrt() + 20.0; // never go below 20 Hz
                    let cutoff = cutoff.min(bandwidth * 0.5);
                    self.filters[idx].set_lowpass(cutoff, self.sample_rate);
                }
            }
        }
    }
}

/// Approximate bandwidth per octave band (Hz).
const FREQ_BAND_WIDTHS: [f32; 8] = [
    44.0, 88.0, 177.0, 355.0, 710.0, 1420.0, 2840.0, 5680.0,
];

impl AudioNode for AirAbsorptionOcclusionNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, params: &SpatialCoefficients) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;
        // Average direct gain across bands for distance/occlusion attenuation
        let gain = params.direct_gain.mean();

        for ch in 0..self.input_channels as usize {
            let in_ch = input.channel(ch as u16);
            let out_ch = output.channel_mut(ch as u16);

            for i in 0..num_samples {
                let mut sample = in_ch[i];
                for band in 0..8 {
                    let idx = ch * 8 + band;
                    if idx < self.filters.len() {
                        sample = self.filters[idx].process(sample);
                    }
                }
                self.delay_line.push(sample);
                out_ch[i] = self.delay_line.tap(params.direct_delay_samples) * gain;
            }
        }
    }

    fn reset(&mut self) {
        for f in self.filters.iter_mut() {
            f.reset();
        }
        self.delay_line.clear();
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}
