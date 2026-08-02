use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use crate::audio_buffer::AudioBuffer;
use crate::fractional_delay::HermiteInterpolatingDelayLine;
use crate::occlusion::BiquadFilter;
use crate::node_graph::AudioNode;

/// Implementation of a Feedback Delay Network reverberator.
///
/// Architecture:
/// - 16 mutually-coupled delay lines of varying lengths (typically 15-80ms)
/// - Hadamard-based orthogonal feedback matrix for maximum echo density
/// - Per-line one-pole lowpass filters for frequency-dependent decay (T60 control)
/// - Modulation on delay lines for chorusing effect (smooths out metallic artifacts)
///
/// All memory allocated at construction. NEVER allocates during process().
pub struct FdnReverbNode {
    /// Delay lines
    delay_lines: [HermiteInterpolatingDelayLine; 16],
    /// Damping lowpass filters (one per delay line)
    dampings: [BiquadFilter; 16],
    /// Modulated delay offsets for chorus effect
    modulation_phase: [f32; 16],
    /// Current T60 target
    t60: Band8,
    /// Pre-delay line
    pre_delay: HermiteInterpolatingDelayLine,
    /// Current pre-delay in samples
    pre_delay_samples: f32,
    /// Mix parameters
    wet_gain: f32,
    dry_gain: f32,
    /// Sample rate
    sample_rate: f32,
    /// Number of input/output channels
    input_channels: u16,
    output_channels: u16,
}

// Prime-based coprime delay lengths (in samples at 48 kHz) for ~15-80ms range.
const FDN_DELAY_LENGTHS: [usize; 16] = [
    719,   // ~15.0 ms
    857,   // ~17.9 ms
    1103,  // ~23.0 ms
    1321,  // ~27.5 ms
    1613,  // ~33.6 ms
    1871,  // ~39.0 ms
    2213,  // ~46.1 ms
    2657,  // ~55.4 ms
    3079,  // ~64.1 ms
    3491,  // ~72.7 ms
    109,   // ~2.3 ms  (for short modulation)
    151,   // ~3.1 ms
    197,   // ~4.1 ms
    251,   // ~5.2 ms
    313,   // ~6.5 ms
    401,   // ~8.4 ms
];

impl FdnReverbNode {
    /// Create a new FDN reverb.
    pub fn new(input_channels: u16, sample_rate: f32) -> Self {
        let max_delay = 0.15;
        let max_pre_delay = 0.1;

        let delay_lines = [
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
            HermiteInterpolatingDelayLine::new(max_delay, sample_rate),
        ];

        let dampings = [
            BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(),
            BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(),
            BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(),
            BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(), BiquadFilter::new(),
        ];

        let pre_delay = HermiteInterpolatingDelayLine::new(max_pre_delay, sample_rate);

        Self {
            delay_lines,
            dampings,
            modulation_phase: [0.0; 16],
            t60: Band8::splat(2.0),
            pre_delay,
            pre_delay_samples: 0.0,
            wet_gain: 0.5,
            dry_gain: 0.5,
            sample_rate,
            input_channels,
            output_channels: input_channels,
        }
    }

    /// Build the primitive root delay line lengths.
    pub fn init_delay_lengths(&mut self, _sample_rate: f32) {
        // Delay lines are pre-configured with max_delay; lengths are managed internally
        // via the fixed FDN_DELAY_LENGTHS constants.
    }

    /// Compute the Hadamard feedback matrix (Householder reflection).
    pub fn feedback_matrix(input: &[f32; 16]) -> [f32; 16] {
        // Householder reflection: H = I - (2/n) * u*u^T where u = [1,1,...,1]
        // This gives an orthogonal matrix optimal for FDN echo density.
        let n = 16.0;
        let sum: f32 = input.iter().sum();
        let scale = 2.0 / n;
        let mut out = [0.0_f32; 16];
        for i in 0..16 {
            out[i] = -input[i] + scale * sum;
        }
        out
    }

    /// Set T60 target per band. Adjusts damping filter coefficients.
    pub fn set_t60(&mut self, t60: &Band8) {
        self.t60 = *t60;
        // Map average T60 to damping coefficient
        let avg_t60 = t60.mean().max(0.1).min(10.0);
        // Higher T60 → lower damping
        let damping = (-3.0 / (avg_t60 * self.sample_rate)).exp();
        for i in 0..16 {
            let b0 = 1.0 - damping;
            let a1 = -damping;
            self.dampings[i].set_coefficients(b0, 0.0, 0.0, a1, 0.0);
        }
    }

    /// Set wet/dry mix.
    pub fn set_mix(&mut self, wet: f32, dry: f32) {
        self.wet_gain = wet;
        self.dry_gain = dry;
    }

    /// Set pre-delay in seconds.
    pub fn set_pre_delay(&mut self, delay_secs: f32) {
        self.pre_delay_samples = delay_secs * self.sample_rate;
    }

    /// Update reverb parameters from spatial coefficients.
    pub fn update_from_coefficients(&mut self, params: &SpatialCoefficients) {
        self.set_t60(&params.late_t60);
        self.wet_gain = 10.0_f32.powf(params.late_gain_db / 20.0);
    }

    fn process_fdn_channel(&mut self, input_sample: f32) -> f32 {
        // Write input into pre-delay
        self.pre_delay.push(input_sample);
        let signal = self.pre_delay.tap(self.pre_delay_samples);

        // Read current state of all delay lines
        let mut vec_in = [0.0_f32; 16];
        for i in 0..16 {
            // Add modulation for chorusing (slow LFO)
            self.modulation_phase[i] += 0.5 / self.sample_rate;
            if self.modulation_phase[i] > 1.0 {
                self.modulation_phase[i] -= 1.0;
            }
            let mod_offset = (self.modulation_phase[i] * std::f32::consts::TAU).sin() * 2.0;

            let tap_pos = (FDN_DELAY_LENGTHS[i] as f32 + mod_offset)
                .min(self.delay_lines[i].max_samples() as f32 - 3.0)
                .max(0.0);
            vec_in[i] = self.delay_lines[i].tap(tap_pos);
        }

        // Apply damping filters
        for i in 0..16 {
            vec_in[i] = self.dampings[i].process(vec_in[i]);
        }

        // Apply feedback matrix (Householder reflection)
        let vec_out = Self::feedback_matrix(&vec_in);

        // Write back to delay lines with input signal injected
        for i in 0..16 {
            let feedback = vec_out[i] * 0.85; // feedback gain for stability
            self.delay_lines[i].push(signal * (1.0 / 16.0) + feedback);
        }

        // Sum all delay lines for output
        vec_out.iter().sum::<f32>() * (1.0 / 16.0_f32.sqrt())
    }
}

impl AudioNode for FdnReverbNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, _params: &SpatialCoefficients) {
        debug_assert_eq!(input.channels(), self.input_channels);
        debug_assert_eq!(output.channels(), self.output_channels);
        debug_assert_eq!(input.samples(), output.samples());

        let num_samples = input.samples() as usize;

        for ch in 0..self.input_channels as usize {
            let in_ch = input.channel(ch as u16);
            let out_ch = output.channel_mut(ch as u16);

            for i in 0..num_samples {
                let dry = in_ch[i] * self.dry_gain;
                let wet = self.process_fdn_channel(in_ch[i]) * self.wet_gain;
                out_ch[i] = dry + wet;
            }
        }
    }

    fn reset(&mut self) {
        for dl in self.delay_lines.iter_mut() {
            dl.clear();
        }
        for d in self.dampings.iter_mut() {
            d.reset();
        }
        self.pre_delay.clear();
        self.modulation_phase = [0.0; 16];
    }

    fn input_channels(&self) -> u16 {
        self.input_channels
    }

    fn output_channels(&self) -> u16 {
        self.output_channels
    }
}
