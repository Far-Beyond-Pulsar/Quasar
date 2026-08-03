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
        let q = 0.7071;
        let alpha = sin_w0 / (2.0 * q);

        let inv_a0 = 1.0 / (1.0 + alpha);
        let b0 = (1.0 - cos_w0) * 0.5 * inv_a0;
        let b1 = (1.0 - cos_w0) * inv_a0;
        let b2 = (1.0 - cos_w0) * 0.5 * inv_a0;
        let a1 = (-2.0 * cos_w0) * inv_a0;
        let a2 = (1.0 - alpha) * inv_a0;

        self.set_coefficients(b0, b1, b2, a1, a2);
    }
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new()
    }
}
