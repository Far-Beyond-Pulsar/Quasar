/// A variable fractional delay line using 4-point Hermite interpolation.
///
/// Formula: `p(t) = (2t³ - 3t² + 1)·v₀ + (t³ - 2t² + t)·m₀ + (-2t³ + 3t²)·v₁ + (t³ - t²)·m₁`
/// where `m₀ = (v₁ - v₋₁)/2`, `m₁ = (v₂ - v₀)/2` (Catmull-Rom tangents).
///
/// All memory allocated at construction. NEVER allocates during processing.
pub struct HermiteInterpolatingDelayLine {
    buffer: Vec<f32>,
    max_samples: usize,
    write_pos: usize,
    sample_rate: f32,
}

impl HermiteInterpolatingDelayLine {
    /// Create a delay line with `max_delay_secs` capacity.
    ///
    /// This is the ONLY allocation — called at init time.
    pub fn new(max_delay_secs: f32, sample_rate: f32) -> Self {
        let max_samples = (max_delay_secs * sample_rate).ceil() as usize + 4;
        let buffer = vec![0.0; max_samples];
        Self {
            buffer,
            max_samples,
            write_pos: 0,
            sample_rate,
        }
    }

    /// Maximum delay in samples.
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    /// Write a sample into the delay line.
    pub fn push(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.max_samples;
    }

    /// Read a sample at the given delay (fractional).
    ///
    /// `delay_samples`: fractional sample delay in `[0, max_samples - 3]`
    /// (clamped, also in release builds; NaN reads as 0). Delay 0 is the most
    /// recently pushed sample, delay `n` the sample pushed `n` pushes earlier.
    /// Uses 4-point Hermite (Catmull-Rom) interpolation.
    pub fn tap(&self, delay_samples: f32) -> f32 {
        let max_delay = (self.max_samples - 3) as f32;
        // `max`/`min` also map NaN to a finite value.
        let delay_samples = delay_samples.max(0.0).min(max_delay);
        let int_delay = delay_samples as usize;
        let frac = delay_samples - int_delay as f32;

        // `idx(k)` is the sample that is `k` samples old (k = 0: newest).
        let idx = |k: usize| -> f32 {
            let pos = (self.write_pos + self.max_samples - 1 - k) % self.max_samples;
            self.buffer[pos]
        };

        // Neighbours in age order: v_m1 is newer than v0, v1/v2 are older.
        // The interpolation runs from v0 (age n) towards v1 (age n + 1) with
        // t = frac, so a larger fraction means a longer delay.
        let v0 = idx(int_delay);
        let v1 = idx(int_delay + 1);
        let v2 = idx(int_delay + 2);
        // At delay 0 there is no newer sample yet; extrapolate linearly.
        let v_m1 = if int_delay == 0 { 2.0 * v0 - v1 } else { idx(int_delay - 1) };

        // Catmull-Rom tangents
        let m0 = (v1 - v_m1) * 0.5;
        let m1 = (v2 - v0) * 0.5;

        let t = frac;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * v0 + h10 * m0 + h01 * v1 + h11 * m1
    }

    /// Process a full channel: read input, apply delay, write output.
    ///
    /// `input_len` and `output_len` must match.
    pub fn process_channel(&mut self, input: &[f32], output: &mut [f32], delay_samples: f32) {
        debug_assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            self.push(input[i]);
            output[i] = self.tap(delay_samples);
        }
    }

    /// Clear the delay line (fill with zeros).
    pub fn clear(&mut self) {
        for v in self.buffer.iter_mut() {
            *v = 0.0;
        }
        self.write_pos = 0;
    }

    /// Set the sample rate (recalculates buffer size).
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Number of samples currently in the buffer (write position).
    pub fn write_pos(&self) -> usize {
        self.write_pos
    }
}
