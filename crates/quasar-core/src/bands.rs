/// Number of octave frequency bands used throughout the spatial audio engine.
pub const FREQ_BAND_COUNT: usize = 8;

/// Centre frequencies (Hz) of the 8 standard octave bands.
pub const FREQ_BAND_CENTRES: [f32; FREQ_BAND_COUNT] = [
    62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
];

/// Per-band acoustic values across all 8 octave bands.
///
/// Operations are per-band (component-wise).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Band8(pub [f32; 8]);

impl Band8 {
    /// Create a new `Band8` from an array of 8 values.
    pub fn new(values: [f32; 8]) -> Self {
        Self(values)
    }

    /// Create a `Band8` where all 8 bands are set to zero.
    pub fn zeros() -> Self {
        Self([0.0; 8])
    }

    /// Create a `Band8` where every band has the same value `v`.
    pub fn splat(v: f32) -> Self {
        Self([v; 8])
    }

    /// Per-band addition.
    pub fn add(&self, other: &Band8) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] + other.0[i];
        }
        Band8(out)
    }

    /// Per-band subtraction.
    pub fn sub(&self, other: &Band8) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] - other.0[i];
        }
        Band8(out)
    }

    /// Per-band multiplication.
    pub fn mul(&self, other: &Band8) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] * other.0[i];
        }
        Band8(out)
    }

    /// Multiply every band by a scalar.
    pub fn scale(&self, factor: f32) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] * factor;
        }
        Band8(out)
    }

    /// Linear interpolation toward `other` by factor `t` (per-band).
    pub fn lerp(&self, other: &Band8, t: f32) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] + t * (other.0[i] - self.0[i]);
        }
        Band8(out)
    }

    /// Apply gain in dB (per-band): result = self * 10^(gain_db/20)
    pub fn apply_gain_db(&self, gain_db: &Band8) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = self.0[i] * 10.0_f32.powf(gain_db.0[i] / 20.0);
        }
        Band8(out)
    }

    /// Convert to dB scale per band: 20 * log10(value).
    ///
    /// Values ≤ 0 clamp to -160 dB to avoid -inf.
    pub fn to_db(&self) -> Band8 {
        let mut out = [0.0; 8];
        for i in 0..8 {
            out[i] = if self.0[i] > 0.0 {
                20.0 * self.0[i].log10()
            } else {
                -160.0
            };
        }
        Band8(out)
    }

    /// Sum all bands.
    pub fn sum(&self) -> f32 {
        self.0.iter().sum()
    }

    /// Mean of all bands.
    pub fn mean(&self) -> f32 {
        self.0.iter().sum::<f32>() / 8.0
    }
}
