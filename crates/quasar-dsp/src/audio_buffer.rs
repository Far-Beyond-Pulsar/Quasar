use std::fmt;

/// Maximum number of audio channels supported.
pub const MAX_AUDIO_CHANNELS: usize = 32;
/// Default max samples per channel (256 samples @ 48kHz ≈ 5.3ms).
pub const DEFAULT_BLOCK_SIZE: usize = 256;

/// Fixed-capacity, deinterleaved audio buffer.
///
/// All memory is stack-allocated (no heap). NEVER allocates on the hot path.
/// Channels are stored in deinterleaved format: `data[channel][sample]`.
#[derive(Clone)]
pub struct AudioBuffer {
    data: [[f32; DEFAULT_BLOCK_SIZE]; MAX_AUDIO_CHANNELS],
    num_channels: u16,
    num_samples: u16,
}

impl AudioBuffer {
    /// Create a new zeroed buffer.
    pub fn new(num_channels: u16, num_samples: u16) -> Self {
        Self {
            data: [[0.0; DEFAULT_BLOCK_SIZE]; MAX_AUDIO_CHANNELS],
            num_channels,
            num_samples,
        }
    }

    /// Create a buffer from a slice of channel data (for testing).
    ///
    /// # Panics
    ///
    /// Panics if there are more than `MAX_AUDIO_CHANNELS` channels or if any channel
    /// exceeds `DEFAULT_BLOCK_SIZE` samples.
    pub fn from_channels(channels: &[&[f32]]) -> Self {
        let num_channels = channels.len() as u16;
        let num_samples = channels.first().map_or(0, |c| c.len()) as u16;
        let mut buf = Self::new(num_channels, num_samples);
        for (ch_idx, ch_data) in channels.iter().enumerate() {
            let len = ch_data.len().min(DEFAULT_BLOCK_SIZE);
            let dst = &mut buf.data[ch_idx][..len];
            dst.copy_from_slice(&ch_data[..len]);
        }
        buf
    }

    /// Number of channels.
    pub fn channels(&self) -> u16 {
        self.num_channels
    }

    /// Number of samples per channel.
    pub fn samples(&self) -> u16 {
        self.num_samples
    }

    /// Get a reference to a channel's sample data.
    pub fn channel(&self, channel: u16) -> &[f32] {
        let idx = channel as usize;
        debug_assert!(idx < MAX_AUDIO_CHANNELS);
        &self.data[idx][..self.num_samples as usize]
    }

    /// Get a mutable reference to a channel's sample data.
    pub fn channel_mut(&mut self, channel: u16) -> &mut [f32] {
        let idx = channel as usize;
        debug_assert!(idx < MAX_AUDIO_CHANNELS);
        &mut self.data[idx][..self.num_samples as usize]
    }

    /// Get sample at (channel, sample).
    pub fn get(&self, channel: u16, sample: u16) -> f32 {
        let ch = channel as usize;
        let s = sample as usize;
        debug_assert!(ch < MAX_AUDIO_CHANNELS && s < DEFAULT_BLOCK_SIZE);
        self.data[ch][s]
    }

    /// Set sample at (channel, sample).
    pub fn set(&mut self, channel: u16, sample: u16, value: f32) {
        let ch = channel as usize;
        let s = sample as usize;
        debug_assert!(ch < MAX_AUDIO_CHANNELS && s < DEFAULT_BLOCK_SIZE);
        self.data[ch][s] = value;
    }

    /// Copy all samples from another buffer (must match dimensions).
    pub fn copy_from(&mut self, other: &AudioBuffer) {
        debug_assert_eq!(self.num_channels, other.num_channels);
        debug_assert_eq!(self.num_samples, other.num_samples);
        let n = self.num_channels as usize;
        let s = self.num_samples as usize;
        for ch in 0..n {
            self.data[ch][..s].copy_from_slice(&other.data[ch][..s]);
        }
    }

    /// Zero all samples.
    pub fn clear(&mut self) {
        let n = self.num_channels as usize;
        let s = self.num_samples as usize;
        for ch in 0..n {
            self.data[ch][..s].fill(0.0);
        }
    }

    /// Add samples from another buffer (in-place mixing).
    pub fn add_from(&mut self, other: &AudioBuffer) {
        debug_assert_eq!(self.num_channels, other.num_channels);
        debug_assert_eq!(self.num_samples, other.num_samples);
        let n = self.num_channels as usize;
        let s = self.num_samples as usize;
        for ch in 0..n {
            for smp in 0..s {
                self.data[ch][smp] += other.data[ch][smp];
            }
        }
    }

    /// Multiply every sample by a scalar gain.
    pub fn apply_gain(&mut self, gain: f32) {
        let n = self.num_channels as usize;
        let s = self.num_samples as usize;
        for ch in 0..n {
            for smp in 0..s {
                self.data[ch][smp] *= gain;
            }
        }
    }

    /// Return the RMS level across all channels (for metering).
    pub fn rms(&self) -> f32 {
        let n = self.num_channels as usize;
        let s = self.num_samples as usize;
        let mut sum_sq = 0.0_f64;
        let mut count = 0_u64;
        for ch in 0..n {
            for smp in 0..s {
                let v = self.data[ch][smp] as f64;
                sum_sq += v * v;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        (sum_sq / count as f64).sqrt() as f32
    }

    /// Return the peak level across all channels (for metering/limiting).
    pub fn peak(&self) -> f32 {
        let n = self.num_channels as usize;
        let s = self.num_samples as usize;
        let mut peak_val = 0.0_f32;
        for ch in 0..n {
            for smp in 0..s {
                let v = self.data[ch][smp].abs();
                if v > peak_val {
                    peak_val = v;
                }
            }
        }
        peak_val
    }
}

impl fmt::Debug for AudioBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioBuffer")
            .field("num_channels", &self.num_channels)
            .field("num_samples", &self.num_samples)
            .field("rms", &self.rms())
            .field("peak", &self.peak())
            .finish()
    }
}
