//! Output format-conversion matrix (#83): downmix / remap the channels of a rendered bus.
//!
//! A [`ChannelMatrix`] computes `out[o] = sum_i M[o][i] * in[i]` with per-sample linear ramps
//! when the matrix changes (no zipper noise, no click when switching a downmix on or off), and
//! never allocates after construction. [`ChannelMatrix::from_downmix`] builds the standard
//! downmixes between the engine's named layouts and returns an ERROR for conversions it does not
//! define (upmix, custom layouts) instead of silently dropping channels.
//!
//! # Channel orders and coefficients
//!
//! Orders are the engine's (WASAPI / SMPTE) orders: stereo `L R`; quad `FL FR BL BR`; 5.1
//! `FL FR C LFE BL BR`; 7.1 `FL FR C LFE BL BR SL SR`. With `k = 1/sqrt(2)` (-3 dB):
//!
//! * **5.1 -> stereo** (ITU-R BS.775-3, 3/2 -> 2/0): `Lo = FL + k C + k BL`, `Ro = FR + k C + k BR`.
//!   The LFE is not mixed in (BS.775 leaves it out).
//! * **quad -> stereo** (the same surround coefficient): `Lo = FL + k BL`, `Ro = FR + k BR`.
//! * **7.1 -> 5.1** (BS.775 does not tabulate 3/4 -> 3/2; this is the conventional fold): front,
//!   centre and LFE pass unchanged, the side and rear surround of each side are summed into the 5.1
//!   surround at -3 dB each: `BL' = k (BL + SL)`, `BR' = k (BR + SR)`.
//! * **7.1 -> stereo** is the product of the two (7.1 -> 5.1 -> stereo).
//! * Identical layouts map through the identity.
//!
//! Power: a front or centre channel keeps unit total power (the centre is split `k`, `k`, so
//! `2 k^2 = 1`); a surround that goes to ONE side loses 3 dB by design; the matrix power
//! (`sum M^2`) and a per-input power ([`ChannelMatrix::input_power`]) are available for checks.

use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use crate::master_decoder::SpeakerLayout;

/// `1 / sqrt(2)`: the -3 dB coefficient of the BS.775 downmixes.
pub const BS775_K: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Default ramp time (ms) of a matrix change.
pub const DEFAULT_MATRIX_RAMP_MS: f32 = 10.0;

/// Why a matrix could not be built or applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// No standard conversion is defined between the two layouts (channel counts shown).
    UnsupportedConversion { from_channels: usize, to_channels: usize },
    /// A channel count is zero or above [`MAX_AUDIO_CHANNELS`].
    BadChannelCount(usize),
    /// A matrix slice does not hold `out * in` gains (given, expected).
    BadShape { given: usize, expected: usize },
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::UnsupportedConversion { from_channels, to_channels } => {
                write!(f, "no standard conversion from {from_channels} to {to_channels} channels")
            }
            MatrixError::BadChannelCount(n) => write!(f, "invalid channel count {n} (1..={MAX_AUDIO_CHANNELS})"),
            MatrixError::BadShape { given, expected } => write!(f, "matrix has {given} gains, expected {expected}"),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Number of channels of a layout.
pub fn layout_channel_count(layout: &SpeakerLayout) -> usize {
    match layout {
        SpeakerLayout::Stereo => 2,
        SpeakerLayout::Quad => 4,
        SpeakerLayout::Surround51 => 6,
        SpeakerLayout::Surround714 => 8,
        SpeakerLayout::Custom { positions } => positions.len(),
    }
}

const K: f32 = BS775_K;

/// 7.1 -> 5.1, rows `FL FR C LFE BL BR`, columns `FL FR C LFE BL BR SL SR`.
#[rustfmt::skip]
const M71_TO_51: [f32; 6 * 8] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, K,   0.0, K,   0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, K,   0.0, K,
];

/// 5.1 -> stereo (BS.775-3), rows `L R`, columns `FL FR C LFE BL BR`.
#[rustfmt::skip]
const M51_TO_20: [f32; 2 * 6] = [
    1.0, 0.0, K, 0.0, K,   0.0,
    0.0, 1.0, K, 0.0, 0.0, K,
];

/// quad -> stereo, rows `L R`, columns `FL FR BL BR`.
#[rustfmt::skip]
const MQUAD_TO_20: [f32; 2 * 4] = [
    1.0, 0.0, K,   0.0,
    0.0, 1.0, 0.0, K,
];

/// `a (m x n) * b (n x p)`, row-major.
fn matmul(a: &[f32], m: usize, n: usize, b: &[f32], p: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * p];
    for r in 0..m {
        for c in 0..p {
            out[r * p + c] = (0..n).map(|k| a[r * n + k] * b[k * p + c]).sum();
        }
    }
    out
}

/// Row-major `out x in` gains of the standard downmix `from` -> `to` (identity for identical
/// layouts). API thread only (allocates). Errors for conversions that are not defined.
pub fn downmix_gains(from: &SpeakerLayout, to: &SpeakerLayout) -> Result<Vec<f32>, MatrixError> {
    use SpeakerLayout::*;
    let (nf, nt) = (layout_channel_count(from), layout_channel_count(to));
    let ident = |n: usize| (0..n * n).map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
    match (from, to) {
        (Stereo, Stereo) | (Quad, Quad) | (Surround51, Surround51) | (Surround714, Surround714) => Ok(ident(nf)),
        (Surround714, Surround51) => Ok(M71_TO_51.to_vec()),
        (Surround51, Stereo) => Ok(M51_TO_20.to_vec()),
        (Quad, Stereo) => Ok(MQUAD_TO_20.to_vec()),
        (Surround714, Stereo) => Ok(matmul(&M51_TO_20, 2, 6, &M71_TO_51, 8)),
        _ => Err(MatrixError::UnsupportedConversion { from_channels: nf, to_channels: nt }),
    }
}

/// A channel matrix with per-sample ramped changes. See the module docs.
pub struct ChannelMatrix {
    in_ch: usize,
    out_ch: usize,
    /// Current gains, row-major `out x in` (capacity `in * out`, never reallocated).
    cur: Vec<f32>,
    /// Target gains.
    tgt: Vec<f32>,
    /// Per-sample gain increments of the running ramp.
    step: Vec<f32>,
    ramp_samples: u32,
    remaining: u32,
}

impl ChannelMatrix {
    /// Silent `in_channels` -> `out_channels` matrix (all gains 0) with a ramp of `ramp_ms` at
    /// `sample_rate`. Allocates.
    pub fn new(in_channels: usize, out_channels: usize, sample_rate: f32, ramp_ms: f32) -> Result<Self, MatrixError> {
        for n in [in_channels, out_channels] {
            if n == 0 || n > MAX_AUDIO_CHANNELS {
                return Err(MatrixError::BadChannelCount(n));
            }
        }
        let len = in_channels * out_channels;
        let ramp_samples = if sample_rate.is_finite() && ramp_ms.is_finite() {
            (ramp_ms.max(0.0) * 0.001 * sample_rate).round() as u32
        } else {
            0
        };
        Ok(Self {
            in_ch: in_channels,
            out_ch: out_channels,
            cur: vec![0.0; len],
            tgt: vec![0.0; len],
            step: vec![0.0; len],
            ramp_samples,
            remaining: 0,
        })
    }

    /// Matrix of the standard downmix `from` -> `to` ([`downmix_gains`]), starting AT the target
    /// (no ramp from silence). Allocates.
    pub fn from_downmix(from: &SpeakerLayout, to: &SpeakerLayout, sample_rate: f32, ramp_ms: f32) -> Result<Self, MatrixError> {
        let gains = downmix_gains(from, to)?;
        let mut m = Self::new(layout_channel_count(from), layout_channel_count(to), sample_rate, ramp_ms)?;
        m.set_matrix_immediate(&gains)?;
        Ok(m)
    }

    /// Input channel count.
    pub fn input_channels(&self) -> usize {
        self.in_ch
    }

    /// Output channel count.
    pub fn output_channels(&self) -> usize {
        self.out_ch
    }

    /// Target gain from input `i` to output `o` (0 if out of range).
    pub fn gain(&self, o: usize, i: usize) -> f32 {
        if o < self.out_ch && i < self.in_ch {
            self.tgt[o * self.in_ch + i]
        } else {
            0.0
        }
    }

    /// Power gain of input `i`: `sum_o M[o][i]^2` (target matrix).
    pub fn input_power(&self, i: usize) -> f32 {
        (0..self.out_ch).map(|o| self.gain(o, i).powi(2)).sum()
    }

    /// Set a new target matrix (row-major `out x in`); the gains ramp linearly per sample from
    /// their current values over the configured ramp time. No allocation.
    pub fn set_matrix(&mut self, gains: &[f32]) -> Result<(), MatrixError> {
        self.check(gains)?;
        for (t, &g) in self.tgt.iter_mut().zip(gains) {
            *t = if g.is_finite() { g } else { 0.0 };
        }
        if self.ramp_samples == 0 {
            self.cur.copy_from_slice(&self.tgt);
            self.remaining = 0;
        } else {
            let inv = 1.0 / self.ramp_samples as f32;
            for ((s, &t), &c) in self.step.iter_mut().zip(&self.tgt).zip(&self.cur) {
                *s = (t - c) * inv;
            }
            self.remaining = self.ramp_samples;
        }
        Ok(())
    }

    /// Set the matrix without a ramp (initial setup). No allocation.
    pub fn set_matrix_immediate(&mut self, gains: &[f32]) -> Result<(), MatrixError> {
        self.check(gains)?;
        for (t, &g) in self.tgt.iter_mut().zip(gains) {
            *t = if g.is_finite() { g } else { 0.0 };
        }
        self.cur.copy_from_slice(&self.tgt);
        self.step.fill(0.0);
        self.remaining = 0;
        Ok(())
    }

    /// Ramp to the standard downmix `from` -> `to` (channel counts must equal this matrix's).
    /// Allocates (builds the preset): call from the API thread and hand the result over, or use
    /// [`set_matrix`](Self::set_matrix) with a prepared slice on the audio thread.
    pub fn switch_downmix(&mut self, from: &SpeakerLayout, to: &SpeakerLayout) -> Result<(), MatrixError> {
        let gains = downmix_gains(from, to)?;
        self.set_matrix(&gains)
    }

    fn check(&self, gains: &[f32]) -> Result<(), MatrixError> {
        let expected = self.in_ch * self.out_ch;
        if gains.len() != expected {
            return Err(MatrixError::BadShape { given: gains.len(), expected });
        }
        Ok(())
    }

    /// `output = M * input` for one block (`min` of the buffers' sample counts, at most
    /// [`DEFAULT_BLOCK_SIZE`]); channels beyond this matrix's counts are left untouched / ignored.
    /// Never allocates, locks or panics.
    pub fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let n = (input.samples() as usize).min(output.samples() as usize).min(DEFAULT_BLOCK_SIZE);
        let in_ch = self.in_ch.min(input.channels() as usize);
        let out_ch = self.out_ch.min(output.channels() as usize);
        let row = self.in_ch;
        for o in 0..out_ch {
            let dst = &mut output.channel_mut(o as u16)[..n];
            dst.fill(0.0);
            for i in 0..in_ch {
                let src = &input.channel(i as u16)[..n];
                let idx = o * row + i;
                let (g0, step) = (self.cur[idx], self.step[idx]);
                let ramp = (self.remaining as usize).min(n);
                for j in 0..ramp {
                    dst[j] += src[j] * (g0 + step * (j + 1) as f32);
                }
                if ramp < n {
                    // The ramp ended inside the block (or none is running): the target.
                    let g = if self.remaining > 0 { self.tgt[idx] } else { g0 };
                    if g != 0.0 {
                        for j in ramp..n {
                            dst[j] += src[j] * g;
                        }
                    }
                }
            }
        }
        // Advance the ramp (all entries together).
        if self.remaining > 0 {
            let used = (self.remaining as usize).min(n) as u32;
            if used >= self.remaining {
                self.cur.copy_from_slice(&self.tgt);
                self.step.fill(0.0);
                self.remaining = 0;
            } else {
                for (c, &s) in self.cur.iter_mut().zip(&self.step) {
                    *c += s * used as f32;
                }
                self.remaining -= used;
            }
        }
    }
}
