/// Extra ring samples beyond `max_samples`, so a whole block (<= `BLOCK_HEADROOM` samples) can be
/// pushed before its taps are read (see [`HermiteInterpolatingDelayLine::tap_back`]).
pub const BLOCK_HEADROOM: usize = 512;
/// Longest block [`HermiteInterpolatingDelayLine::tap_block_const`] reads.
pub const CONST_BLOCK: usize = 256;

/// A variable fractional delay line using 4-point Hermite interpolation.
///
/// Formula: `p(t) = (2t³ - 3t² + 1)·v₀ + (t³ - 2t² + t)·m₀ + (-2t³ + 3t²)·v₁ + (t³ - t²)·m₁`
/// where `m₀ = (v₁ - v₋₁)/2`, `m₁ = (v₂ - v₀)/2` (Catmull-Rom tangents).
///
/// The ring is a power of two long (at least `max_samples + BLOCK_HEADROOM`), so every index wraps
/// with a mask instead of a division and one tap costs one masked base index plus four loads.
///
/// All memory allocated at construction. NEVER allocates during processing.
pub struct HermiteInterpolatingDelayLine {
    buffer: Vec<f32>,
    mask: usize,
    /// Logical capacity (the delay clamp is derived from it).
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
        let len = (max_samples + BLOCK_HEADROOM).next_power_of_two();
        Self {
            buffer: vec![0.0; len],
            mask: len - 1,
            max_samples,
            write_pos: 0,
            sample_rate,
        }
    }

    /// Create a delay line holding `max_samples` samples (largest tap delay `max_samples - 3`)
    /// that can read a block of up to `block` samples pushed ahead of the read (see
    /// [`Self::tap_back`]; `block` is capped at [`BLOCK_HEADROOM`]).
    pub fn with_capacity(max_samples: usize, block: usize, sample_rate: f32) -> Self {
        let max_samples = max_samples.max(8);
        let len = (max_samples + block.min(BLOCK_HEADROOM)).next_power_of_two();
        Self { buffer: vec![0.0; len], mask: len - 1, max_samples, write_pos: 0, sample_rate }
    }

    /// Maximum delay in samples.
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    /// Write a sample into the delay line.
    #[inline]
    pub fn push(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) & self.mask;
    }

    /// Write a block of samples (same as calling [`Self::push`] for each one).
    pub fn push_slice(&mut self, samples: &[f32]) {
        let len = self.buffer.len();
        let mut rest = samples;
        while !rest.is_empty() {
            let n = rest.len().min(len - self.write_pos);
            self.buffer[self.write_pos..self.write_pos + n].copy_from_slice(&rest[..n]);
            self.write_pos = (self.write_pos + n) & self.mask;
            rest = &rest[n..];
        }
    }

    /// Read a sample at the given delay (fractional).
    ///
    /// `delay_samples`: fractional sample delay in `[0, max_samples - 3]`
    /// (clamped, also in release builds; NaN reads as 0). Delay 0 is the most
    /// recently pushed sample, delay `n` the sample pushed `n` pushes earlier.
    /// Uses 4-point Hermite (Catmull-Rom) interpolation.
    #[inline]
    pub fn tap(&self, delay_samples: f32) -> f32 {
        self.tap_back(delay_samples, 0)
    }

    /// The four neighbours and the fraction of a tap: `(v_m1, v0, v1, v2, frac)` for `delay_samples`
    /// counted from `newer` pushes before the newest sample (see [`Self::tap_back`]).
    #[inline(always)]
    fn gather(&self, delay_samples: f32, newer: usize) -> ([f32; 4], f32) {
        let max_delay = (self.max_samples - 3) as f32;
        // `max`/`min` also map NaN to a finite value: the delay is finite, in `[0, max_samples - 3]`.
        let delay_samples = delay_samples.max(0.0).min(max_delay);
        // SAFETY: `delay_samples` is in `[0, max_samples - 3]`, far inside the `i32` range.
        let int_delay = unsafe { delay_samples.to_int_unchecked::<i32>() } as usize;
        let frac = delay_samples - int_delay as f32;

        // The sample `int_delay + newer` pushes older than the newest is at `base`; v1 / v2 are
        // older (lower index), v_m1 newer.
        let m = self.mask;
        let base = self.write_pos.wrapping_sub(1 + int_delay + newer) & m;
        let at = |i: usize| -> f32 {
            let i = i & m;
            debug_assert!(i < self.buffer.len());
            // SAFETY: `buffer.len() == mask + 1` (power of two, set in the constructors only), so
            // `i & mask` is always in bounds.
            unsafe { *self.buffer.get_unchecked(i) }
        };
        let v0 = at(base);
        let v1 = at(base.wrapping_sub(1));
        let v2 = at(base.wrapping_sub(2));
        // At delay < 1 there is no newer sample (as when the tap is read right after the push):
        // extrapolate linearly, whatever is already in the ring from a block pushed ahead.
        let v_m1 = if int_delay == 0 { 2.0 * v0 - v1 } else { at(base.wrapping_add(1)) };
        ([v_m1, v0, v1, v2], frac)
    }

    /// Like [`Self::tap`], but counted from `newer` pushes before the newest sample, as if the
    /// last `newer` pushes had not happened yet: after pushing a whole block of `n` samples,
    /// `tap_back(d, n - 1 - i)` returns exactly what `tap(d)` returned right after the push of
    /// sample `i`. This lets a block be pushed first (one `memcpy`) and read per sample afterwards.
    /// `delay_samples` is clamped to `[0, max_samples - 3]` like `tap`; `newer` must be at most
    /// [`BLOCK_HEADROOM`].
    #[inline]
    pub fn tap_back(&self, delay_samples: f32, newer: usize) -> f32 {
        let ([v_m1, v0, v1, v2], t) = self.gather(delay_samples, newer);
        // Catmull-Rom tangents
        let m0 = (v1 - v_m1) * 0.5;
        let m1 = (v2 - v0) * 0.5;

        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * v0 + h10 * m0 + h01 * v1 + h11 * m1
    }

    /// Constant-delay block read: `out[i] = tap_back(delay_samples, out.len() - 1 - i)` for every
    /// `i`, i.e. the block of `out.len()` samples just pushed, read back at one fixed delay.
    /// Bit-identical to the per-sample `tap_back` loop, but the interpolation weights are computed
    /// once and the four neighbours are read from one contiguous window (a 4-tap FIR over
    /// consecutive samples, which vectorises). At most [`CONST_BLOCK`] (256) samples are read.
    pub fn tap_block_const(&self, delay_samples: f32, out: &mut [f32]) {
        let n = out.len().min(CONST_BLOCK);
        if n == 0 {
            return;
        }
        let max_delay = (self.max_samples - 3) as f32;
        let d = delay_samples.max(0.0).min(max_delay);
        // SAFETY: `d` is in `[0, max_samples - 3]`, far inside the `i32` range.
        let int_delay = unsafe { d.to_int_unchecked::<i32>() } as usize;
        let t = d - int_delay as f32;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        // Window `w[k] = buffer[start + k]`, `start` = (index of v2 of the first sample).
        let m = self.mask;
        let start = self.write_pos.wrapping_sub(1 + int_delay + (n - 1) + 2) & m;
        let mut w = [0.0_f32; CONST_BLOCK + 3];
        let len = n + 3;
        let first = len.min(self.buffer.len() - start);
        w[..first].copy_from_slice(&self.buffer[start..start + first]);
        if first < len {
            w[first..len].copy_from_slice(&self.buffer[..len - first]);
        }
        // Delay < 1: no newer neighbour, extrapolate (see `gather`).
        let extrap = int_delay == 0;
        for i in 0..n {
            let (v2, v1, v0) = (w[i], w[i + 1], w[i + 2]);
            let vm = if extrap { 2.0 * v0 - v1 } else { w[i + 3] };
            let m0 = (v1 - vm) * 0.5;
            let m1 = (v2 - v0) * 0.5;
            out[i] = h00 * v0 + h10 * m0 + h01 * v1 + h11 * m1;
        }
    }

    /// Read a whole block of taps: `out[j] = tap_back(delays[j], newer_j)` with
    /// `newer_j = newer0 - j` when `count_down` (the block was pushed before reading, `newer0 = n - 1`)
    /// or `newer_j = newer0` otherwise. `min(delays, out)` samples; arithmetic identical to
    /// [`Self::tap_back`] (bit for bit), but organised as a gather pass followed by a branch-free
    /// interpolation pass over 8 samples at a time so the compiler can vectorise the latter.
    pub fn tap_many(&self, delays: &[f32], newer0: usize, count_down: bool, out: &mut [f32]) {
        const L: usize = 8;
        let n = delays.len().min(out.len());
        let mut j = 0;
        while j < n {
            let c = (n - j).min(L);
            let (mut vm, mut v0, mut v1, mut v2, mut fr) = ([0.0_f32; L], [0.0_f32; L], [0.0_f32; L], [0.0_f32; L], [0.0_f32; L]);
            for l in 0..c {
                let newer = if count_down { newer0.wrapping_sub(j + l) } else { newer0 };
                let ([a, b, c1, d], f) = self.gather(delays[j + l], newer);
                vm[l] = a;
                v0[l] = b;
                v1[l] = c1;
                v2[l] = d;
                fr[l] = f;
            }
            let mut y = [0.0_f32; L];
            // Branch-free over the full lane width (lanes >= c compute on zeros, unused).
            for l in 0..L {
                let m0 = (v1[l] - vm[l]) * 0.5;
                let m1 = (v2[l] - v0[l]) * 0.5;
                let t = fr[l];
                let t2 = t * t;
                let t3 = t2 * t;
                y[l] = (2.0 * t3 - 3.0 * t2 + 1.0) * v0[l]
                    + (t3 - 2.0 * t2 + t) * m0
                    + (-2.0 * t3 + 3.0 * t2) * v1[l]
                    + (t3 - t2) * m1;
            }
            out[j..j + c].copy_from_slice(&y[..c]);
            j += c;
        }
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

    /// Ring write position (modulo the internal power-of-two capacity, which is at least
    /// `max_samples()`).
    pub fn write_pos(&self) -> usize {
        self.write_pos
    }
}
