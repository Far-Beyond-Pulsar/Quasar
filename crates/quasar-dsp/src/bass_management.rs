//! Bass management (#85): a Linkwitz-Riley 4th-order crossover that moves the low band of the
//! "small" speakers to the LFE / sub channel.
//!
//! # Signal flow (per block, on the speaker feeds of one listener)
//!
//! ```text
//! small speaker x --LR4 HP (fc)--> speaker feed          (large speakers: untouched, bit exact)
//!                 \-LR4 LP (fc)--> sum over small speakers --(sub gain)--> LFE channel(s) (added)
//! ```
//!
//! * **Crossover.** Each branch is two cascaded RBJ biquads with Q = 1/sqrt(2) (2nd-order
//!   Butterworth squared = LR4, -6 dB at `fc`). The analogue prototypes satisfy
//!   `LP4 + HP4 = all-pass`, and the bilinear transform preserves that, so the COHERENT sum of
//!   the speaker feed and the sub feed of one source is flat in magnitude (it only has the 360
//!   degree all-pass phase rotation around `fc`). The POWER sum `|LP|^2 + |HP|^2` is NOT flat:
//!   it dips by 3 dB at `fc` (both branches are at -6 dB there); that is inherent to LR4 and is
//!   the reason an LR crossover relies on the two drivers adding in phase in the room.
//! * **Level compensation.** The default is `0 dB`: the redirected bass has the level it had in
//!   the speaker (one source on one small speaker: speaker + sub sum to unity). Several small
//!   speakers carry the same bass for a centred image; their lows are SUMMED into the sub (no
//!   division), which is what an in-phase acoustic sum of the removed contributions is. Set
//!   `sub_gain_db` to trim for the sub's sensitivity. The "+10 dB LFE" convention (Dolby / ITU
//!   practice) applies to the DISCRETE LFE track in the playback chain and is not applied to
//!   redirected bass. With several LFE slots the sub signal is split equally between them.
//! * **LFE low-pass.** The redirected signal is already band-limited above by the LR4 low-pass.
//!   An additional LFE low-pass (the 120 Hz, 4th-order Butterworth convention of the engine's
//!   discrete LFE path) can be requested with `lfe_lowpass_hz`; it is off by default because it
//!   breaks the exact flatness when it is within an octave of `fc`.
//! * **Switching.** The stage has a mix `amount` in `[0, 1]` that ramps linearly (default 50 ms)
//!   when it is enabled or disabled: `feed = x + amount * (HP(x) - x)`, `sub += amount * LP(x)`.
//!   While `amount == 0` and the target is 0 the stage is skipped entirely (no state updates).
//! * **Real time.** All memory is allocated in [`BassManager::new`]; `process` never allocates,
//!   locks or panics. Large speakers and LFE slots are never filtered.

use crate::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use crate::biquad::BiquadFilter;

/// Default crossover frequency (Hz): the THX / common home-theatre value.
pub const DEFAULT_CROSSOVER_HZ: f32 = 80.0;
/// Allowed crossover range (Hz).
pub const CROSSOVER_RANGE_HZ: (f32, f32) = (40.0, 250.0);
/// Default time (ms) of an enable / disable ramp.
pub const DEFAULT_BASS_RAMP_MS: f32 = 50.0;

const Q_BUTTERWORTH: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Why a bass-management configuration was rejected.
#[derive(Clone, Debug, PartialEq)]
pub enum BassError {
    /// The layout has no LFE / sub channel to receive the bass: it would be lost.
    NoLfeChannel,
    /// `small.len()` differs from the layout's channel count (given, expected).
    BadSpeakerFlags { given: usize, expected: usize },
    /// Crossover frequency outside [`CROSSOVER_RANGE_HZ`] or not finite.
    BadCrossover(f32),
    /// Sub gain or LFE low-pass out of range / not finite.
    BadParameter,
    /// Channel count is zero or above [`MAX_AUDIO_CHANNELS`].
    BadChannelCount(usize),
}

impl std::fmt::Display for BassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BassError::NoLfeChannel => write!(f, "the layout has no LFE channel to receive the redirected bass"),
            BassError::BadSpeakerFlags { given, expected } => write!(f, "{given} speaker flags given, layout has {expected} channels"),
            BassError::BadCrossover(hz) => write!(f, "crossover {hz} Hz outside {:?}", CROSSOVER_RANGE_HZ),
            BassError::BadParameter => write!(f, "invalid bass-management parameter"),
            BassError::BadChannelCount(n) => write!(f, "invalid channel count {n}"),
        }
    }
}

impl std::error::Error for BassError {}

/// Bass-management settings. `small[i]` marks output channel `i` as a small speaker (its bass is
/// redirected); LFE channels are never redirected whatever their flag says.
#[derive(Clone, Debug, PartialEq)]
pub struct BassManagementConfig {
    pub crossover_hz: f32,
    pub small: Vec<bool>,
    /// Trim of the redirected bass at the sub (dB, `-24 ..= +12`). 0 = level neutral.
    pub sub_gain_db: f32,
    /// Optional extra 4th-order Butterworth low-pass on the sub feed (Hz, `40 ..= 250`).
    pub lfe_lowpass_hz: Option<f32>,
    /// Enable / disable ramp time in ms.
    pub ramp_ms: f32,
}

impl BassManagementConfig {
    /// Every non-LFE speaker of an `n`-channel layout is small.
    pub fn all_small(channels: usize, lfe_slots: &[usize]) -> Self {
        Self {
            crossover_hz: DEFAULT_CROSSOVER_HZ,
            small: (0..channels).map(|c| !lfe_slots.contains(&c)).collect(),
            sub_gain_db: 0.0,
            lfe_lowpass_hz: None,
            ramp_ms: DEFAULT_BASS_RAMP_MS,
        }
    }
}

/// One channel's LR4 split.
#[derive(Clone, Debug)]
struct Lr4 {
    lp: [BiquadFilter; 2],
    hp: [BiquadFilter; 2],
}

impl Lr4 {
    fn new(fc: f32, sr: f32) -> Self {
        let mut s = Self { lp: [BiquadFilter::new(), BiquadFilter::new()], hp: [BiquadFilter::new(), BiquadFilter::new()] };
        s.set_crossover(fc, sr);
        s
    }

    fn set_crossover(&mut self, fc: f32, sr: f32) {
        for f in self.lp.iter_mut() {
            f.set_lowpass_q_f64(fc, Q_BUTTERWORTH, sr);
        }
        for f in self.hp.iter_mut() {
            f.set_highpass_q(fc, Q_BUTTERWORTH, sr);
        }
    }

    fn reset(&mut self) {
        for f in self.lp.iter_mut().chain(self.hp.iter_mut()) {
            f.reset();
        }
    }

    #[inline]
    fn split(&mut self, x: f32) -> (f32, f32) {
        let l0 = self.lp[0].process(x);
        let lo = self.lp[1].process(l0);
        let h0 = self.hp[0].process(x);
        let hi = self.hp[1].process(h0);
        (lo, hi)
    }
}

/// The bass-management stage of one listener. See the module docs.
pub struct BassManager {
    channels: usize,
    small: [bool; MAX_AUDIO_CHANNELS],
    lfe: [usize; MAX_AUDIO_CHANNELS],
    n_lfe: usize,
    filters: Vec<Lr4>,
    lfe_lp: [BiquadFilter; 2],
    lfe_lp_on: bool,
    sub_gain: f32,
    crossover_hz: f32,
    sample_rate: f32,
    amount: f32,
    target: f32,
    ramp_samples: f32,
    sub: Vec<f32>,
}

impl BassManager {
    /// Build the stage for a layout of `channels` channels with the given LFE slots, starting
    /// DISABLED (`amount = 0`); call [`set_enabled`](Self::set_enabled) to ramp it in. Validates the
    /// configuration. API thread (allocates).
    pub fn new(sample_rate: f32, channels: usize, lfe_slots: &[usize], cfg: &BassManagementConfig) -> Result<Self, BassError> {
        if channels == 0 || channels > MAX_AUDIO_CHANNELS {
            return Err(BassError::BadChannelCount(channels));
        }
        let lfe_slots: Vec<usize> = lfe_slots.iter().copied().filter(|&s| s < channels).collect();
        if lfe_slots.is_empty() {
            return Err(BassError::NoLfeChannel);
        }
        if cfg.small.len() != channels {
            return Err(BassError::BadSpeakerFlags { given: cfg.small.len(), expected: channels });
        }
        if !cfg.crossover_hz.is_finite() || cfg.crossover_hz < CROSSOVER_RANGE_HZ.0 || cfg.crossover_hz > CROSSOVER_RANGE_HZ.1 {
            return Err(BassError::BadCrossover(cfg.crossover_hz));
        }
        if !cfg.sub_gain_db.is_finite() || cfg.sub_gain_db < -24.0 || cfg.sub_gain_db > 12.0 || !cfg.ramp_ms.is_finite() {
            return Err(BassError::BadParameter);
        }
        if let Some(hz) = cfg.lfe_lowpass_hz {
            if !hz.is_finite() || hz < CROSSOVER_RANGE_HZ.0 || hz > CROSSOVER_RANGE_HZ.1 {
                return Err(BassError::BadParameter);
            }
        }
        let mut small = [false; MAX_AUDIO_CHANNELS];
        for (i, &s) in cfg.small.iter().enumerate() {
            small[i] = s && !lfe_slots.contains(&i);
        }
        let mut lfe = [0usize; MAX_AUDIO_CHANNELS];
        for (d, &s) in lfe.iter_mut().zip(&lfe_slots) {
            *d = s;
        }
        let (mut a, mut b) = (BiquadFilter::new(), BiquadFilter::new());
        let lfe_lp_on = cfg.lfe_lowpass_hz.is_some();
        if let Some(hz) = cfg.lfe_lowpass_hz {
            a.set_lowpass_q_f64(hz, 0.5412, sample_rate);
            b.set_lowpass_q_f64(hz, 1.3066, sample_rate);
        }
        Ok(Self {
            channels,
            small,
            lfe,
            n_lfe: lfe_slots.len(),
            filters: (0..channels).map(|_| Lr4::new(cfg.crossover_hz, sample_rate)).collect(),
            lfe_lp: [a, b],
            lfe_lp_on,
            sub_gain: 10.0_f32.powf(cfg.sub_gain_db / 20.0) / lfe_slots.len() as f32,
            crossover_hz: cfg.crossover_hz,
            sample_rate,
            amount: 0.0,
            target: 0.0,
            ramp_samples: (cfg.ramp_ms.max(0.0) * 0.001 * sample_rate).max(1.0),
            sub: vec![0.0; DEFAULT_BLOCK_SIZE],
        })
    }

    /// Crossover frequency in Hz.
    pub fn crossover_hz(&self) -> f32 {
        self.crossover_hz
    }

    /// Current mix amount (0 = off, 1 = fully on).
    pub fn amount(&self) -> f32 {
        self.amount
    }

    /// Ramp the stage in (`true`) or out (`false`). No allocation.
    pub fn set_enabled(&mut self, on: bool) {
        self.target = if on { 1.0 } else { 0.0 };
    }

    /// True when the stage does nothing (fully off, not ramping).
    pub fn is_idle(&self) -> bool {
        self.amount == 0.0 && self.target == 0.0
    }

    /// Carry the running state (filter memory, amount) of `old` over when this stage replaces it
    /// with the same channel count, so a reconfiguration does not restart the filters.
    pub fn adopt_state(&mut self, old: &BassManager) {
        if old.channels == self.channels {
            for (n, o) in self.filters.iter_mut().zip(&old.filters) {
                n.lp = o.lp.clone();
                n.hp = o.hp.clone();
                n.set_crossover(self.crossover_hz, self.sample_rate);
            }
            self.amount = old.amount;
        }
    }

    /// Process one block of speaker feeds in place. Never allocates, locks or panics.
    pub fn process(&mut self, buf: &mut AudioBuffer) {
        let n = (buf.samples() as usize).min(DEFAULT_BLOCK_SIZE);
        if n == 0 || self.is_idle() {
            return;
        }
        if self.amount == 0.0 {
            // Starting from off: the filters hold stale state, begin from silence.
            for f in self.filters.iter_mut() {
                f.reset();
            }
            for f in self.lfe_lp.iter_mut() {
                f.reset();
            }
        }
        let ch = self.channels.min(buf.channels() as usize);
        let step = if self.amount < self.target {
            1.0 / self.ramp_samples
        } else if self.amount > self.target {
            -1.0 / self.ramp_samples
        } else {
            0.0
        };
        let a0 = self.amount;
        let target = self.target;
        let amount_at = |i: usize| -> f32 {
            let a = a0 + step * (i + 1) as f32;
            if step > 0.0 { a.min(target) } else if step < 0.0 { a.max(target) } else { a0 }
        };
        let sub = &mut self.sub[..n];
        sub.fill(0.0);
        for c in 0..ch {
            if !self.small[c] {
                continue;
            }
            let x = buf.channel_mut(c as u16);
            let f = &mut self.filters[c];
            for i in 0..n {
                let a = amount_at(i);
                let xi = x[i];
                let (lo, hi) = f.split(xi);
                x[i] = xi + a * (hi - xi);
                sub[i] += a * lo;
            }
        }
        if self.lfe_lp_on {
            let [f1, f2] = &mut self.lfe_lp;
            for s in sub.iter_mut() {
                *s = f2.process(f1.process(*s));
            }
        }
        for k in 0..self.n_lfe {
            let slot = self.lfe[k];
            if slot < buf.channels() as usize {
                let dst = buf.channel_mut(slot as u16);
                for i in 0..n {
                    dst[i] += sub[i] * self.sub_gain;
                }
            }
        }
        self.amount = amount_at(n - 1);
        if self.amount == self.target && self.target == 0.0 {
            self.amount = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn mgr(channels: usize, lfe: &[usize], fc: f32) -> BassManager {
        let mut cfg = BassManagementConfig::all_small(channels, lfe);
        cfg.crossover_hz = fc;
        cfg.ramp_ms = 1.0;
        let mut m = BassManager::new(SR, channels, lfe, &cfg).unwrap();
        m.set_enabled(true);
        m
    }

    /// Run a sine of `hz` on channel 0 of a 6-channel buffer (LFE = 3) long enough to settle and
    /// return the RMS of (speaker, lfe, speaker + lfe) over the last second.
    fn sine_rms(m: &mut BassManager, hz: f32) -> (f32, f32, f32) {
        let mut phase = 0.0f64;
        let (mut s0, mut s1, mut s2, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for blk in 0..(2 * 48_000 / DEFAULT_BLOCK_SIZE) {
            let mut b = AudioBuffer::new(6, DEFAULT_BLOCK_SIZE as u16);
            for i in 0..DEFAULT_BLOCK_SIZE {
                b.set(0, i as u16, phase.sin() as f32);
                phase += 2.0 * std::f64::consts::PI * hz as f64 / SR as f64;
            }
            m.process(&mut b);
            if blk >= (48_000 / DEFAULT_BLOCK_SIZE) {
                for i in 0..DEFAULT_BLOCK_SIZE {
                    let (a, l) = (b.channel(0)[i] as f64, b.channel(3)[i] as f64);
                    s0 += a * a;
                    s1 += l * l;
                    s2 += (a + l) * (a + l);
                    cnt += 1;
                }
            }
        }
        let r = |s: f64| (s / cnt as f64).sqrt() as f32 * std::f32::consts::SQRT_2;
        (r(s0), r(s1), r(s2))
    }

    #[test]
    fn coherent_sum_is_flat_through_the_crossover() {
        let mut worst = 0.0f32;
        for &hz in &[20.0, 40.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 160.0, 250.0, 1000.0, 5000.0, 15000.0] {
            let mut m = mgr(6, &[3], 80.0);
            let (_, _, sum) = sine_rms(&mut m, hz);
            let db = 20.0 * sum.log10();
            worst = worst.max(db.abs());
            assert!(db.abs() < 0.1, "{hz} Hz: coherent sum {db:.3} dB");
        }
        eprintln!("worst coherent-sum deviation {worst:.4} dB");
    }

    #[test]
    fn each_branch_is_minus_6_db_at_the_crossover_and_power_sum_dips_3_db() {
        let mut m = mgr(6, &[3], 80.0);
        let (hi, lo, _) = sine_rms(&mut m, 80.0);
        let (hi_db, lo_db) = (20.0 * hi.log10(), 20.0 * lo.log10());
        eprintln!("at fc: speaker {hi_db:.3} dB, sub {lo_db:.3} dB, power sum {:.3} dB", 10.0 * (hi * hi + lo * lo).log10());
        assert!((hi_db + 6.02).abs() < 0.1 && (lo_db + 6.02).abs() < 0.1);
    }

    #[test]
    fn lfe_gets_only_the_low_band_and_the_speaker_loses_it() {
        let mut m = mgr(6, &[3], 80.0);
        let (spk, lfe, _) = sine_rms(&mut m, 1000.0);
        assert!(20.0 * lfe.log10() < -60.0, "1 kHz leaks into the LFE at {} dB", 20.0 * lfe.log10());
        assert!((20.0 * spk.log10()).abs() < 0.05);
        let mut m = mgr(6, &[3], 80.0);
        let (spk, lfe, _) = sine_rms(&mut m, 20.0);
        assert!(20.0 * spk.log10() < -30.0, "20 Hz stays in the small speaker at {} dB", 20.0 * spk.log10());
        assert!((20.0 * lfe.log10()).abs() < 0.3, "20 Hz in the LFE: {} dB", 20.0 * lfe.log10());
    }

    #[test]
    fn large_speakers_and_idle_stage_are_bit_exact() {
        let mut cfg = BassManagementConfig::all_small(6, &[3]);
        cfg.small[1] = false; // FR is large
        cfg.ramp_ms = 1.0;
        let mut m = BassManager::new(SR, 6, &[3], &cfg).unwrap();
        let mk = || {
            let mut b = AudioBuffer::new(6, 128);
            for c in 0..6u16 {
                for i in 0..128u16 {
                    b.set(c, i, ((i as f32 + 1.0) * 0.37 + c as f32).sin() * 0.5);
                }
            }
            b
        };
        // Disabled: nothing changes at all.
        let (mut a, r) = (mk(), mk());
        m.process(&mut a);
        for c in 0..6u16 {
            assert_eq!(a.channel(c), r.channel(c));
        }
        m.set_enabled(true);
        for _ in 0..8 {
            let (mut a, r) = (mk(), mk());
            m.process(&mut a);
            assert_eq!(a.channel(1), r.channel(1), "large speaker must be untouched");
            assert_eq!(a.channel(2).iter().zip(r.channel(2)).filter(|(x, y)| x != y).count() > 0, true);
        }
    }

    #[test]
    fn bad_configurations_are_errors() {
        let ok = BassManagementConfig::all_small(6, &[3]);
        assert_eq!(BassManager::new(SR, 2, &[], &BassManagementConfig::all_small(2, &[])).err(), Some(BassError::NoLfeChannel));
        let mut c = ok.clone();
        c.small.pop();
        assert!(matches!(BassManager::new(SR, 6, &[3], &c), Err(BassError::BadSpeakerFlags { .. })));
        let mut c = ok.clone();
        c.crossover_hz = 10.0;
        assert!(matches!(BassManager::new(SR, 6, &[3], &c), Err(BassError::BadCrossover(_))));
        let mut c = ok;
        c.sub_gain_db = f32::NAN;
        assert!(matches!(BassManager::new(SR, 6, &[3], &c), Err(BassError::BadParameter)));
    }

    #[test]
    fn sub_gain_and_lfe_lowpass_apply() {
        let mut cfg = BassManagementConfig::all_small(6, &[3]);
        cfg.sub_gain_db = -6.0;
        cfg.ramp_ms = 1.0;
        let mut m = BassManager::new(SR, 6, &[3], &cfg).unwrap();
        m.set_enabled(true);
        let (_, lfe, _) = sine_rms(&mut m, 20.0);
        assert!((20.0 * lfe.log10() + 6.0).abs() < 0.3, "{}", 20.0 * lfe.log10());
        let mut cfg = BassManagementConfig::all_small(6, &[3]);
        cfg.crossover_hz = 250.0;
        cfg.lfe_lowpass_hz = Some(120.0);
        cfg.ramp_ms = 1.0;
        let mut m = BassManager::new(SR, 6, &[3], &cfg).unwrap();
        m.set_enabled(true);
        let (_, lfe, _) = sine_rms(&mut m, 240.0);
        assert!(20.0 * lfe.log10() < -20.0, "LFE low-pass at 120 Hz: 240 Hz at {} dB", 20.0 * lfe.log10());
    }

    #[test]
    fn enabling_and_disabling_ramps_without_a_step() {
        // 40 Hz + 1 kHz test signal; enable then disable; the sample-to-sample jump of the speaker
        // feed must stay close to that of the plain signal.
        let mut cfg = BassManagementConfig::all_small(6, &[3]);
        cfg.ramp_ms = 50.0;
        let mut m = BassManager::new(SR, 6, &[3], &cfg).unwrap();
        let (mut p1, mut p2, mut last) = (0.0f64, 0.0f64, 0.0f32);
        let (mut max_jump, mut max_plain) = (0.0f32, 0.0f32);
        for blk in 0..400 {
            if blk == 50 {
                m.set_enabled(true);
            }
            if blk == 250 {
                m.set_enabled(false);
            }
            let mut b = AudioBuffer::new(6, DEFAULT_BLOCK_SIZE as u16);
            let mut plain_prev = 0.0f32;
            for i in 0..DEFAULT_BLOCK_SIZE {
                let v = (0.4 * p1.sin() + 0.2 * p2.sin()) as f32;
                b.set(0, i as u16, v);
                p1 += 2.0 * std::f64::consts::PI * 40.0 / SR as f64;
                p2 += 2.0 * std::f64::consts::PI * 1000.0 / SR as f64;
                if i > 0 {
                    max_plain = max_plain.max((v - plain_prev).abs());
                }
                plain_prev = v;
            }
            m.process(&mut b);
            for i in 0..DEFAULT_BLOCK_SIZE {
                let v = b.channel(0)[i];
                if blk > 1 {
                    max_jump = max_jump.max((v - last).abs());
                }
                last = v;
            }
        }
        eprintln!("max jump {max_jump:.4} vs plain signal {max_plain:.4}");
        assert!(max_jump < 1.5 * max_plain, "click: {max_jump} vs {max_plain}");
        assert!(m.is_idle());
    }
}
