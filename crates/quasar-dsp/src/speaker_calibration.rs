//! Per-output-channel speaker calibration (#84): time alignment, level trim, parametric EQ and
//! an optional high-pass for every speaker feed.
//!
//! # Chain (per channel, per sample)
//!
//! ```text
//! x -> fractional delay (Hermite) -> high-pass (2nd order Butterworth, optional)
//!   -> up to 3 EQ bands (peaking / low shelf / high shelf, RBJ biquads) -> gain trim -> y
//! ```
//!
//! The delay is the first element so that a change of the delay can be done by CROSS-FADING two
//! taps of the same line (no pitch bend, which a ramped read position would cause).
//!
//! # Auto-alignment from distances
//!
//! [`CalibrationConfig::from_distances`] takes the distance of every speaker to the sweet spot
//! (metres). Sound from the farthest speaker arrives last, so every nearer speaker is delayed by
//! `(d_max - d_i) / c` seconds (fractional samples: Hermite interpolation reproduces any integer
//! delay exactly and the fractional part with a first moment equal to the requested delay, so
//! arrival times are aligned to well within one sample) and attenuated by `20 log10(d_i / d_max)`
//! dB, which equalises the free-field 1/r level at the sweet spot (`level_exponent` = 1; a
//! different exponent models a different decay law). The farthest speaker is the reference:
//! delay 0, gain 0 dB, so the stage adds no latency to it and never boosts.
//!
//! # Changes
//!
//! A new configuration is applied with a cross-fade (default 50 ms, linear weights) between the
//! previous chain (old delay, old filters, old gain) and the new one: no click, no zipper. A
//! second configuration arriving during a fade completes the first one immediately (rare).
//!
//! # Real time
//!
//! [`SpeakerCalibration::new`] allocates (delay lines); [`set_config`](SpeakerCalibration::set_config)
//! and [`process`](SpeakerCalibration::process) never allocate, lock or panic.
//!
//! The engine runs it on the listener's physical speaker feeds after bass management and before
//! the output conversion and the limiter (see `SpatialAudioEngine::set_listener_calibration`).

use crate::audio_buffer::{AudioBuffer, MAX_AUDIO_CHANNELS};
use crate::biquad::BiquadFilter;
use crate::fractional_delay::HermiteInterpolatingDelayLine;

/// EQ bands per channel.
pub const MAX_EQ_BANDS: usize = 3;
/// Default cross-fade time (ms) of a configuration change.
pub const DEFAULT_CALIBRATION_RAMP_MS: f32 = 50.0;
/// Default longest delay a stage can apply (seconds): 50 ms = 17 m of path difference.
pub const DEFAULT_MAX_DELAY_SECS: f32 = 0.05;

/// One EQ band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EqBand {
    Off,
    /// Bell: `gain_db` at `hz`, bandwidth from `q`.
    Peaking { hz: f32, q: f32, gain_db: f32 },
    /// `gain_db` below `hz`.
    LowShelf { hz: f32, gain_db: f32 },
    /// `gain_db` above `hz`.
    HighShelf { hz: f32, gain_db: f32 },
}

/// Calibration of one output channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelCalibration {
    /// Delay in (fractional) samples at the stage's sample rate, `>= 0`.
    pub delay_samples: f32,
    /// Gain trim in dB (`-60 ..= +12`).
    pub gain_db: f32,
    /// Optional 2nd-order high-pass (Hz, 10 ..= 500).
    pub highpass_hz: Option<f32>,
    pub eq: [EqBand; MAX_EQ_BANDS],
}

impl Default for ChannelCalibration {
    fn default() -> Self {
        Self { delay_samples: 0.0, gain_db: 0.0, highpass_hz: None, eq: [EqBand::Off; MAX_EQ_BANDS] }
    }
}

/// Why a calibration could not be built or applied.
#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationError {
    /// Zero channels or more than [`MAX_AUDIO_CHANNELS`].
    BadChannelCount(usize),
    /// The configuration has a different number of channels than the stage (given, expected).
    ChannelMismatch { given: usize, expected: usize },
    /// A distance is not a positive finite number (index).
    BadDistance(usize),
    /// The required delay exceeds the stage's capacity (needed, max samples).
    DelayTooLong { needed: f32, max: f32 },
    /// A parameter is out of range or not finite (channel).
    BadParameter(usize),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::BadChannelCount(n) => write!(f, "invalid channel count {n}"),
            CalibrationError::ChannelMismatch { given, expected } => write!(f, "configuration has {given} channels, stage has {expected}"),
            CalibrationError::BadDistance(i) => write!(f, "distance of channel {i} is not a positive finite number"),
            CalibrationError::DelayTooLong { needed, max } => write!(f, "delay of {needed} samples exceeds the stage maximum {max}"),
            CalibrationError::BadParameter(c) => write!(f, "parameter of channel {c} out of range"),
        }
    }
}

impl std::error::Error for CalibrationError {}

/// Calibration of every channel of a layout.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationConfig {
    pub channels: Vec<ChannelCalibration>,
    /// Cross-fade time of the change in ms.
    pub ramp_ms: f32,
}

impl CalibrationConfig {
    /// A flat (transparent) configuration for `n` channels.
    pub fn flat(n: usize) -> Self {
        Self { channels: vec![ChannelCalibration::default(); n], ramp_ms: DEFAULT_CALIBRATION_RAMP_MS }
    }

    /// Auto-align from the distances (m) of the speakers to the sweet spot. `speed_of_sound` in
    /// m/s, `level_exponent` 1.0 for the free-field 1/r law. Fails when a distance is invalid or
    /// when the largest delay exceeds `max_delay_samples`. Existing EQ / high-pass are not set.
    pub fn from_distances(
        distances_m: &[f32],
        sample_rate: f32,
        speed_of_sound: f32,
        level_exponent: f32,
        max_delay_samples: f32,
    ) -> Result<Self, CalibrationError> {
        let n = distances_m.len();
        if n == 0 || n > MAX_AUDIO_CHANNELS {
            return Err(CalibrationError::BadChannelCount(n));
        }
        if !(speed_of_sound.is_finite() && speed_of_sound > 0.0 && level_exponent.is_finite() && sample_rate > 0.0) {
            return Err(CalibrationError::BadParameter(0));
        }
        for (i, &d) in distances_m.iter().enumerate() {
            if !(d.is_finite() && d > 0.0) {
                return Err(CalibrationError::BadDistance(i));
            }
        }
        let dmax = distances_m.iter().copied().fold(0.0_f32, f32::max);
        let mut cfg = Self::flat(n);
        for (c, &d) in cfg.channels.iter_mut().zip(distances_m) {
            c.delay_samples = (dmax - d) / speed_of_sound * sample_rate;
            c.gain_db = 20.0 * (d / dmax).powf(level_exponent).log10();
            if c.delay_samples > max_delay_samples {
                return Err(CalibrationError::DelayTooLong { needed: c.delay_samples, max: max_delay_samples });
            }
        }
        Ok(cfg)
    }

    fn validate(&self, sample_rate: f32, max_delay: f32) -> Result<(), CalibrationError> {
        let nyq = sample_rate * 0.5;
        if !self.ramp_ms.is_finite() {
            return Err(CalibrationError::BadParameter(0));
        }
        for (i, c) in self.channels.iter().enumerate() {
            let bad = !(c.delay_samples.is_finite() && c.delay_samples >= 0.0)
                || !(c.gain_db.is_finite() && (-60.0..=12.0).contains(&c.gain_db))
                || c.highpass_hz.map_or(false, |h| !(h.is_finite() && (10.0..=500.0).contains(&h)))
                || c.eq.iter().any(|b| match *b {
                    EqBand::Off => false,
                    EqBand::Peaking { hz, q, gain_db } => {
                        !(hz.is_finite() && hz > 10.0 && hz < nyq * 0.95 && q.is_finite() && q > 0.1 && q < 20.0 && gain_db.is_finite() && gain_db.abs() <= 24.0)
                    }
                    EqBand::LowShelf { hz, gain_db } | EqBand::HighShelf { hz, gain_db } => {
                        !(hz.is_finite() && hz > 10.0 && hz < nyq * 0.95 && gain_db.is_finite() && gain_db.abs() <= 24.0)
                    }
                });
            if bad {
                return Err(CalibrationError::BadParameter(i));
            }
            if c.delay_samples > max_delay {
                return Err(CalibrationError::DelayTooLong { needed: c.delay_samples, max: max_delay });
            }
        }
        Ok(())
    }
}

/// Resolved parameters of one chain.
#[derive(Clone, Copy)]
struct Chain {
    delay: f32,
    gain: f32,
    hp_on: bool,
    eq_on: [bool; MAX_EQ_BANDS],
    hp: BiquadFilterCopy,
    eq: [BiquadFilterCopy; MAX_EQ_BANDS],
}

/// `BiquadFilter` is `Clone` but not `Copy`; keep a small Copy wrapper around the DF1 state so
/// the chains can be copied around without allocation.
#[derive(Clone, Copy)]
struct BiquadFilterCopy {
    b: [f32; 5],
    z: [f32; 4],
}

impl BiquadFilterCopy {
    const IDENT: Self = Self { b: [1.0, 0.0, 0.0, 0.0, 0.0], z: [0.0; 4] };

    fn from_filter(f: &BiquadFilter) -> Self {
        Self { b: f.coefficients(), z: [0.0; 4] }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let [b0, b1, b2, a1, a2] = self.b;
        let y = b0 * x + b1 * self.z[0] + b2 * self.z[1] - a1 * self.z[2] - a2 * self.z[3];
        self.z[1] = self.z[0];
        self.z[0] = x;
        self.z[3] = self.z[2];
        self.z[2] = y;
        y
    }
}

impl Chain {
    const FLAT: Self = Self {
        delay: 0.0,
        gain: 1.0,
        hp_on: false,
        eq_on: [false; MAX_EQ_BANDS],
        hp: BiquadFilterCopy::IDENT,
        eq: [BiquadFilterCopy::IDENT; MAX_EQ_BANDS],
    };

    fn from_params(p: &ChannelCalibration, sr: f32) -> Self {
        let mut c = Self::FLAT;
        c.delay = p.delay_samples;
        c.gain = 10.0_f32.powf(p.gain_db / 20.0);
        if let Some(hz) = p.highpass_hz {
            let mut f = BiquadFilter::new();
            f.set_highpass_q(hz, std::f32::consts::FRAC_1_SQRT_2, sr);
            c.hp = BiquadFilterCopy::from_filter(&f);
            c.hp_on = true;
        }
        for (k, band) in p.eq.iter().enumerate() {
            let mut f = BiquadFilter::new();
            match *band {
                EqBand::Off => continue,
                EqBand::Peaking { hz, q, gain_db } => f.set_peaking(hz, q, gain_db, sr),
                EqBand::LowShelf { hz, gain_db } => f.set_low_shelf(hz, gain_db, sr),
                EqBand::HighShelf { hz, gain_db } => f.set_high_shelf(hz, gain_db, sr),
            }
            c.eq[k] = BiquadFilterCopy::from_filter(&f);
            c.eq_on[k] = true;
        }
        c
    }

    /// Filters + gain on an already delayed sample.
    #[inline]
    fn tail(&mut self, mut x: f32) -> f32 {
        if self.hp_on {
            x = self.hp.process(x);
        }
        for k in 0..MAX_EQ_BANDS {
            if self.eq_on[k] {
                x = self.eq[k].process(x);
            }
        }
        x * self.gain
    }
}

struct ChannelState {
    line: HermiteInterpolatingDelayLine,
    new: Chain,
    old: Chain,
}

/// The calibration stage of one listener. See the module docs.
pub struct SpeakerCalibration {
    sample_rate: f32,
    max_delay: f32,
    chans: Vec<ChannelState>,
    fading: bool,
    fade_pos: u32,
    fade_len: u32,
}

impl SpeakerCalibration {
    /// Stage for `channels` channels, starting FLAT (bit-exact pass-through), able to delay up
    /// to `max_delay_secs`. Allocates the delay lines.
    pub fn new(sample_rate: f32, channels: usize, max_delay_secs: f32) -> Result<Self, CalibrationError> {
        if channels == 0 || channels > MAX_AUDIO_CHANNELS {
            return Err(CalibrationError::BadChannelCount(channels));
        }
        let chans = (0..channels)
            .map(|_| ChannelState {
                line: HermiteInterpolatingDelayLine::new(max_delay_secs.max(0.001), sample_rate),
                new: Chain::FLAT,
                old: Chain::FLAT,
            })
            .collect::<Vec<_>>();
        let max_delay = chans[0].line.max_samples().saturating_sub(4) as f32;
        Ok(Self { sample_rate, max_delay, chans, fading: false, fade_pos: 0, fade_len: 1 })
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.chans.len()
    }

    /// Largest delay (samples) this stage can apply.
    pub fn max_delay_samples(&self) -> f32 {
        self.max_delay
    }

    /// True while a configuration change is cross-fading.
    pub fn is_fading(&self) -> bool {
        self.fading
    }

    /// Apply a configuration (validated; nothing changes on error). Starts a cross-fade from the
    /// chain that is running now. Never allocates.
    pub fn set_config(&mut self, cfg: &CalibrationConfig) -> Result<(), CalibrationError> {
        if cfg.channels.len() != self.chans.len() {
            return Err(CalibrationError::ChannelMismatch { given: cfg.channels.len(), expected: self.chans.len() });
        }
        cfg.validate(self.sample_rate, self.max_delay)?;
        for st in self.chans.iter_mut() {
            st.old = st.new; // (a fade in progress is completed abruptly: the old chain is dropped)
        }
        for (st, p) in self.chans.iter_mut().zip(&cfg.channels) {
            let mut n = Chain::from_params(p, self.sample_rate);
            // Filters of an unchanged band keep their memory (no restart transient).
            if st.old.hp_on && n.hp_on && st.old.hp.b == n.hp.b {
                n.hp.z = st.old.hp.z;
            }
            for k in 0..MAX_EQ_BANDS {
                if st.old.eq_on[k] && n.eq_on[k] && st.old.eq[k].b == n.eq[k].b {
                    n.eq[k].z = st.old.eq[k].z;
                }
            }
            st.new = n;
        }
        self.fade_len = ((cfg.ramp_ms.max(0.0) * 0.001 * self.sample_rate) as u32).max(1);
        self.fade_pos = 0;
        self.fading = true;
        Ok(())
    }

    /// Process one block of speaker feeds in place. Never allocates, locks or panics.
    pub fn process(&mut self, buf: &mut AudioBuffer) {
        let n = buf.samples() as usize;
        let ch = self.chans.len().min(buf.channels() as usize);
        let (fading, pos0, len) = (self.fading, self.fade_pos, self.fade_len);
        for c in 0..ch {
            let st = &mut self.chans[c];
            let x = buf.channel_mut(c as u16);
            if fading {
                for i in 0..n {
                    st.line.push(x[i]);
                    let w = ((pos0 as usize + i + 1) as f32 / len as f32).min(1.0);
                    let yn = {
                        let d = st.line.tap(st.new.delay);
                        st.new.tail(d)
                    };
                    let yo = {
                        let d = st.line.tap(st.old.delay);
                        st.old.tail(d)
                    };
                    x[i] = yo + (yn - yo) * w;
                }
            } else {
                for i in 0..n {
                    st.line.push(x[i]);
                    let d = st.line.tap(st.new.delay);
                    x[i] = st.new.tail(d);
                }
            }
        }
        if fading {
            self.fade_pos = pos0.saturating_add(n as u32);
            if self.fade_pos >= len {
                self.fading = false;
                self.fade_pos = 0;
                for st in self.chans.iter_mut() {
                    st.old = st.new;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn run(cal: &mut SpeakerCalibration, chans: usize, input: &[f32], block: usize) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::new(); chans];
        for chunk in input.chunks(block) {
            let mut b = AudioBuffer::new(chans as u16, chunk.len() as u16);
            for c in 0..chans {
                b.channel_mut(c as u16).copy_from_slice(chunk);
            }
            cal.process(&mut b);
            for c in 0..chans {
                out[c].extend_from_slice(b.channel(c as u16));
            }
        }
        out
    }

    fn settle(cal: &mut SpeakerCalibration, chans: usize) {
        run(cal, chans, &vec![0.0; 8 * 4800], 256);
    }

    fn sine_gain_db(cal: &mut SpeakerCalibration, hz: f32) -> f32 {
        let n = 2 * 48_000;
        let x: Vec<f32> = (0..n).map(|i| (2.0 * std::f64::consts::PI * hz as f64 * i as f64 / SR as f64).sin() as f32).collect();
        let y = run(cal, 1, &x, 256);
        let rms = |v: &[f32]| (v.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        (20.0 * (rms(&y[0][n / 2..]) / rms(&x[n / 2..])).log10()) as f32
    }

    #[test]
    fn flat_stage_is_bit_exact() {
        let mut cal = SpeakerCalibration::new(SR, 2, 0.05).unwrap();
        let x: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let y = run(&mut cal, 2, &x, 100);
        assert_eq!(y[0], x);
        assert_eq!(y[1], x);
    }

    #[test]
    fn auto_align_arrival_within_one_sample_and_level_within_0_1_db() {
        // 5.1-ish rig: distances in metres.
        let dist = [2.0, 2.3, 1.7, 2.6, 3.4, 3.1];
        let cfg = CalibrationConfig::from_distances(&dist, SR, 343.0, 1.0, 2000.0).unwrap();
        let mut cal = SpeakerCalibration::new(SR, dist.len(), 0.05).unwrap();
        cal.set_config(&cfg).unwrap();
        settle(&mut cal, dist.len());
        let mut x = vec![0.0f32; 4000];
        x[500] = 1.0;
        let y = run(&mut cal, dist.len(), &x, 256);
        let dmax = 3.4;
        let (mut t_min, mut t_max) = (f64::MAX, f64::MIN);
        let (mut l_min, mut l_max) = (f64::MAX, f64::MIN);
        for (c, &d) in dist.iter().enumerate() {
            // Centroid of the impulse response = arrival time in samples after the physical
            // propagation delay is added.
            let sum: f64 = y[c].iter().map(|&v| v as f64).sum();
            let centroid: f64 = y[c].iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum::<f64>() / sum - 500.0;
            let arrival = centroid + d as f64 / 343.0 * SR as f64;
            // Level at the sweet spot: DC gain of the chain times the 1/r free-field decay.
            let level = sum / d as f64;
            t_min = t_min.min(arrival);
            t_max = t_max.max(arrival);
            l_min = l_min.min(level);
            l_max = l_max.max(level);
            let _ = dmax;
        }
        let spread_db = 20.0 * (l_max / l_min).log10();
        eprintln!("arrival spread {:.4} samples, level spread {spread_db:.4} dB", t_max - t_min);
        assert!(t_max - t_min < 1.0, "arrival spread {}", t_max - t_min);
        assert!(spread_db < 0.1, "level spread {spread_db} dB");
    }

    #[test]
    fn farthest_speaker_is_the_reference() {
        let cfg = CalibrationConfig::from_distances(&[1.0, 4.0, 2.0], SR, 343.0, 1.0, 5000.0).unwrap();
        assert_eq!(cfg.channels[1].delay_samples, 0.0);
        assert_eq!(cfg.channels[1].gain_db, 0.0);
        assert!(cfg.channels[0].delay_samples > cfg.channels[2].delay_samples);
        assert!((cfg.channels[0].gain_db + 12.04).abs() < 0.01);
        assert!(CalibrationConfig::from_distances(&[1.0, 0.0], SR, 343.0, 1.0, 5000.0).is_err());
        assert!(matches!(
            CalibrationConfig::from_distances(&[1.0, 40.0], SR, 343.0, 1.0, 2000.0),
            Err(CalibrationError::DelayTooLong { .. })
        ));
    }

    #[test]
    fn eq_bands_have_the_requested_response() {
        let mut c = ChannelCalibration::default();
        c.eq[0] = EqBand::Peaking { hz: 1000.0, q: 1.0, gain_db: 6.0 };
        c.eq[1] = EqBand::LowShelf { hz: 200.0, gain_db: -4.0 };
        c.eq[2] = EqBand::HighShelf { hz: 6000.0, gain_db: 3.0 };
        let cfg = CalibrationConfig { channels: vec![c], ramp_ms: 1.0 };
        let measure = |hz: f32| {
            let mut cal = SpeakerCalibration::new(SR, 1, 0.05).unwrap();
            cal.set_config(&cfg).unwrap();
            sine_gain_db(&mut cal, hz)
        };
        let g_1k = measure(1000.0);
        let g_lo = measure(30.0);
        let g_hi = measure(18_000.0);
        eprintln!("EQ: 1 kHz {g_1k:.3} dB, 30 Hz {g_lo:.3} dB, 18 kHz {g_hi:.3} dB");
        // 1 kHz also sees a little of both shelves' skirts: within 0.5 dB of the +6 dB bell.
        assert!((g_1k - 6.0).abs() < 0.5, "{g_1k}");
        assert!((g_lo + 4.0).abs() < 0.3, "{g_lo}");
        assert!((g_hi - 3.0).abs() < 0.4, "{g_hi}");
        // High-pass: -3 dB at the corner, flat above.
        let mut c = ChannelCalibration::default();
        c.highpass_hz = Some(100.0);
        let mut cal = SpeakerCalibration::new(SR, 1, 0.05).unwrap();
        cal.set_config(&CalibrationConfig { channels: vec![c], ramp_ms: 1.0 }).unwrap();
        let at_corner = sine_gain_db(&mut cal, 100.0);
        assert!((at_corner + 3.01).abs() < 0.15, "{at_corner}");
        let mut cal = SpeakerCalibration::new(SR, 1, 0.05).unwrap();
        cal.set_config(&CalibrationConfig { channels: vec![c], ramp_ms: 1.0 }).unwrap();
        assert!(sine_gain_db(&mut cal, 2000.0).abs() < 0.05);
    }

    #[test]
    fn configuration_changes_do_not_click() {
        // 300 Hz sine; mid-stream switch to +10 ms delay, -10 dB, and a bell.
        let mut cal = SpeakerCalibration::new(SR, 1, 0.05).unwrap();
        let n = 48_000;
        let x: Vec<f32> = (0..n).map(|i| 0.5 * (2.0 * std::f64::consts::PI * 300.0 * i as f64 / SR as f64).sin() as f32).collect();
        let mut y = Vec::new();
        for (blk, chunk) in x.chunks(256).enumerate() {
            if blk == 60 {
                let mut c = ChannelCalibration::default();
                c.delay_samples = 480.0;
                c.gain_db = -10.0;
                c.eq[0] = EqBand::Peaking { hz: 300.0, q: 2.0, gain_db: 6.0 };
                cal.set_config(&CalibrationConfig { channels: vec![c], ramp_ms: 50.0 }).unwrap();
            }
            let mut b = AudioBuffer::new(1, chunk.len() as u16);
            b.channel_mut(0).copy_from_slice(chunk);
            cal.process(&mut b);
            y.extend_from_slice(b.channel(0));
        }
        let max_jump = y.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        let plain = 0.5 * 2.0 * std::f32::consts::PI * 300.0 / SR;
        eprintln!("max sample jump {max_jump:.4}, plain sine slope {plain:.4}");
        assert!(max_jump < 1.5 * plain, "click: {max_jump}");
        assert!(!cal.is_fading());
    }

    #[test]
    fn invalid_configurations_are_rejected_and_change_nothing() {
        let mut cal = SpeakerCalibration::new(SR, 2, 0.05).unwrap();
        assert!(matches!(cal.set_config(&CalibrationConfig::flat(3)), Err(CalibrationError::ChannelMismatch { .. })));
        let mut cfg = CalibrationConfig::flat(2);
        cfg.channels[1].gain_db = 40.0;
        assert_eq!(cal.set_config(&cfg), Err(CalibrationError::BadParameter(1)));
        let mut cfg = CalibrationConfig::flat(2);
        cfg.channels[0].delay_samples = 1e6;
        assert!(matches!(cal.set_config(&cfg), Err(CalibrationError::DelayTooLong { .. })));
        let mut cfg = CalibrationConfig::flat(2);
        cfg.channels[0].eq[0] = EqBand::Peaking { hz: f32::NAN, q: 1.0, gain_db: 3.0 };
        assert!(cal.set_config(&cfg).is_err());
        assert!(!cal.is_fading());
    }
}
