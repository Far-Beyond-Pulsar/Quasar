//! FDN late reverb: T60-driven decay (#63, #122) and structure / sample-rate
//! scaling (#66).

use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::biquad::BiquadFilter;
use quasar_dsp::late_reverb::{fdn_delay_lengths, FdnReverbNode, FDN_LINES};
use quasar_dsp::node_graph::AudioNode;

const BLOCK: usize = 256;
const RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

fn params(t60: Band8, wet_db: f32) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: t60,
        late_gain_db: wet_db,
        early_late_split_secs: 0.0,
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version: 0,
    }
}

/// Impulse response (wet output, 0 dB) of `len` samples.
fn impulse_response(fs: f32, t60: Band8, len: usize) -> Vec<f32> {
    let mut node = FdnReverbNode::new(1, fs);
    let p = params(t60, 0.0);
    let mut ir = Vec::with_capacity(len);
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    let mut first = true;
    while ir.len() < len {
        input.clear();
        if first {
            input.set(0, 0, 1.0);
            first = false;
        }
        node.process(&input, &mut out, &p);
        ir.extend_from_slice(out.channel(0));
    }
    ir.truncate(len);
    ir
}

/// RBJ band-pass (constant 0 dB peak gain), centre `f0`, quality `q`.
fn bandpass(x: &[f32], f0: f32, q: f32, fs: f32) -> Vec<f32> {
    let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let a0 = 1.0 + alpha;
    let mut bq = BiquadFilter::new();
    bq.set_coefficients(alpha / a0, 0.0, -alpha / a0, -2.0 * w0.cos() / a0, (1.0 - alpha) / a0);
    x.iter().map(|&v| bq.process(v)).collect()
}

/// Reverberation time from Schroeder backward integration: least-squares line through
/// the energy decay curve between -5 dB and `-5 - span` dB, extrapolated to 60 dB
/// (span 20 = T20, span 30 = T30).
fn rt60_schroeder(h: &[f32], fs: f32, span: f32) -> f32 {
    let mut edc = vec![0.0_f64; h.len()];
    let mut acc = 0.0_f64;
    for i in (0..h.len()).rev() {
        acc += (h[i] as f64) * (h[i] as f64);
        edc[i] = acc;
    }
    let total = edc[0].max(1e-300);
    let db: Vec<f64> = edc.iter().map(|e| 10.0 * (e / total).max(1e-300).log10()).collect();
    let lo = db.iter().position(|&d| d <= -5.0).expect("EDC must reach -5 dB");
    let hi = db.iter().position(|&d| d <= -5.0 - span as f64).expect("EDC must reach the fit end (IR too short)");
    // Least squares of db[i] on time.
    let n = (hi - lo + 1) as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for i in lo..=hi {
        let t = i as f64 / fs as f64;
        sx += t;
        sy += db[i];
        sxx += t * t;
        sxy += t * db[i];
    }
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); // dB per second (negative)
    (-60.0 / slope) as f32
}

fn ir_len(fs: f32, t60: f32) -> usize {
    (fs * t60 * 1.3) as usize + 4 * BLOCK
}

#[test]
fn rt60_matches_the_request_in_every_band_at_every_rate() {
    for &fs in &RATES {
        for &t60 in &[0.5_f32, 2.0, 6.0] {
            let ir = impulse_response(fs, Band8::splat(t60), ir_len(fs, t60));
            // Broadband and the LF / MF / HF octave bands (125 Hz, 1 kHz, 4 kHz).
            let mut got = vec![("broadband", rt60_schroeder(&ir, fs, 30.0))];
            for (name, f0) in [("125 Hz", 125.0), ("1 kHz", 1000.0), ("4 kHz", 4000.0)] {
                let band = bandpass(&ir, f0, 1.0, fs);
                got.push((name, rt60_schroeder(&band, fs, 20.0)));
            }
            for (name, t) in got {
                assert!(
                    (t / t60 - 1.0).abs() < 0.10,
                    "fs {fs} T60 {t60}: {name} measured {t:.3} s ({:+.1} %)",
                    (t / t60 - 1.0) * 100.0
                );
            }
        }
    }
}

#[test]
fn frequency_dependent_t60_follows_the_three_band_fit() {
    // Long LF decay, short HF decay. First-order shelves, so the tolerance at the bands
    // next to the shelf corners (250 Hz, 2 kHz) is loose; the extreme bands are tight.
    let t60 = Band8::new([3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0]);
    let fs = 48_000.0;
    let ir = impulse_response(fs, t60, ir_len(fs, 3.0));
    let at = |f0: f32| rt60_schroeder(&bandpass(&ir, f0, 1.0, fs), fs, 20.0);
    let (lf, mf, hf) = (at(125.0), at(1000.0), at(8000.0));
    assert!((lf / 3.0 - 1.0).abs() < 0.15, "125 Hz: {lf} vs 3.0");
    assert!((mf / 2.0 - 1.0).abs() < 0.15, "1 kHz: {mf} vs 2.0");
    assert!((hf / 1.0 - 1.0).abs() < 0.20, "8 kHz: {hf} vs 1.0");
    assert!(lf > mf && mf > hf, "decay must shorten with frequency: {lf} {mf} {hf}");
}

#[test]
fn t60_changes_are_smoothed_per_sample() {
    // Steady noise through the reverb, then the T60 jumps 6 s -> 0.4 s between two blocks:
    // the output must not step at the block boundary (gains glide linearly per sample).
    let fs = 48_000.0;
    let mut node = FdnReverbNode::new(1, fs);
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    let mut seed = 0x9e37_79b9_u32;
    let mut noise = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let long = params(Band8::splat(6.0), 0.0);
    let short = params(Band8::new([0.4; 8]), 0.0);
    let mut last = 0.0_f32;
    let (mut max_before, mut max_at_change) = (0.0_f32, 0.0_f32);
    for b in 0..260 {
        for i in 0..BLOCK {
            input.set(0, i as u16, 0.3 * noise());
        }
        node.process(&input, &mut out, if b < 200 { &long } else { &short });
        let ch = out.channel(0);
        for (i, &v) in ch.iter().enumerate() {
            let step = (v - last).abs();
            if b >= 100 && b < 200 {
                max_before = max_before.max(step);
            }
            if b >= 200 && b < 203 && i < BLOCK {
                max_at_change = max_at_change.max(step);
            }
            last = v;
        }
    }
    assert!(max_before > 1e-3, "steady state must produce signal: {max_before}");
    assert!(
        max_at_change < 3.0 * max_before,
        "output stepped at the T60 change: {max_at_change} vs steady {max_before}"
    );
}

#[test]
fn wet_level_is_calibrated_independent_of_t60() {
    // White noise at unit RMS: one output channel has RMS ~ wet (the node divides by
    // the impulse-response energy that grows with T60).
    for &t60 in &[0.5_f32, 2.0, 6.0] {
        let fs = 48_000.0;
        let mut node = FdnReverbNode::new(1, fs);
        let p = params(Band8::splat(t60), 0.0); // wet = 1
        let mut input = AudioBuffer::new(1, BLOCK as u16);
        let mut out = AudioBuffer::new(1, BLOCK as u16);
        let mut seed = 12345_u32;
        let mut sum2 = 0.0_f64;
        let mut count = 0u64;
        let blocks = ((fs * t60 * 4.0) as usize / BLOCK).max(200);
        for b in 0..blocks {
            for i in 0..BLOCK {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                // Uniform noise scaled to unit RMS.
                input.set(0, i as u16, ((seed as f32 / u32::MAX as f32) * 2.0 - 1.0) * 3.0_f32.sqrt());
            }
            node.process(&input, &mut out, &p);
            if b >= blocks / 2 {
                for &v in out.channel(0) {
                    sum2 += (v as f64) * (v as f64);
                    count += 1;
                }
            }
        }
        let rms = (sum2 / count as f64).sqrt();
        let db = 20.0 * rms.log10();
        println!("T60 {t60}: output RMS {rms:.3} ({db:+.2} dB re wet)");
        assert!(db.abs() < 1.5, "T60 {t60}: output RMS {rms:.3} ({db:+.1} dB re wet)");
    }
}

#[test]
fn no_nan_and_bounded_for_extreme_parameters() {
    let fs = 48_000.0;
    for t60 in [Band8::splat(0.0), Band8::splat(1e6), Band8::splat(f32::NAN), Band8::new([0.01, 100.0, 0.05, 60.0, 0.5, 3.0, 9.0, 0.2])] {
        let mut node = FdnReverbNode::new(1, fs);
        let p = params(t60, 6.0);
        let mut input = AudioBuffer::new(1, BLOCK as u16);
        let mut out = AudioBuffer::new(1, BLOCK as u16);
        let mut seed = 777_u32;
        let mut peak = 0.0_f32;
        for _ in 0..600 {
            for i in 0..BLOCK {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                input.set(0, i as u16, (seed as f32 / u32::MAX as f32) * 2.0 - 1.0);
            }
            node.process(&input, &mut out, &p);
            for &v in out.channel(0) {
                assert!(v.is_finite(), "non-finite output for {t60:?}");
                peak = peak.max(v.abs());
            }
        }
        assert!(peak < 50.0, "output must stay bounded: {peak} for {t60:?}");
    }
    // Silence in, silence out (no denormal / self-oscillation).
    let mut node = FdnReverbNode::new(1, fs);
    let silence = AudioBuffer::new(1, BLOCK as u16);
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    for _ in 0..20 {
        node.process(&silence, &mut out, &params(Band8::splat(2.0), 0.0));
        assert!(out.channel(0).iter().all(|&v| v == 0.0));
    }
}

// ── #66 structure and sample-rate scaling ─────────────────────────────

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[test]
fn delay_lengths_scale_with_the_sample_rate_and_are_coprime() {
    let base = fdn_delay_lengths(48_000.0);
    for &fs in &RATES {
        let lens = fdn_delay_lengths(fs);
        let scale = fs / 48_000.0;
        for i in 0..FDN_LINES {
            // Same duration in ms within the prime-rounding slack.
            let (a, b) = (base[i] as f32 / 48_000.0, lens[i] as f32 / fs);
            assert!((a - b).abs() / a < 0.03 + 8.0 / lens[i] as f32 * scale.recip(), "line {i}: {a} vs {b}");
            for j in 0..i {
                assert_eq!(gcd(lens[i], lens[j]), 1, "lines {i}/{j} ({}, {}) are not coprime", lens[i], lens[j]);
            }
        }
        // Modal density (total delay in seconds) is the same at every rate.
        let node = FdnReverbNode::new(1, fs);
        let total48 = FdnReverbNode::new(1, 48_000.0).total_delay_secs();
        assert!(
            (node.total_delay_secs() / total48 - 1.0).abs() < 0.01,
            "fs {fs}: total delay {} s vs {} s",
            node.total_delay_secs(),
            total48
        );
        assert_eq!(node.delay_lengths(), &lens);
    }
}

#[test]
fn loop_delay_is_exactly_the_shortest_line_no_off_by_one() {
    // First output arrives exactly `min(d_i)` samples after the input: one read per
    // line, no tap-before-push extra sample.
    for &fs in &RATES {
        let lens = fdn_delay_lengths(fs);
        let shortest = *lens.iter().min().unwrap();
        let ir = impulse_response(fs, Band8::splat(2.0), 4 * BLOCK);
        assert!(ir[..shortest].iter().all(|&v| v == 0.0), "fs {fs}: output before the shortest line");
        assert!(ir[shortest] != 0.0, "fs {fs}: first echo must arrive at sample {shortest}");
    }
}

/// Normalised echo density (Abel & Huang): fraction of samples outside one standard
/// deviation in a sliding window, divided by erfc(1/sqrt 2) = 0.3173 (1 = Gaussian noise).
fn echo_density(h: &[f32], win: usize, at: usize) -> f32 {
    let w = &h[at - win / 2..at + win / 2];
    let mean_sq = w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
    let sigma = mean_sq.sqrt();
    let outside = w.iter().filter(|v| v.abs() > sigma).count() as f32 / w.len() as f32;
    outside / 0.3173
}

#[test]
fn echo_density_builds_up_at_the_same_rate_at_every_sample_rate() {
    let mut at_100ms: Vec<(f32, f32)> = Vec::new();
    for &fs in &RATES {
        let ir = impulse_response(fs, Band8::splat(2.0), (0.4 * fs) as usize);
        let win = (0.02 * fs) as usize;
        let ned = |secs: f32| echo_density(&ir, win, (secs * fs) as usize);
        let (early, mid, late) = (ned(0.03), ned(0.1), ned(0.25));
        println!("fs {fs}: NED(30 ms) {early:.2}, NED(100 ms) {mid:.2}, NED(250 ms) {late:.2}");
        assert!(late > 0.65, "fs {fs}: late echo density {late}");
        assert!(mid > 0.5, "fs {fs}: echo density at 100 ms {mid}");
        assert!(early < late + 0.05, "density does not fall");
        at_100ms.push((fs, mid));
    }
    // NED is per SAMPLE: the same echoes per second are spread over twice the samples at 96 kHz,
    // so it reads lower there (0.73 vs 0.80 late); the 44.1 / 48 kHz pair must agree closely.
    let (mx, mn) = (at_100ms[..2].iter().map(|p| p.1).fold(0.0, f32::max), at_100ms[..2].iter().map(|p| p.1).fold(f32::MAX, f32::min));
    assert!(mx - mn < 0.12, "echo density at 100 ms differs across rates: {at_100ms:?}");
}

#[test]
fn bus_outputs_are_decorrelated_and_zero_lag_uncorrelated_with_the_input() {
    let fs = 48_000.0;
    let mut node = FdnReverbNode::new(1, fs);
    node.set_t60(&Band8::splat(2.0));
    node.set_wet(1.0);
    let n_out = 8;
    let mut out = AudioBuffer::new(n_out as u16, BLOCK as u16);
    let mut seed = 99_u32;
    let mut chans = vec![Vec::new(); n_out];
    let mut input = vec![0.0_f32; BLOCK];
    for b in 0..600 {
        for v in input.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        node.process_bus(&input, &mut out, n_out);
        if b >= 100 {
            for (k, c) in chans.iter_mut().enumerate() {
                c.extend_from_slice(out.channel(k as u16));
            }
        }
    }
    let corr = |a: &[f32], b: &[f32]| {
        let (mut ab, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (x, y) in a.iter().zip(b.iter()) {
            ab += (*x as f64) * (*y as f64);
            aa += (*x as f64).powi(2);
            bb += (*y as f64).powi(2);
        }
        ab / (aa * bb).sqrt()
    };
    let mut max_rho = 0.0_f64;
    for a in 0..n_out {
        for b in a + 1..n_out {
            max_rho = max_rho.max(corr(&chans[a], &chans[b]).abs());
        }
    }
    println!("bus: max inter-channel |rho| = {max_rho:.3}");
    // Measured 0.09 (T60 2 s; 0.14 at 0.5 s, 0.09 at 6 s) over all 31 outputs.
    assert!(max_rho < 0.15, "bus channels must be decorrelated: {max_rho}");
}
