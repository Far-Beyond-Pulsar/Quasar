//! #76: band-limited polyphase resampler quality, continuity and real-time safety.
//!
//! Measurement methods (no external DSP crates):
//! * passband flatness: a linear sweep and stationary tones are resampled; the amplitude of the
//!   tone at the instantaneous frequency is measured with a Hann-windowed Goertzel on the same time
//!   slice of input and output, and compared in dB;
//! * image / alias products: a stationary tone is resampled, a Kaiser (beta = 14, side lobes below
//!   -110 dB) windowed radix-2 FFT of 65536 output samples is taken, and the largest bin outside
//!   +-12 bins of the tone is compared with the tone's peak bin.

mod common;

use common::count_allocs;
use quasar_audio::resampler::PolyphaseResampler;
use std::f64::consts::PI;

/// Resample `input` (mono) with the given chunk sizes; `in_chunk` frames are presented per call and
/// `out_chunk` frames requested per call.
fn resample_all(r: &mut PolyphaseResampler, input: &[f32], in_chunk: usize, out_chunk: usize) -> Vec<f32> {
    let mut out = Vec::new();
    let mut buf = vec![0.0f32; out_chunk];
    let mut pos = 0;
    loop {
        let end = (pos + in_chunk).min(input.len());
        let res = r.process(&[&input[pos..end]], &mut [&mut buf[..]]);
        out.extend_from_slice(&buf[..res.produced]);
        pos += res.consumed;
        if res.produced == 0 && res.consumed == 0 {
            break;
        }
        if pos >= input.len() && res.produced < out_chunk {
            break;
        }
    }
    out
}

fn sine(n: usize, f: f64, fs: f64, amp: f64) -> Vec<f32> {
    (0..n).map(|i| (amp * (2.0 * PI * f * i as f64 / fs).sin()) as f32).collect()
}

/// Pseudo-random signal in [-1, 1] (deterministic).
fn noise(n: usize, mut s: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// Hann-windowed Goertzel amplitude of frequency `f` in `x` (sample rate `fs`).
fn tone_amp(x: &[f32], f: f64, fs: f64) -> f64 {
    let n = x.len();
    let (mut re, mut im, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    for (i, v) in x.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * (i as f64 + 0.5) / n as f64).cos();
        let ph = 2.0 * PI * f * i as f64 / fs;
        re += *v as f64 * w * ph.cos();
        im += *v as f64 * w * ph.sin();
        wsum += w;
    }
    2.0 * (re * re + im * im).sqrt() / wsum
}

/// Passband edge used by the resampler design (and by these tests).
fn passband_edge(in_rate: f64, out_rate: f64) -> f64 {
    18_000.0 * in_rate.min(out_rate) / 44_100.0
}

const PAIRS: [(f64, f64); 5] = [(44_100.0, 48_000.0), (48_000.0, 44_100.0), (96_000.0, 48_000.0), (48_000.0, 96_000.0), (44_100.0, 96_000.0)];

// ── 1:1 ───────────────────────────────────────────────────────────────

#[test]
fn one_to_one_is_an_exact_copy_with_zero_latency() {
    let input = noise(5000, 1);
    for (ic, oc) in [(5000, 5000), (1, 1), (7, 97), (256, 13), (97, 256)] {
        let mut r = PolyphaseResampler::new(1, 48_000.0, 48_000.0);
        let out = resample_all(&mut r, &input, ic, oc);
        assert_eq!(out, input, "1:1 must be bit-identical (chunks {ic}/{oc})");
    }
    // Multi-channel, planar.
    let (a, b) = (noise(1000, 2), noise(1000, 3));
    let mut r = PolyphaseResampler::new(2, 48_000.0, 48_000.0);
    let (mut oa, mut ob) = (vec![0.0f32; 1000], vec![0.0f32; 1000]);
    let res = r.process(&[&a, &b], &mut [&mut oa[..], &mut ob[..]]);
    assert_eq!((res.consumed, res.produced), (1000, 1000));
    assert_eq!((oa, ob), (a, b));
}

// ── length, phase continuity, chunking independence ──────────────────

#[test]
fn output_is_identical_for_any_chunking_and_has_the_exact_length() {
    for (fin, fout) in PAIRS {
        let input = noise(12_345, 7);
        let mut r = PolyphaseResampler::new(1, fin, fout);
        let reference = resample_all(&mut r, &input, input.len(), 20_000);
        // Exact length: outputs k with floor(k * step) + half <= M - 1  =>  ceil((M - half) / step).
        let step = fin / fout;
        let expect = (((input.len() - r.lookahead_frames()) as f64) / step).ceil() as usize;
        assert_eq!(reference.len(), expect, "{fin} -> {fout}: output length");
        for (ic, oc) in [(1, 1), (7, 97), (255, 61), (97, 256), (4096, 3), (13, 5000)] {
            let mut r = PolyphaseResampler::new(1, fin, fout);
            let got = resample_all(&mut r, &input, ic, oc);
            assert_eq!(got, reference, "{fin} -> {fout}: chunking {ic}/{oc} changed the output");
        }
    }
}

#[test]
fn sine_is_phase_accurate_across_odd_block_sizes() {
    for (fin, fout) in PAIRS {
        let f = 1_000.0;
        let input = sine(30_000, f, fin, 0.5);
        let mut r = PolyphaseResampler::new(1, fin, fout);
        let taps = r.taps();
        let out = resample_all(&mut r, &input, 173, 97);
        let mut max_err = 0.0f64;
        let mut max_step = 0.0f64;
        for k in taps..out.len() {
            let want = 0.5 * (2.0 * PI * f * k as f64 / fout).sin();
            max_err = max_err.max((out[k] as f64 - want).abs());
            max_step = max_step.max((out[k] as f64 - out[k - 1] as f64).abs());
        }
        assert!(max_err < 2e-4, "{fin} -> {fout}: sample error vs the ideal sine {max_err}");
        // No clicks: the largest sample-to-sample step is the sine's own slope.
        let slope = 0.5 * 2.0 * PI * f / fout;
        assert!(max_step < slope * 1.01, "{fin} -> {fout}: click, step {max_step} vs slope {slope}");
    }
}

// ── passband ──────────────────────────────────────────────────────────

#[test]
fn swept_sine_passband_is_flat_within_0_1_db() {
    for (fin, fout) in PAIRS {
        let fp = passband_edge(fin, fout);
        let (f0, f1, secs) = (100.0, fp, 3.0);
        let n = (secs * fin) as usize;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / fin;
                (0.5 * (2.0 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * secs))).sin()) as f32
            })
            .collect();
        let mut r = PolyphaseResampler::new(1, fin, fout);
        let out = resample_all(&mut r, &input, 4096, 4096);
        let slice = 0.04; // seconds
        let mut worst = 0.0f64;
        let mut count = 0;
        let mut t0 = 0.1;
        while t0 + slice < secs - 0.1 {
            let tc = t0 + slice / 2.0;
            let f = f0 + (f1 - f0) * tc / secs; // instantaneous frequency at the slice centre
            let xi = &input[(t0 * fin) as usize..((t0 + slice) * fin) as usize];
            let xo = &out[(t0 * fout) as usize..((t0 + slice) * fout) as usize];
            let db = 20.0 * (tone_amp(xo, f, fout) / tone_amp(xi, f, fin)).log10();
            worst = worst.max(db.abs());
            count += 1;
            t0 += 0.05;
        }
        assert!(count > 40);
        assert!(worst < 0.1, "{fin} -> {fout}: passband deviation {worst:.4} dB (up to {fp} Hz)");
    }
}

#[test]
fn stationary_tones_pass_with_unity_gain() {
    for (fin, fout) in PAIRS {
        let fp = passband_edge(fin, fout);
        for f in [50.0, 440.0, 1_000.0, 5_000.0, 10_000.0, 15_000.0, fp * 0.99] {
            let input = sine(60_000, f, fin, 0.5);
            let mut r = PolyphaseResampler::new(1, fin, fout);
            let out = resample_all(&mut r, &input, 1024, 1024);
            let (a, b) = ((0.2 * fout) as usize, (0.8 * fout) as usize); // steady middle
            let db = 20.0 * (tone_amp(&out[a..b], f, fout) / 0.5).log10();
            assert!(db.abs() < 0.02, "{fin} -> {fout} at {f} Hz: {db:.4} dB");
        }
    }
}

// ── images and aliases ───────────────────────────────────────────────

fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (ur, ui) = (re[start + k], im[start + k]);
                let (vr, vi) = (
                    re[start + k + len / 2] * cr - im[start + k + len / 2] * ci,
                    re[start + k + len / 2] * ci + im[start + k + len / 2] * cr,
                );
                re[start + k] = ur + vr;
                im[start + k] = ui + vi;
                re[start + k + len / 2] = ur - vr;
                im[start + k + len / 2] = ui - vi;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
        len <<= 1;
    }
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..200 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// (peak dB, worst spur dB relative to the peak) of a windowed FFT of `x` (length power of two).
fn spur_level_db(x: &[f32], f: f64, fs: f64) -> f64 {
    let n = x.len();
    let beta = 14.0;
    let i0b = bessel_i0(beta);
    let mut re: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            *v as f64 * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b
        })
        .collect();
    let mut im = vec![0.0; n];
    fft(&mut re, &mut im);
    let mag: Vec<f64> = (0..n / 2).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect();
    let tone_bin = (f / fs * n as f64).round() as usize;
    let peak = (tone_bin.saturating_sub(3)..=(tone_bin + 3).min(n / 2 - 1)).map(|k| mag[k]).fold(0.0, f64::max);
    let mut spur = 0.0f64;
    for (k, m) in mag.iter().enumerate() {
        if k + 12 < tone_bin || k > tone_bin + 12 {
            if k > 2 {
                spur = spur.max(*m); // ignore the DC / window skirt bins
            }
        }
    }
    20.0 * (spur / peak).log10()
}

#[test]
fn image_and_alias_products_are_below_minus_90_db() {
    for (fin, fout) in PAIRS {
        let fp = passband_edge(fin, fout);
        for f in [997.0, 4_321.0, 9_876.0, 14_321.0, fp * 0.98] {
            let n_out = 65_536 + 4096;
            let n_in = ((n_out as f64) * fin / fout) as usize + 4096;
            let input = sine(n_in, f, fin, 0.5);
            let mut r = PolyphaseResampler::new(1, fin, fout);
            let out = resample_all(&mut r, &input, 4096, 4096);
            let seg = &out[2048..2048 + 65_536];
            let spur = spur_level_db(seg, f, fout);
            assert!(spur < -90.0, "{fin} -> {fout}, tone {f:.0} Hz: worst spur {spur:.1} dB re the tone");
        }
    }
}

#[test]
fn input_above_the_lower_nyquist_is_rejected_when_downsampling() {
    // 96 -> 48: a 30 kHz tone (above the output Nyquist) must not alias into the output band.
    let input = sine(200_000, 30_000.0, 96_000.0, 0.5);
    let mut r = PolyphaseResampler::new(1, 96_000.0, 48_000.0);
    let out = resample_all(&mut r, &input, 4096, 4096);
    let peak = out[1000..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak < 0.5 * 10f32.powf(-90.0 / 20.0) * 4.0, "30 kHz leaked: peak {peak}");
}

// ── streaming / demo flow, multichannel, drift, allocation ───────────

#[test]
fn input_needed_flow_produces_exact_blocks() {
    // The callback flow: input_needed(block) -> fetch -> process -> advance by `consumed`.
    for (fin, fout) in [(44_100.0, 48_000.0), (48_000.0, 44_100.0), (48_000.0, 48_000.0), (96_000.0, 48_000.0)] {
        let src = noise(100_000, 11);
        let mut r = PolyphaseResampler::new(1, fin, fout);
        let mut r_ref = PolyphaseResampler::new(1, fin, fout);
        let reference = resample_all(&mut r_ref, &src, src.len(), 90_000);
        let (mut got, mut in_frame) = (Vec::new(), 0usize);
        let mut buf = vec![0.0f32; 256];
        let mut scratch = vec![0.0f32; 8192];
        for blk in 0..300 {
            let block = if blk % 3 == 0 { 256 } else { 173 }; // device buffers of different sizes
            let need = r.input_needed(block);
            assert!(need <= scratch.len());
            scratch[..need].copy_from_slice(&src[in_frame..in_frame + need]);
            let res = r.process(&[&scratch[..need]], &mut [&mut buf[..block]]);
            assert_eq!(res.produced, block, "{fin} -> {fout}: block {blk} underfilled");
            assert!(res.consumed <= need);
            in_frame += res.consumed;
            got.extend_from_slice(&buf[..block]);
        }
        assert_eq!(&got[..], &reference[..got.len()], "{fin} -> {fout}: flow differs from the one-shot result");
        // Exact bookkeeping: the position advanced by exactly the consumed ratio.
        let expect = got.len() as f64 * fin / fout;
        assert!((r.position() - expect).abs() < 1e-6, "position {} vs {}", r.position(), expect);
    }
}

#[test]
fn channels_are_independent() {
    let (a, b) = (noise(8000, 21), sine(8000, 3_000.0, 44_100.0, 0.7));
    let mut r2 = PolyphaseResampler::new(2, 44_100.0, 48_000.0);
    let (mut oa, mut ob) = (vec![0.0f32; 9000], vec![0.0f32; 9000]);
    let res = r2.process(&[&a, &b], &mut [&mut oa[..], &mut ob[..]]);
    let n = res.produced;
    let mut ra = PolyphaseResampler::new(1, 44_100.0, 48_000.0);
    let mut rb = PolyphaseResampler::new(1, 44_100.0, 48_000.0);
    assert_eq!(&oa[..n], &resample_all(&mut ra, &a, 8000, 9000)[..n]);
    assert_eq!(&ob[..n], &resample_all(&mut rb, &b, 8000, 9000)[..n]);
}

#[test]
fn drift_trim_changes_the_consumption_ratio_without_clicks() {
    let (fin, fout) = (48_000.0, 48_000.0);
    let n_out = 600_000;
    let input = sine(n_out + 10_000, 1_000.0, fin, 0.5);
    let mut r = PolyphaseResampler::new(1, fin, fout);
    let mut buf = vec![0.0f32; 512];
    let mut y: Vec<f32> = Vec::new();
    let mut in_pos = 0;
    // 100 ms of nominal playback, then a +500 ppm trim (the source ring is filling up).
    while y.len() < n_out {
        if y.len() >= 4800 && r.ratio_trim() == 0.0 {
            r.set_ratio_trim(500e-6);
        }
        let need = r.input_needed(512);
        let res = r.process(&[&input[in_pos..in_pos + need]], &mut [&mut buf[..]]);
        in_pos += res.consumed;
        y.extend_from_slice(&buf[..res.produced]);
    }
    assert!((r.ratio_trim() - 500e-6).abs() < 1e-12, "slew completed");
    // Position: nominal for 4800 frames, then a slewed ramp to +500 ppm, then constant.
    let ramp = 500e-6 / quasar_audio::resampler::TRIM_SLEW_PER_SAMPLE;
    let expect = y.len() as f64 + ((y.len() as f64 - 4800.0) - ramp) * 500e-6 + ramp * 500e-6 / 2.0;
    assert!((r.position() - expect).abs() < 2.0, "position {} vs expected {}", r.position(), expect);
    assert!(r.position() > y.len() as f64 + 100.0, "consumption really sped up (by ~{} frames)", r.position() - y.len() as f64);
    // Click-free: no sample step beyond the (slightly pitched) sine's slope.
    let slope = 0.5 * 2.0 * PI * 1_000.0 / fout * 1.0006;
    let max_step = y.windows(2).skip(200).map(|w| (w[1] - w[0]).abs() as f64).fold(0.0, f64::max);
    assert!(max_step < slope * 1.001, "click while trimming: {max_step} vs {slope}");
}

#[test]
fn process_does_not_allocate() {
    let src = noise(200_000, 5);
    let mut r = PolyphaseResampler::new(2, 44_100.0, 48_000.0);
    let mut out = vec![vec![0.0f32; 256], vec![0.0f32; 256]];
    let mut scratch = vec![vec![0.0f32; 4096], vec![0.0f32; 4096]];
    let mut in_frame = 0;
    // Warm up, then count.
    for round in 0..2 {
        let ((), n) = count_allocs(|| {
            for blk in 0..300 {
                let need = r.input_needed(256);
                for ch in 0..2 {
                    scratch[ch][..need].copy_from_slice(&src[in_frame..in_frame + need]);
                }
                let (o0, o1) = out.split_at_mut(1);
                let res = r.process(&[&scratch[0][..need], &scratch[1][..need]], &mut [&mut o0[0][..], &mut o1[0][..]]);
                in_frame += res.consumed;
                if blk == 100 {
                    r.set_ratio_trim(200e-6);
                }
            }
        });
        if round == 1 {
            assert_eq!(n, 0, "process / input_needed allocated {n} time(s)");
        }
    }
}
