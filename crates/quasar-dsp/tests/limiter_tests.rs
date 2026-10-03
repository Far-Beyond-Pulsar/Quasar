//! #80: output safety stage (limiter, scrub, meters, FTZ).

use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::limiter::{
    flush_to_zero_enabled, set_flush_to_zero, OutputSafety, OutputSafetyConfig, MAX_LOOKAHEAD_SAMPLES,
};

const SR: f32 = 48_000.0;

fn db(x: f32) -> f32 {
    10f32.powf(x / 20.0)
}

fn cfg(lookahead_ms: f32) -> OutputSafetyConfig {
    OutputSafetyConfig { lookahead_ms, ..OutputSafetyConfig::default() }
}

/// Run `signal` (one `Vec<f32>` per channel) through the stage in blocks of `block` samples.
fn run(stage: &mut OutputSafety, signal: &[Vec<f32>], block: usize) -> Vec<Vec<f32>> {
    let n = signal[0].len();
    let mut out = vec![Vec::with_capacity(n); signal.len()];
    let mut pos = 0;
    while pos < n {
        let len = block.min(n - pos);
        let mut b = AudioBuffer::new(signal.len() as u16, len as u16);
        for (c, ch) in signal.iter().enumerate() {
            b.channel_mut(c as u16).copy_from_slice(&ch[pos..pos + len]);
        }
        stage.process(&mut b);
        for c in 0..signal.len() {
            out[c].extend_from_slice(b.channel(c as u16));
        }
        pos += len;
    }
    out
}

fn sine(n: usize, freq: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| amp * (std::f32::consts::TAU * freq * i as f32 / SR).sin()).collect()
}

#[test]
fn output_never_exceeds_the_ceiling_for_an_overdriven_mix() {
    for lookahead_ms in [0.0, 1.0, 5.0] {
        let mut stage = OutputSafety::new(SR, cfg(lookahead_ms));
        let ceiling = db(-1.0);
        // +12 dB overdriven two-tone mix with hard bursts and a DC-ish step.
        let n = 48_000;
        let mut l = sine(n, 220.0, 2.0);
        let r: Vec<f32> = sine(n, 3_100.0, 1.0).iter().zip(&l).map(|(a, b)| a + 0.5 * b).collect();
        for i in 10_000..10_200 {
            l[i] += 6.0; // a violent burst
        }
        let out = run(&mut stage, &[l, r], 256);
        let peak = out.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak <= ceiling, "lookahead {lookahead_ms} ms: peak {peak} exceeds the ceiling {ceiling}");
        assert!(peak > ceiling * 0.9, "the limiter should be holding the level near the ceiling ({peak})");
        assert_eq!(stage.meter().hard_clipped_samples(), 0, "the gain computer, not the clamp, does the work");
        assert!(stage.meter().limited_samples() > 0);
        assert!(stage.meter().peak() <= ceiling);
    }
}

#[test]
fn ceiling_and_headroom_are_configurable() {
    let mut stage = OutputSafety::new(SR, OutputSafetyConfig { ceiling_db: -6.0, ..cfg(1.0) });
    let out = run(&mut stage, &[sine(9600, 440.0, 1.0)], 128);
    let peak = out[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak <= db(-6.0) && peak > db(-6.0) * 0.95, "peak {peak} vs ceiling {}", db(-6.0));

    // Headroom: a -6 dB pre-gain brings a full-scale-ish 0.9 sine under the -1 dBFS ceiling untouched.
    let mut stage = OutputSafety::new(SR, OutputSafetyConfig { headroom_db: -6.0, ..cfg(0.0) });
    let input = sine(4800, 440.0, 0.9);
    let out = run(&mut stage, &[input.clone()], 256);
    let expect: Vec<f32> = input.iter().map(|v| v * db(-6.0)).collect();
    assert_eq!(out[0], expect);
    assert_eq!(stage.meter().limited_samples(), 0);
}

#[test]
fn nan_and_inf_input_become_silence_and_count_as_errors() {
    for lookahead_ms in [0.0, 1.0] {
        let mut stage = OutputSafety::new(SR, cfg(lookahead_ms));
        let mut l = sine(2000, 440.0, 0.5);
        let mut r = sine(2000, 330.0, 0.5);
        l[500] = f32::NAN;
        l[501] = f32::INFINITY;
        r[900] = f32::NEG_INFINITY;
        for i in 1000..1100 {
            r[i] = f32::NAN; // a NaN run (e.g. a broken filter)
        }
        let out = run(&mut stage, &[l.clone(), r.clone()], 200);
        for ch in &out {
            assert!(ch.iter().all(|v| v.is_finite()), "NaN / inf leaked to the output");
        }
        assert_eq!(stage.meter().nonfinite_samples(), 103, "every bad sample is counted");
        // The bad samples are silent (after the look-ahead delay) and the limiter state is not
        // poisoned: the audio after them is untouched.
        let d = stage.latency_samples();
        assert_eq!(out[0][500 + d], 0.0);
        assert_eq!(out[0][501 + d], 0.0);
        assert_eq!(out[1][1050 + d], 0.0);
        for i in 1200..2000 - d {
            assert_eq!(out[0][i + d], l[i], "audio after a NaN burst must be unchanged (lookahead {lookahead_ms})");
        }
        assert_eq!(stage.meter().limited_samples(), 0, "scrubbed samples must not trigger the limiter");
    }
    // An all-NaN block is silence.
    let mut stage = OutputSafety::new(SR, cfg(0.0));
    let out = run(&mut stage, &[vec![f32::NAN; 256], vec![f32::NAN; 256]], 256);
    assert!(out.iter().flatten().all(|&v| v == 0.0));
    assert_eq!(stage.meter().nonfinite_samples(), 512);
}

#[test]
fn transparent_below_the_ceiling_and_after_a_limiting_event() {
    let n = 24_000;
    let quiet: Vec<f32> = sine(n, 997.0, 0.7);
    // Zero look-ahead: bit-identical.
    let mut stage = OutputSafety::new(SR, cfg(0.0));
    let out = run(&mut stage, &[quiet.clone(), quiet.clone()], 256);
    assert_eq!(out[0], quiet);
    assert_eq!(out[1], quiet);
    assert_eq!(stage.meter().limited_samples(), 0);
    assert_eq!(stage.meter().gain(), 1.0);

    // With look-ahead: the same signal, delayed by exactly the look-ahead.
    let mut stage = OutputSafety::new(SR, cfg(1.0));
    let d = stage.latency_samples();
    assert_eq!(d, 48);
    let out = run(&mut stage, &[quiet.clone(), quiet.clone()], 256);
    assert!(out[0][..d].iter().all(|&v| v == 0.0));
    assert_eq!(&out[0][d..], &quiet[..n - d]);

    // After a limiting event the gain recovers to EXACTLY 1.0 and the signal is bit-identical again.
    let mut sig = quiet.clone();
    for v in sig[2000..2400].iter_mut() {
        *v *= 5.0;
    }
    let mut stage = OutputSafety::new(SR, OutputSafetyConfig { release_ms: 20.0, ..cfg(1.0) });
    let out = run(&mut stage, &[sig.clone()], 256);
    assert!(stage.meter().limited_samples() > 0);
    assert_eq!(&out[0][d + 20_000..], &sig[20_000..n - d], "transparent again long after the event");
    assert_eq!(stage.meter().gain(), 1.0);
}

#[test]
fn reported_latency_equals_measured_latency() {
    for (ms, expect) in [(0.0, 0usize), (0.5, 24), (1.0, 48), (2.5, 120)] {
        let mut stage = OutputSafety::new(SR, cfg(ms));
        assert_eq!(stage.latency_samples(), expect);
        assert_eq!(cfg(ms).latency_samples(SR), expect);
        // An impulse (below the ceiling) at sample 37, in odd block sizes.
        let mut x = vec![0.0f32; 1500];
        x[37] = 0.5;
        let out = run(&mut stage, &[x.clone(), x], 97);
        let pos = out[0].iter().position(|&v| v != 0.0).expect("impulse");
        assert_eq!(pos, 37 + expect, "measured delay of the impulse (look-ahead {ms} ms)");
        assert_eq!(out[0][pos], 0.5);
        assert_eq!(out[1][pos], 0.5);
    }
    // Disabled limiter: zero latency regardless of the look-ahead setting.
    let stage = OutputSafety::new(SR, OutputSafetyConfig { enabled: false, ..cfg(5.0) });
    assert_eq!(stage.latency_samples(), 0);
    // Look-ahead is capped.
    let stage = OutputSafety::new(SR, cfg(1000.0));
    assert_eq!(stage.latency_samples(), MAX_LOOKAHEAD_SAMPLES);
}

#[test]
fn gain_is_linked_across_channels_and_releases() {
    let mut stage = OutputSafety::new(SR, OutputSafetyConfig { release_ms: 50.0, ..cfg(0.0) });
    // Left loud (2.0), right quiet (0.1): both get the same gain.
    let (l, r) = (vec![2.0f32; 256], vec![0.1f32; 256]);
    let out = run(&mut stage, &[l, r], 256);
    let g = out[0][100] / 2.0;
    assert!((out[1][100] / 0.1 - g).abs() < 1e-6, "channels must share one gain (image stability)");
    assert!(out[0].iter().all(|&v| v <= db(-1.0)));
    // Release: once the loud input stops the gain climbs back; after 5 time constants it is ~1.
    let q = vec![vec![0.1f32; 30_000], vec![0.1f32; 30_000]];
    let out = run(&mut stage, &q, 256);
    assert!(out[0][0] < 0.1 - 1e-4, "still reduced right after the event");
    assert!((out[0][29_999] - 0.1).abs() < 1e-5, "released after 625 ms (12.5 x 50 ms)");
}

#[test]
fn disabled_limiter_still_scrubs_and_meters() {
    let mut stage = OutputSafety::new(SR, OutputSafetyConfig { enabled: false, ..cfg(0.0) });
    let out = run(&mut stage, &[vec![3.0, f32::NAN, -2.5, 0.25]], 4);
    assert_eq!(out[0], vec![3.0, 0.0, -2.5, 0.25], "no limiting, NaN scrubbed");
    assert_eq!(stage.meter().nonfinite_samples(), 1);
    assert_eq!(stage.meter().peak(), 3.0);
}

#[test]
fn flush_to_zero_sets_ftz_daz_on_this_thread() {
    // FTZ is per thread; use a fresh one so the test thread's FP mode is untouched.
    std::thread::spawn(|| {
        // A denormal is a genuine non-zero value before FTZ is enabled.
        let tiny = std::hint::black_box(f32::MIN_POSITIVE / 4.0);
        assert!(tiny > 0.0 && !tiny.is_normal());
        match set_flush_to_zero() {
            true => {
                assert_eq!(flush_to_zero_enabled(), Some(true));
                // With DAZ the denormal operand reads as 0; with FTZ a denormal result flushes.
                assert_eq!(std::hint::black_box(tiny) * std::hint::black_box(1.0f32), 0.0);
                assert_eq!(std::hint::black_box(f32::MIN_POSITIVE) * std::hint::black_box(0.25f32), 0.0);
            }
            false => assert_eq!(flush_to_zero_enabled(), None),
        }
    })
    .join()
    .unwrap();
}
