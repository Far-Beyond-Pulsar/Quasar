//! Parametric binaural renderer tests (`quasar_dsp::binaural`).

use quasar_dsp::binaural::{woodworth_itd_seconds, BinauralConfig, BinauralRenderer, ParametricBinauralRenderer};
use std::f32::consts::PI;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn deg(d: f32) -> f32 {
    d.to_radians()
}

fn renderer() -> ParametricBinauralRenderer {
    ParametricBinauralRenderer::with_sample_rate(SR)
}

/// Render `input` in blocks with the direction given per block; returns (L, R).
fn render_with(
    r: &mut ParametricBinauralRenderer,
    input: &[f32],
    dir_of_block: impl Fn(usize) -> (f32, f32),
) -> (Vec<f32>, Vec<f32>) {
    let mut l = vec![0.0; input.len()];
    let mut rr = vec![0.0; input.len()];
    for (b, start) in (0..input.len()).step_by(BLOCK).enumerate() {
        let end = (start + BLOCK).min(input.len());
        let (az, el) = dir_of_block(b);
        r.render_add(&input[start..end], az, el, &mut l[start..end], &mut rr[start..end]);
    }
    (l, rr)
}

fn render_fixed(input: &[f32], az: f32, el: f32) -> (Vec<f32>, Vec<f32>) {
    let mut r = renderer();
    render_with(&mut r, input, |_| (az, el))
}

fn sine(freq: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|i| amp * (2.0 * PI * freq * i as f32 / SR).sin()).collect()
}

/// Deterministic white-ish noise in [-1, 1].
fn noise(n: usize) -> Vec<f32> {
    let mut s: u32 = 0x1234_5678;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn db(ratio: f32) -> f32 {
    20.0 * ratio.log10()
}

fn argmax_abs(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if v.abs() > x[best].abs() {
            best = i;
        }
    }
    best
}

fn tail_rms(x: &[f32]) -> f32 {
    rms(&x[x.len() / 2..])
}

#[test]
fn itd_sign_and_magnitude_match_woodworth() {
    let expected = woodworth_itd_seconds(PI / 2.0, 0.0875, 343.0) * SR;
    assert!((expected - 31.5).abs() < 0.2, "sanity: Woodworth max ITD = {expected} samples");
    let mut imp = vec![0.0; 1024];
    imp[100] = 1.0;
    for &(az, sign) in &[(90.0f32, 1.0f32), (-90.0, -1.0)] {
        let (l, r) = render_fixed(&imp, deg(az), 0.0);
        // Positive = right ear leads.
        let measured = argmax_abs(&l) as f32 - argmax_abs(&r) as f32;
        assert!(
            (measured - sign * expected).abs() <= 2.0,
            "az {az}: measured ITD {measured} samples vs Woodworth {}",
            sign * expected
        );
    }
    // Straight ahead: no ITD.
    let (l, r) = render_fixed(&imp, 0.0, 0.0);
    assert!((argmax_abs(&l) as i32 - argmax_abs(&r) as i32).abs() <= 1);
    // The analytic helper agrees at an intermediate angle (45 degrees).
    let rnd = renderer();
    let want = 0.0875 / 343.0 * (PI / 4.0 + (PI / 4.0).sin());
    assert!((rnd.itd_seconds(deg(45.0), 0.0) - want).abs() < 1e-7);
    // Odd symmetry.
    assert!((rnd.itd_seconds(deg(-45.0), 0.0) + want).abs() < 1e-7);
}

#[test]
fn head_radius_scales_the_itd() {
    let mut cfg = BinauralConfig::new(SR);
    cfg.head_radius = 0.10;
    let big = ParametricBinauralRenderer::new(cfg);
    let small = renderer();
    let ratio = big.itd_seconds(PI / 2.0, 0.0) / small.itd_seconds(PI / 2.0, 0.0);
    assert!((ratio - 0.10 / 0.0875).abs() < 1e-3);
}

#[test]
fn far_ear_loses_high_frequencies_but_not_lows() {
    let n = 8192;
    // Source at the right (+90 deg).
    let (l, r) = render_fixed(&sine(6000.0, n, 0.5), deg(90.0), 0.0);
    let hf = db(tail_rms(&r) / tail_rms(&l));
    assert!(hf > 6.0, "6 kHz ILD only {hf} dB (near ear must exceed far ear by > 6 dB)");
    let (l, r) = render_fixed(&sine(200.0, n, 0.5), deg(90.0), 0.0);
    let lf = db(tail_rms(&r) / tail_rms(&l)).abs();
    assert!(lf < 3.0, "200 Hz level difference {lf} dB should be small");
}

#[test]
fn left_right_are_mirror_symmetric_even_while_moving() {
    let input = noise(BLOCK * 12);
    let path = |b: usize| (deg(-80.0 + 15.0 * b as f32), deg(20.0 - 3.0 * b as f32));
    let mirrored = |b: usize| {
        let (az, el) = path(b);
        (-az, el)
    };
    let mut r1 = renderer();
    let mut r2 = renderer();
    let (l1, rr1) = render_with(&mut r1, &input, path);
    let (l2, rr2) = render_with(&mut r2, &input, mirrored);
    for i in 0..input.len() {
        assert!((l1[i] - rr2[i]).abs() < 1e-5, "L(+az) != R(-az) at {i}");
        assert!((rr1[i] - l2[i]).abs() < 1e-5, "R(+az) != L(-az) at {i}");
    }
}

#[test]
fn front_and_back_differ() {
    let n = 8192;
    // Same lateral position (az 30 vs 150 share sin(az)); only the pinna cues differ.
    let (fl, fr) = render_fixed(&sine(6000.0, n, 0.5), deg(30.0), 0.0);
    let (bl, br) = render_fixed(&sine(6000.0, n, 0.5), deg(150.0), 0.0);
    let front = tail_rms(&fl).hypot(tail_rms(&fr));
    let back = tail_rms(&bl).hypot(tail_rms(&br));
    assert!(db(front / back) > 1.0, "front/back HF difference only {} dB", db(front / back));
}

#[test]
fn elevation_changes_the_spectrum() {
    let n = 8192;
    let (l0, _) = render_fixed(&sine(9000.0, n, 0.5), 0.0, 0.0);
    let (l1, _) = render_fixed(&sine(9000.0, n, 0.5), 0.0, deg(60.0));
    assert!(db(tail_rms(&l0) / tail_rms(&l1)).abs() > 0.5, "elevation has no spectral effect");
}

#[test]
fn output_is_finite_for_any_input_angle() {
    let input = noise(BLOCK * 4);
    let mut r = renderer();
    let angles = [
        (0.0, 0.0),
        (f32::NAN, 0.0),
        (0.0, f32::INFINITY),
        (1.0e6, -1.0e6),
        (-7.5, 3.0),
        (PI, PI / 2.0),
        (-PI, -PI / 2.0),
    ];
    for &(az, el) in &angles {
        let (l, rr) = render_with(&mut r, &input, |_| (az, el));
        assert!(l.iter().chain(rr.iter()).all(|v| v.is_finite()), "non-finite output for az={az} el={el}");
    }
}

#[test]
fn silence_in_gives_silence_out() {
    let input = vec![0.0; BLOCK * 8];
    let mut r = renderer();
    let (l, rr) = render_with(&mut r, &input, |b| (deg(40.0 * b as f32), deg(10.0 * b as f32 - 30.0)));
    assert!(l.iter().chain(rr.iter()).all(|v| *v == 0.0));
}

#[test]
fn swept_azimuth_has_no_clicks() {
    // 500 Hz at 0.5: natural max step is 0.5 * 2 pi 500 / 48000 = 0.033 (up to ~1.5x
    // with the head-shadow boost). A step discontinuity would exceed 0.15 easily.
    let blocks = 120;
    let input = sine(500.0, BLOCK * blocks, 0.5);
    let mut r = renderer();
    let (l, rr) = render_with(&mut r, &input, |b| (deg(-170.0 + 340.0 * b as f32 / blocks as f32), deg(20.0)));
    let mut worst = 0.0f32;
    for ch in [&l, &rr] {
        for w in ch.windows(2) {
            worst = worst.max((w[1] - w[0]).abs());
        }
    }
    assert!(worst < 0.15, "max sample step {worst} on a swept azimuth");
    assert!(tail_rms(&l) > 0.05 && tail_rms(&rr) > 0.05, "sweep rendered nothing");
}

#[test]
fn delay_ramps_instead_of_jumping_between_blocks() {
    // 1 kHz sine, azimuth jumps 0 -> +90 -> -90 -> 0 at block boundaries: each ear's
    // delay moves by ~16 samples. Un-ramped, that is a 1/3-cycle phase jump
    // (step ~0.5 at amplitude 0.5); ramped, the step stays near the natural slope.
    let blocks = 16;
    let input = sine(1000.0, BLOCK * blocks, 0.5);
    let seq = [0.0f32, 90.0, 90.0, -90.0, -90.0, 0.0];
    let mut r = renderer();
    let (l, rr) = render_with(&mut r, &input, |b| (deg(seq[(b / 2).min(seq.len() - 1)]), 0.0));
    let natural = 0.5 * 2.0 * PI * 1000.0 / SR; // 0.065
    let mut worst = 0.0f32;
    for ch in [&l, &rr] {
        for w in ch.windows(2) {
            worst = worst.max((w[1] - w[0]).abs());
        }
    }
    assert!(worst < natural * 3.0, "max step {worst} vs natural slope {natural}: delay is jumping");
}

#[test]
fn render_add_accumulates_and_reset_clears_state() {
    let input = noise(BLOCK);
    let mut r = renderer();
    let mut l = vec![1.0; BLOCK];
    let mut rr = vec![1.0; BLOCK];
    r.render_add(&input, 0.5, 0.0, &mut l, &mut rr);
    // Starts from the 1.0 already in the buffers.
    assert!(l.iter().any(|v| (*v - 1.0).abs() > 1e-3));
    let (a, b) = render_fixed(&input, 0.5, 0.0);
    for i in 0..BLOCK {
        assert!((l[i] - 1.0 - a[i]).abs() < 1e-6 && (rr[i] - 1.0 - b[i]).abs() < 1e-6);
    }
    // After reset a second identical render reproduces the first exactly.
    r.reset();
    let mut l2 = vec![0.0; BLOCK];
    let mut r2 = vec![0.0; BLOCK];
    r.render_add(&input, 0.5, 0.0, &mut l2, &mut r2);
    assert_eq!(l2, a);
    assert_eq!(r2, b);
}
