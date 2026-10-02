//! Biquad design tests for the constructors added for the binaural and LFE paths.

use quasar_dsp::biquad::BiquadFilter;
use std::f32::consts::PI;

const SR: f32 = 48_000.0;

/// Steady-state gain (dB) of a cascade at `freq`, measured with a sine.
fn gain_db(filters: &mut [BiquadFilter], freq: f32) -> f32 {
    let n = 48_000;
    let mut sum_in = 0.0f64;
    let mut sum_out = 0.0f64;
    for i in 0..n {
        let x = (2.0 * PI * freq * i as f32 / SR).sin();
        let mut y = x;
        for f in filters.iter_mut() {
            y = f.process(y);
        }
        if i > n / 2 {
            sum_in += (x * x) as f64;
            sum_out += (y * y) as f64;
        }
    }
    10.0 * (sum_out / sum_in).log10() as f32
}

#[test]
fn butterworth4_lowpass_at_120hz() {
    let make = || {
        let (mut a, mut b) = (BiquadFilter::new(), BiquadFilter::new());
        a.set_lowpass_q(120.0, 0.5412, SR);
        b.set_lowpass_q(120.0, 1.3066, SR);
        [a, b]
    };
    assert!(gain_db(&mut make(), 60.0).abs() < 0.3);
    assert!((gain_db(&mut make(), 120.0) + 3.0).abs() < 0.5, "-3 dB corner");
    let hf = gain_db(&mut make(), 1000.0);
    assert!(hf < -60.0, "1 kHz only {hf} dB down");
}

#[test]
fn peaking_hits_its_gain_at_the_centre_and_is_flat_far_away() {
    for &g in &[-9.0f32, -3.0, 6.0] {
        let mut f = [BiquadFilter::new()];
        f[0].set_peaking(7000.0, 2.5, g, SR);
        assert!((gain_db(&mut f, 7000.0) - g).abs() < 0.2, "centre gain for {g} dB");
        let mut f = [BiquadFilter::new()];
        f[0].set_peaking(7000.0, 2.5, g, SR);
        assert!(gain_db(&mut f, 100.0).abs() < 0.2);
    }
    // 0 dB is the identity.
    let mut f = [BiquadFilter::new()];
    f[0].set_peaking(7000.0, 2.5, 0.0, SR);
    assert!(gain_db(&mut f, 7000.0).abs() < 1e-3);
}

#[test]
fn high_shelf_reaches_its_gain_above_the_corner() {
    for &g in &[-5.0f32, 4.0] {
        let mut f = [BiquadFilter::new()];
        f[0].set_high_shelf(4000.0, g, SR);
        assert!(gain_db(&mut f, 100.0).abs() < 0.2, "lows untouched");
        let mut f = [BiquadFilter::new()];
        f[0].set_high_shelf(4000.0, g, SR);
        assert!((gain_db(&mut f, 16000.0) - g).abs() < 0.6, "highs at {g} dB");
    }
}

#[test]
fn coefficients_roundtrip() {
    let mut f = BiquadFilter::new();
    f.set_coefficients(0.1, 0.2, 0.3, 0.4, 0.5);
    assert_eq!(f.coefficients(), [0.1, 0.2, 0.3, 0.4, 0.5]);
}
