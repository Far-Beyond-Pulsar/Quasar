//! Tests for `HermiteInterpolatingDelayLine` (issues #97 / #60).

use quasar_dsp::fractional_delay::HermiteInterpolatingDelayLine;

fn ramp_line() -> HermiteInterpolatingDelayLine {
    let mut dl = HermiteInterpolatingDelayLine::new(0.01, 48_000.0);
    for i in 0..200 {
        dl.push(i as f32);
    }
    dl
}

/// Newest sample is 199, one unit per sample: tap(d) must be 199 - d for every
/// fractional d, including the short-delay region below one sample.
#[test]
fn ramp_returns_exact_delayed_value() {
    let dl = ramp_line();
    let mut d = 0.0_f32;
    while d <= 100.0 {
        let got = dl.tap(d);
        assert!((got - (199.0 - d)).abs() < 1e-3, "tap({d}) = {got}, expected {}", 199.0 - d);
        d += 0.0625;
    }
    // The values quoted in the issue.
    for (d, e) in [(3.0, 196.0), (3.25, 195.75), (3.5, 195.5), (3.75, 195.25), (3.99, 195.01), (4.0, 195.0)] {
        assert!((dl.tap(d) - e).abs() < 1e-3, "tap({d}) = {}, expected {e}", dl.tap(d));
    }
}

/// Sweeping the delay through integer boundaries must not step by more than
/// the signal slope times the step (slope is 1 per sample here).
#[test]
fn sweep_through_integer_boundaries_is_continuous() {
    let dl = ramp_line();
    let step = 0.01_f32;
    let mut prev = dl.tap(1.0);
    let mut d = 1.0 + step;
    while d <= 50.0 {
        let cur = dl.tap(d);
        assert!(
            (cur - prev).abs() <= step * 1.01 + 1e-3,
            "step {} at d={d} exceeds slope * step",
            cur - prev
        );
        prev = cur;
        d += step;
    }
}

/// 1 kHz sine delayed by 10.37 samples matches the analytic phase.
#[test]
fn sine_fractional_delay_matches_analytic_phase() {
    let fs = 48_000.0_f32;
    let freq = 1000.0_f32;
    let delay = 10.37_f32;
    let w = 2.0 * std::f64::consts::PI * freq as f64 / fs as f64;
    let mut dl = HermiteInterpolatingDelayLine::new(0.01, fs);
    let n = 400;
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        dl.push((w * i as f64).sin() as f32);
        out[i] = dl.tap(delay);
    }
    // Estimate the phase offset over the last 200 samples by projecting the
    // output on sin/cos of the input frequency.
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    for i in 100..n {
        let ph = w * i as f64;
        s += out[i] as f64 * ph.sin();
        c += out[i] as f64 * ph.cos();
    }
    // out ~ sin(ph - w*delay) = sin(ph)cos(wd) - cos(ph)sin(wd)
    let measured = (-c).atan2(s);
    let expected = w * delay as f64;
    assert!((measured - expected).abs() < 0.01, "phase {measured} vs {expected} rad");
}

/// Out-of-range and non-finite delays are clamped, never read out of bounds.
#[test]
fn out_of_range_delay_is_clamped() {
    let mut dl = HermiteInterpolatingDelayLine::new(0.001, 48_000.0);
    for i in 0..1000 {
        dl.push(i as f32);
    }
    let max = (dl.max_samples() - 3) as f32;
    assert_eq!(dl.tap(1.0e9), dl.tap(max));
    assert!(dl.tap(f32::NAN).is_finite());
    assert!(dl.tap(f32::INFINITY).is_finite());
}
