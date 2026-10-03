//! Shared distance model (#51).

use quasar_core::distance::{DistanceCurve, DistanceModel};

#[test]
fn default_is_inverse_with_6_02_db_per_doubling() {
    let m = DistanceModel::default();
    assert!((m.gain(1.0) - 1.0).abs() < 1e-6);
    for d in [1.0_f32, 2.0, 4.0, 8.0, 64.0] {
        let db = 20.0 * (m.gain(2.0 * d) / m.gain(d)).log10();
        assert!((db + 6.0206).abs() < 1e-3, "doubling at {d} m: {db} dB");
    }
    assert!((m.gain(10.0) - 0.1).abs() < 1e-6);
}

#[test]
fn clamped_at_min_distance_never_exceeds_unity() {
    let m = DistanceModel::default();
    for d in [0.0_f32, 0.05, 0.5, 0.999] {
        assert!((m.gain(d) - 1.0).abs() < 1e-6, "d={d}");
    }
    assert!((m.gain(f32::NAN) - 1.0).abs() < 1e-6);
}

#[test]
fn reference_and_min_distance_are_honoured() {
    let m = DistanceModel { reference_distance: 2.0, min_distance: 0.5, ..DistanceModel::default() };
    assert!((m.gain(2.0) - 1.0).abs() < 1e-6);
    assert!((m.gain(4.0) - 0.5).abs() < 1e-6);
    // Clamp at 0.5 m: 2/(2 + (0.5-2)) = 4
    assert!((m.gain(0.0) - 4.0).abs() < 1e-5);
    assert!((DistanceModel::with_reference(2.0).gain(1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn max_distance_clamps_far_gain() {
    let m = DistanceModel { max_distance: 100.0, ..DistanceModel::default() };
    assert!((m.gain(100.0) - 0.01).abs() < 1e-7);
    assert!((m.gain(1.0e6) - 0.01).abs() < 1e-7);
}

#[test]
fn rolloff_factor_scales_the_slope() {
    let m = DistanceModel { rolloff_factor: 2.0, ..DistanceModel::default() };
    // 1 / (1 + 2 (d - 1))
    assert!((m.gain(3.0) - 0.2).abs() < 1e-6);
    assert!(m.gain(3.0) < DistanceModel::default().gain(3.0));
}

#[test]
fn linear_curve_reaches_zero_at_max_distance() {
    let m = DistanceModel { curve: DistanceCurve::Linear, max_distance: 101.0, ..DistanceModel::default() };
    assert!((m.gain(1.0) - 1.0).abs() < 1e-6);
    assert!((m.gain(51.0) - 0.5).abs() < 1e-6);
    assert!(m.gain(101.0).abs() < 1e-6);
    assert!(m.gain(500.0).abs() < 1e-6);
}

#[test]
fn exponential_curve_is_power_law() {
    let m = DistanceModel { curve: DistanceCurve::Exponential, rolloff_factor: 2.0, ..DistanceModel::default() };
    assert!((m.gain(2.0) - 0.25).abs() < 1e-6);
    assert!((m.gain(10.0) - 0.01).abs() < 1e-6);
}

#[test]
fn gain_is_finite_and_monotonic_for_all_curves() {
    for curve in [DistanceCurve::Inverse, DistanceCurve::Linear, DistanceCurve::Exponential] {
        let m = DistanceModel { curve, ..DistanceModel::default() };
        let mut prev = f32::INFINITY;
        let mut d = 0.0_f32;
        while d < 20_000.0 {
            let g = m.gain(d);
            assert!(g.is_finite() && g >= 0.0);
            assert!(g <= prev + 1e-6, "{curve:?} not monotonic at {d}");
            prev = g;
            d += 7.3;
        }
    }
    let bad = DistanceModel { reference_distance: f32::NAN, rolloff_factor: f32::INFINITY, ..DistanceModel::default() };
    assert!(bad.gain(5.0).is_finite());
}
