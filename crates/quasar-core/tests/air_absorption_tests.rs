//! ISO 9613-1 air absorption (#53).
//!
//! NOTE on references: these tests do not embed the standard's published table
//! to the digit (the author could not reproduce those numbers from memory with
//! confidence). They check (a) well-known physical properties and (b) broadly
//! known magnitudes, with generous tolerances, plus (c) the formulation's own
//! invariants. Replace the ranges with exact table values if the standard is at hand.

use quasar_core::air::{air_absorption_db_per_m, air_absorption_db_per_m_at, air_absorption_gain};
use quasar_core::bands::FREQ_BAND_CENTRES;

fn a(f: f32, t: f32, rh: f32) -> f32 {
    air_absorption_db_per_m(f, t, rh)
}

#[test]
fn zero_at_dc_and_tiny_at_low_frequencies() {
    assert_eq!(a(0.0, 20.0, 50.0), 0.0);
    // 62.5 Hz is well below 1 dB/km at room conditions.
    assert!(a(62.5, 20.0, 50.0) < 1.0e-3);
}

#[test]
fn magnitudes_at_20c_50rh_are_in_the_published_range() {
    // Published values for these conditions are of the order of: 1 kHz a few
    // dB/km, 4 kHz a few tens of dB/km, 8 kHz about a hundred dB/km.
    let k1 = a(1000.0, 20.0, 50.0) * 1000.0;
    let k4 = a(4000.0, 20.0, 50.0) * 1000.0;
    let k8 = a(8000.0, 20.0, 50.0) * 1000.0;
    assert!((2.0..10.0).contains(&k1), "1 kHz: {k1} dB/km");
    assert!((15.0..60.0).contains(&k4), "4 kHz: {k4} dB/km");
    assert!((50.0..200.0).contains(&k8), "8 kHz: {k8} dB/km");
}

#[test]
fn monotonic_with_frequency_across_the_audio_band() {
    for (t, rh) in [(-10.0, 30.0), (0.0, 70.0), (20.0, 10.0), (20.0, 50.0), (20.0, 100.0), (40.0, 90.0)] {
        let mut prev = 0.0_f32;
        for f in FREQ_BAND_CENTRES {
            let v = a(f, t, rh);
            assert!(v > prev, "T={t} RH={rh}: alpha({f}) = {v} not above {prev}");
            prev = v;
        }
    }
}

#[test]
fn hf_absorption_is_larger_in_dry_air_than_at_high_humidity_at_8k_and_20c() {
    // The classic curve: at 8 kHz / 20 C absorption falls as RH rises from 10 % to 50 %.
    assert!(a(8000.0, 20.0, 10.0) > a(8000.0, 20.0, 50.0));
}

#[test]
fn finite_everywhere_in_the_valid_range_including_the_old_nan_point() {
    let mut t = -20.0_f32;
    while t <= 50.0 {
        let mut rh = 0.0_f32;
        while rh <= 100.0 {
            for f in FREQ_BAND_CENTRES {
                let v = a(f, t, rh);
                assert!(v.is_finite() && v >= 0.0, "T={t} RH={rh} f={f}: {v}");
            }
            rh += 2.5;
        }
        t += 1.0;
    }
    // Out-of-range / NaN inputs are clamped, never poison the gains.
    for v in [a(1000.0, f32::NAN, 50.0), a(1000.0, 20.0, f32::NAN), a(1000.0, 500.0, 1000.0), a(f32::NAN, 20.0, 50.0)] {
        assert!(v.is_finite());
    }
}

#[test]
fn pressure_ratio_scales_the_relaxation_frequencies() {
    // Lower ambient pressure (altitude) lowers the relaxation frequencies;
    // it must stay finite and change the result.
    let sea = air_absorption_db_per_m_at(4000.0, 20.0, 50.0, 101.325);
    let high = air_absorption_db_per_m_at(4000.0, 20.0, 50.0, 70.0);
    assert!(high.is_finite() && (high - sea).abs() > 1e-5);
}

#[test]
fn gain_is_db_not_nepers() {
    let d = 100.0;
    let g = air_absorption_gain(d, 20.0, 50.0);
    for i in 0..8 {
        let alpha = a(FREQ_BAND_CENTRES[i], 20.0, 50.0);
        let expect = 10.0_f32.powf(-alpha * d / 20.0);
        assert!((g.0[i] - expect).abs() < 1e-6);
        assert!(g.0[i] > 0.0 && g.0[i] <= 1.0);
    }
    // 8 kHz over 100 m loses several dB, but not more than ~20.
    let loss_db = -20.0 * g.0[7].log10();
    assert!((3.0..20.0).contains(&loss_db), "8 kHz / 100 m: {loss_db} dB");
    // Zero distance = unity.
    assert!(air_absorption_gain(0.0, 20.0, 50.0).0.iter().all(|&x| x == 1.0));
}
