//! Crossfader unit-convergence tests (P1 bug fix).
//!
//! `advance()` is called once per audio block but must advance the fade by
//! `block_size` frames, so a 15 ms fade converges after ~⌈fade_frames/block⌉
//! blocks — NOT after `fade_frames` single-sample steps (~3.8 s of real time).

use quasar_core::bands::Band8;
use quasar_core::param_exchange::{EarlyReflectionCoeffs, SpatialCoefficients};
use quasar_dsp::crossfader::EqualPowerCrossfader;

fn coeffs(source_id: u32, gain: f32) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id,
        direct_gain: Band8::splat(gain),
        direct_delay_samples: 0.0,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        early_late_split_secs: 0.0,
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version: 0,
    }
}

// ── crossfade_converges_in_blocks_not_samples ────────────────────────

#[test]
fn crossfade_converges_in_blocks_not_samples() {
    let initial = coeffs(0, 0.0);
    let mut xfader = EqualPowerCrossfader::new(15.0, 48000.0, initial);
    let target = coeffs(0, 0.9);
    xfader.set_target(&target);

    // 15 ms @ 48 kHz = 720 frames. With 512-frame blocks the fade completes
    // after ⌈720/512⌉ = 2 calls. 10 calls is far fewer than the 720 that the
    // old per-sample `advance()` would have needed (~3.8 s of audio).
    for _ in 0..10 {
        xfader.advance(512);
    }

    assert!(xfader.is_complete(), "fade must complete within a few blocks");
    let cur = xfader.current_coefficients();
    assert!(
        (cur.direct_gain.0[0] - 0.9).abs() < 1e-4,
        "band 0 should reach target, got {}",
        cur.direct_gain.0[0]
    );
    assert!(
        (cur.direct_gain.0[7] - 0.9).abs() < 1e-4,
        "band 7 should reach target, got {}",
        cur.direct_gain.0[7]
    );
    assert!((cur.late_gain_db - -10.0).abs() < 1e-4);
}

// ── crossfade_completes_after_ceiling_blocks ─────────────────────────

#[test]
fn crossfade_completes_after_ceiling_blocks() {
    let initial = coeffs(1, 1.0);
    let mut xfader = EqualPowerCrossfader::new(15.0, 48000.0, initial);
    let target = coeffs(1, 0.25);
    xfader.set_target(&target);

    // After 1 block the fade is mid-flight…
    xfader.advance(512);
    assert!(!xfader.is_complete());

    // …and after the 2nd block it has converged (720 ≤ 2·512).
    xfader.advance(512);
    assert!(xfader.is_complete());
    let cur = xfader.current_coefficients();
    assert!((cur.direct_gain.0[0] - 0.25).abs() < 1e-4);
}

// ── crossfade_stays_converged ────────────────────────────────────────

#[test]
fn crossfade_stays_converged() {
    let initial = coeffs(0, 0.0);
    let mut xfader = EqualPowerCrossfader::new(15.0, 48000.0, initial);
    xfader.set_target(&coeffs(0, 0.5));

    for _ in 0..20 {
        xfader.advance(256);
    }
    assert!(xfader.is_complete());
    let cur = xfader.current_coefficients();
    // Extra blocks after convergence must not drift the coefficients.
    assert!((cur.direct_gain.0[0] - 0.5).abs() < 1e-4);
    assert_eq!(xfader.blend_factor(), 1.0);
}

// ── crossfade_partial_block_still_converges ──────────────────────────

#[test]
fn crossfade_partial_block_still_converges() {
    // A fade shorter than one block must still converge to the target.
    let initial = coeffs(2, 0.0);
    let mut xfader = EqualPowerCrossfader::new(2.0, 48000.0, initial); // 96 frames
    xfader.set_target(&coeffs(2, 1.0));

    xfader.advance(512); // overshoots the 96-frame fade in one block
    assert!(xfader.is_complete());
    let cur = xfader.current_coefficients();
    assert!((cur.direct_gain.0[0] - 1.0).abs() < 1e-4);
}

// ── Smoothing-path fixes (B1–B4, G2) ─────────────────────────────────

fn refl(delay: f32, az: f32, g: f32) -> EarlyReflectionCoeffs {
    EarlyReflectionCoeffs {
        azimuth: az,
        elevation: 0.0,
        delay_samples: delay,
        gain: Band8::splat(g),
    }
}

#[test]
fn constant_parameter_stays_exactly_constant() {
    // B1: the old cos/sin blend of parameters overshot a constant value.
    let mut xfader = EqualPowerCrossfader::new(15.0, 48000.0, coeffs(0, 0.7));
    xfader.set_target(&coeffs(0, 0.7));
    for _ in 0..8 {
        xfader.advance(100);
        let cur = xfader.current_coefficients();
        assert_eq!(cur.direct_gain.0[0], 0.7);
        assert_eq!(cur.late_gain_db, -10.0);
        assert_eq!(cur.late_t60.0[3], 0.5);
    }
}

#[test]
fn fade_is_linear_and_monotonic_and_ends_exactly_on_target() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 0.0)); // 480 frames
    xfader.set_target(&coeffs(0, 0.9));
    xfader.advance(240);
    assert!((xfader.current_coefficients().direct_gain.0[0] - 0.45).abs() < 1e-6);
    let mut prev = 0.45;
    while !xfader.is_complete() {
        xfader.advance(60);
        let g = xfader.current_coefficients().direct_gain.0[0];
        assert!(g >= prev && g <= 0.9, "monotonic, no overshoot: {g}");
        prev = g;
    }
    assert_eq!(xfader.current_coefficients().direct_gain.0[0], 0.9);
}

#[test]
fn azimuth_takes_shortest_arc_across_wrap() {
    let mut a = coeffs(0, 1.0);
    a.direct_azimuth = 3.1;
    let mut b = coeffs(0, 1.0);
    b.direct_azimuth = -3.1;
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, a);
    xfader.set_target(&b);
    let mut saw_wrap_region = false;
    while !xfader.is_complete() {
        xfader.advance(16);
        let az = xfader.current_coefficients().direct_azimuth;
        assert!(az.abs() > 3.0, "azimuth passed near 0: {az}");
        assert!(az.abs() <= std::f32::consts::PI + 1e-6);
        if az.abs() > 3.14 {
            saw_wrap_region = true;
        }
    }
    assert!(saw_wrap_region, "should pass through ±pi");
    assert_eq!(xfader.current_coefficients().direct_azimuth, -3.1);
}

#[test]
fn reflection_azimuth_takes_shortest_arc() {
    let mut a = coeffs(0, 1.0);
    a.early_reflections = vec![refl(100.0, 3.1, 0.5)];
    let mut b = coeffs(0, 1.0);
    b.early_reflections = vec![refl(100.0, -3.1, 0.5)];
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, a);
    xfader.set_target(&b);
    xfader.advance(240);
    let az = xfader.current_coefficients().early_reflections[0].azimuth;
    assert!(az.abs() > 3.0, "reflection azimuth passed near 0: {az}");
}

#[test]
fn reflections_from_empty_take_target_and_fade_in() {
    // B3: a pair starting with 0 reflections must end up with the target's.
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 1.0));
    let mut t = coeffs(0, 1.0);
    t.early_reflections = vec![refl(100.0, 0.2, 0.8), refl(250.0, -0.4, 0.6)];
    xfader.set_target(&t);
    assert_eq!(xfader.current_coefficients().early_reflections.len(), 2);
    assert_eq!(xfader.current_coefficients().early_reflections[0].gain.0[0], 0.0);
    xfader.advance(240);
    let r = &xfader.current_coefficients().early_reflections[0];
    assert!((r.gain.0[0] - 0.4).abs() < 1e-6, "fade-in gain, got {}", r.gain.0[0]);
    assert_eq!(r.delay_samples, 100.0, "delay held while fading in");
    xfader.advance(512);
    let cur = xfader.current_coefficients();
    assert_eq!(cur.early_reflections.len(), 2);
    assert_eq!(cur.early_reflections[1].gain.0[0], 0.6);
    assert_eq!(cur.early_reflections[1].delay_samples, 250.0);
}

#[test]
fn reflections_shrink_to_target_and_fade_out() {
    let mut a = coeffs(0, 1.0);
    a.early_reflections = vec![refl(100.0, 0.0, 0.8), refl(300.0, 0.0, 0.6), refl(500.0, 0.0, 0.4)];
    let mut b = coeffs(0, 1.0);
    b.early_reflections = vec![refl(310.0, 0.0, 0.6)];
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, a);
    xfader.set_target(&b);
    xfader.advance(240);
    let cur = xfader.current_coefficients();
    // Matched by nearest delay (300 -> 310), not by index (100 -> 310).
    assert_eq!(cur.early_reflections.len(), 3);
    assert!((cur.early_reflections[0].delay_samples - 305.0).abs() < 1e-3);
    assert!((cur.early_reflections[0].gain.0[0] - 0.6).abs() < 1e-6);
    // Vanishing ones fade toward zero with their delay held.
    assert_eq!(cur.early_reflections[1].delay_samples, 100.0);
    assert!((cur.early_reflections[1].gain.0[0] - 0.4).abs() < 1e-6);
    xfader.advance(512);
    let cur = xfader.current_coefficients();
    assert_eq!(cur.early_reflections.len(), 1);
    assert_eq!(cur.early_reflections[0].delay_samples, 310.0);
}

#[test]
fn reflections_beyond_capacity_are_truncated_without_realloc() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 1.0));
    let mut t = coeffs(0, 1.0);
    t.early_reflections = (0..100).map(|i| refl(i as f32, 0.0, 0.5)).collect();
    xfader.set_target(&t);
    let cap = xfader.current_coefficients().early_reflections.capacity();
    xfader.advance(10_000);
    let cur = xfader.current_coefficients();
    assert_eq!(cur.early_reflections.len(), quasar_dsp::crossfader::MAX_CROSSFADE_REFLECTIONS);
    assert_eq!(cur.early_reflections.capacity(), cap);
}

#[test]
fn retarget_mid_fade_is_continuous() {
    // G2: set_target mid-fade snapshots the in-flight value as the new start.
    let mut a = coeffs(0, 0.0);
    a.direct_azimuth = 0.5;
    let mut b = coeffs(0, 1.0);
    b.direct_azimuth = 1.5;
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, a);
    xfader.set_target(&b);
    xfader.advance(200);
    let before = xfader.current_coefficients().clone();

    let mut c = coeffs(0, 0.2);
    c.direct_azimuth = -1.0;
    xfader.set_target(&c);
    let after = xfader.current_coefficients();
    assert_eq!(before.direct_gain.0[0], after.direct_gain.0[0]);
    assert_eq!(before.direct_azimuth, after.direct_azimuth);

    // First step after retarget moves by at most one block's worth of slope.
    xfader.advance(1);
    let g = xfader.current_coefficients().direct_gain.0[0];
    assert!((g - before.direct_gain.0[0]).abs() < 0.01, "jump: {} -> {}", before.direct_gain.0[0], g);
}

#[test]
fn source_id_change_snaps() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 0.0));
    let mut t = coeffs(5, 0.9);
    t.early_reflections = vec![refl(10.0, 0.0, 0.3)];
    xfader.set_target(&t);
    assert!(xfader.is_complete());
    let cur = xfader.current_coefficients();
    assert_eq!(cur.source_id, 5);
    assert_eq!(cur.direct_gain.0[0], 0.9);
    assert_eq!(cur.early_reflections.len(), 1);
}

// ── snap_to_ref (#119) ───────────────────────────────────────────────

#[test]
fn snap_to_ref_jumps_to_the_target_by_reference_without_allocating() {
    let mut a = coeffs(0, 1.0);
    let mut x = EqualPowerCrossfader::new(15.0, 48_000.0, a.clone());
    a.direct_delay_samples = 777.0;
    a.direct_gain = Band8::splat(0.25);
    a.early_reflections = vec![refl(100.0, 0.3, 0.5)];
    let cap = x.current_coefficients().early_reflections.capacity();
    x.snap_to_ref(&a);
    let c = x.current_coefficients();
    assert_eq!(c.direct_delay_samples, 777.0);
    assert_eq!(c.direct_gain, Band8::splat(0.25));
    assert_eq!(c.early_reflections.len(), 1);
    assert_eq!(c.early_reflections.capacity(), cap, "no reallocation");
    assert!(x.is_complete(), "snapped: nothing left to fade");
}
