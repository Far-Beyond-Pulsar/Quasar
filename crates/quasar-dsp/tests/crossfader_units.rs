//! Crossfader unit-convergence tests (P1 bug fix).
//!
//! `advance()` is called once per audio block but must advance the fade by
//! `block_size` frames, so a 15 ms fade converges after ~⌈fade_frames/block⌉
//! blocks — NOT after `fade_frames` single-sample steps (~3.8 s of real time).

use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
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
        version: 0,
    }
}

// ── crossfade_converges_in_blocks_not_samples ────────────────────────

#[test]
fn crossfade_converges_in_blocks_not_samples() {
    let initial = coeffs(0, 0.0);
    let mut xfader = EqualPowerCrossfader::new(15.0, 48000.0, initial);
    let target = coeffs(0, 0.9);
    xfader.set_target(target);

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
    xfader.set_target(target);

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
    xfader.set_target(coeffs(0, 0.5));

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
    xfader.set_target(coeffs(2, 1.0));

    xfader.advance(512); // overshoots the 96-frame fade in one block
    assert!(xfader.is_complete());
    let cur = xfader.current_coefficients();
    assert!((cur.direct_gain.0[0] - 1.0).abs() < 1e-4);
}
