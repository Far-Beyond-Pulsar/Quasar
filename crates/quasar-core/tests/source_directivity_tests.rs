//! #74: source directivity pattern maths and geometry.

use quasar_core::source_directivity::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn omni_is_exactly_unity_everywhere() {
    for c in [-1.0, -0.3, 0.0, 0.9, 1.0, f32::NAN] {
        assert_eq!(pattern_band_gains(0.0, c).0, [1.0; 8]);
    }
    assert_eq!(pattern_band_gains(-3.0, -1.0).0, [1.0; 8], "negative directivity clamps to omni");
    assert_eq!(diffuse_send_gain(0.0), 1.0);
    assert_eq!(diffuse_field_power(0.0).0, [1.0; 8]);
}

#[test]
fn cardioid_at_one_kilohertz_and_rear_attenuation_formula() {
    let band_1k = 4;
    let max = pattern_band_gains(1.0, 1.0);
    assert_eq!(max.0, [1.0; 8], "on axis every band is 0 dB");
    let side = pattern_band_gains(1.0, 0.0);
    assert!(close(side.0[band_1k], 0.5, 1e-6), "cardioid side is -6 dB: {}", side.0[band_1k]);
    let rear = pattern_band_gains(1.0, -1.0);
    assert!(rear.0[band_1k] <= MIN_DIRECTIVITY_GAIN * 1.0001, "cardioid rear is a (floored) null");
    // Rear gain at 1 kHz is 1 - d for any d (sub-cardioid family).
    for d in [0.2_f32, 0.5, 0.8] {
        let r = pattern_band_gains(d, -1.0).0[band_1k];
        assert!(close(r, 1.0 - d, 1e-5), "d = {d}: rear {r}, expected {}", 1.0 - d);
    }
}

#[test]
fn gain_falls_monotonically_with_angle() {
    for d in [0.3_f32, 1.0] {
        let mut prev = pattern_band_gains(d, 1.0);
        for k in 1..=40 {
            let c = 1.0 - k as f32 * 0.05;
            let g = pattern_band_gains(d, c);
            for b in 0..8 {
                assert!(g.0[b] <= prev.0[b] + 1e-6, "band {b} must not rise away from the axis");
            }
            prev = g;
        }
    }
}

#[test]
fn high_bands_are_narrower_than_low_bands() {
    // At the side (90 degrees) and a mid rear the gain falls with frequency.
    for c in [0.0_f32, -0.5, -1.0] {
        let g = pattern_band_gains(0.7, c).0;
        for b in 1..8 {
            assert!(g[b] <= g[b - 1] + 1e-6, "cos {c}: band {b} ({}) louder than band {} ({})", g[b], b - 1, g[b - 1]);
        }
        assert!(g[0] > g[7] * 1.05, "cos {c}: the cone must differ between 62.5 Hz and 8 kHz");
    }
}

#[test]
fn diffuse_field_average_matches_the_analytic_integral() {
    // Band 4 has exponent 1: mean of p^2 over the sphere = a^2 + b^2 / 3 with p = a + b c.
    for d in [0.3_f32, 0.8, 1.0] {
        let (a, b) = (1.0 - d / 2.0, d / 2.0);
        let want = a * a + b * b / 3.0;
        let got = diffuse_field_power(d).0[4];
        assert!(close(got, want, 2e-3), "d = {d}: {got} vs {want}");
    }
    // More directional = less total radiated power = lower reverb send; always in (0, 1].
    let mut prev = 1.0f32;
    for k in 1..=10 {
        let g = diffuse_send_gain(k as f32 / 10.0);
        assert!(g < prev && g > 0.3, "diffuse send {g} at d = {}", k as f32 / 10.0);
        prev = g;
    }
}

#[test]
fn emission_angle_from_orientation() {
    let e = [1.0, 2.0, 3.0];
    // Listener straight ahead of the emitter's orientation (-Z).
    assert!(close(emission_cos(Some([0.0, 0.0, -2.0]), e, [1.0, 2.0, -7.0]).unwrap(), 1.0, 1e-6));
    // Rotated by 180 degrees.
    assert!(close(emission_cos(Some([0.0, 0.0, 5.0]), e, [1.0, 2.0, -7.0]).unwrap(), -1.0, 1e-6));
    // Side.
    assert!(close(emission_cos(Some([1.0, 0.0, 0.0]), e, [1.0, 2.0, -7.0]).unwrap(), 0.0, 1e-6));
    // Degenerate inputs read as omni.
    assert!(emission_cos(None, e, [0.0; 3]).is_none());
    assert!(emission_cos(Some([0.0; 3]), e, [0.0; 3]).is_none());
    assert!(emission_cos(Some([0.0, 0.0, 1.0]), e, e).is_none());
}

#[test]
fn bounce_point_is_recovered_from_arrival_direction_and_path_length() {
    // Wall at x = 3; emitter and listener on its left. Image-source construction:
    let e = [0.0_f32, 1.0, 0.0];
    let l = [-1.0, 1.5, -4.0];
    let image = [6.0_f32, 1.0, 0.0]; // e mirrored in x = 3
    let v = [image[0] - l[0], image[1] - l[1], image[2] - l[2]];
    let t = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(); // total path
    let s = (3.0 - l[0]) / v[0]; // L + s v is on the wall
    let p = [l[0] + s * v[0], l[1] + s * v[1], l[2] + s * v[2]];
    let dir = [p[0] - l[0], p[1] - l[1], p[2] - l[2]];
    let got = first_order_bounce_point(e, l, dir, t).expect("bounce point");
    for k in 0..3 {
        assert!(close(got[k], p[k], 2e-4), "axis {k}: {} vs {}", got[k], p[k]);
    }
    // Degenerate: path shorter than the direct distance.
    assert!(first_order_bounce_point(e, l, dir, 1.0).is_none());
    assert!(first_order_bounce_point(e, l, [0.0; 3], t).is_none());
}
