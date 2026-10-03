//! Emitter model (#156): spherical and directional speaker patterns and source shapes.

use quasar_core::bands::Band8;
use quasar_core::emitter_pattern::{
    EmitterModel, EmitterPattern, EmitterShape, EmitterTrace, HORN_COVERAGE_WIDENING, LEGACY_RADIUS,
};
use quasar_core::source_directivity::{diffuse_send_gain, emission_cos, pattern_band_gains};

const FWD: [f32; 3] = [0.0, 0.0, -1.0];

fn dir_at(az_deg: f32, el_deg: f32) -> [f32; 3] {
    // Forward is -Z, right +X, up +Y.
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    [az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()]
}

fn db(g: f32) -> f32 {
    20.0 * g.log10()
}

#[test]
fn omni_is_exactly_one_everywhere() {
    let p = EmitterPattern::Omni;
    assert!(p.is_omni());
    for az in (-180..=180).step_by(15) {
        for el in (-90..=90).step_by(15) {
            assert_eq!(p.band_gains(FWD, dir_at(az as f32, el as f32)), Band8::splat(1.0));
        }
    }
    assert_eq!(p.diffuse_send_gain(), 1.0);
    // The cardioid family with directivity 0 is omni too.
    assert!(EmitterPattern::CardioidFamily { directivity: 0.0 }.is_omni());
}

#[test]
fn cardioid_family_is_bit_identical_to_the_legacy_pattern() {
    for d in [0.3_f32, 0.7, 1.0] {
        let p = EmitterPattern::CardioidFamily { directivity: d };
        for az in (-180..=180).step_by(10) {
            let dir = dir_at(az as f32, 20.0);
            // The engine computes the cosine with `emission_cos`; the pattern must agree with it.
            let cos = emission_cos(Some(FWD), [0.0; 3], dir).unwrap();
            assert_eq!(p.band_gains(FWD, dir).0, pattern_band_gains(d, cos).0, "d={d} az={az}");
        }
        assert_eq!(p.diffuse_send_gain(), diffuse_send_gain(d));
    }
}

#[test]
fn degenerate_input_reads_as_on_axis() {
    let p = EmitterPattern::horn(60.0, 40.0);
    assert_eq!(p.band_gains([0.0; 3], dir_at(90.0, 0.0)), Band8::splat(1.0));
    assert_eq!(p.band_gains(FWD, [0.0; 3]), Band8::splat(1.0));
    assert_eq!(p.band_gains(FWD, [f32::NAN, 0.0, 0.0]), Band8::splat(1.0));
}

#[test]
fn horn_is_minus_6_db_at_its_coverage_edges_at_and_above_1khz() {
    // 60 deg horizontal x 40 deg vertical (full -6 dB angles).
    let p = EmitterPattern::horn(60.0, 40.0);
    // On axis: 0 dB in every band.
    for b in 0..8 {
        assert!((p.band_gains(FWD, FWD).0[b] - 1.0).abs() < 1e-6);
    }
    let h_edge = p.band_gains(FWD, dir_at(30.0, 0.0));
    let v_edge = p.band_gains(FWD, dir_at(0.0, 20.0));
    for b in 4..8 {
        assert!((db(h_edge.0[b]) + 6.0).abs() < 0.05, "H edge band {b}: {} dB", db(h_edge.0[b]));
        assert!((db(v_edge.0[b]) + 6.0).abs() < 0.05, "V edge band {b}: {} dB", db(v_edge.0[b]));
    }
    // Symmetric left/right and up/down.
    let l = p.band_gains(FWD, dir_at(-30.0, 0.0));
    let d = p.band_gains(FWD, dir_at(0.0, -20.0));
    for b in 0..8 {
        assert!((l.0[b] - h_edge.0[b]).abs() < 1e-5);
        assert!((d.0[b] - v_edge.0[b]).abs() < 1e-5);
    }
}

#[test]
fn horn_rear_is_at_least_20_db_down_at_1khz_and_above_and_monotone_in_angle() {
    let p = EmitterPattern::horn(60.0, 40.0);
    for az in [150.0_f32, 180.0] {
        let g = p.band_gains(FWD, dir_at(az, 0.0));
        for b in 4..8 {
            assert!(db(g.0[b]) <= -20.0, "az {az} band {b}: {} dB", db(g.0[b]));
        }
    }
    // Monotone non-increasing with the off-axis angle in every band.
    for b in 0..8 {
        let mut prev = f32::MAX;
        for az in 0..=180 {
            let g = p.band_gains(FWD, dir_at(az as f32, 0.0)).0[b];
            assert!(g <= prev + 1e-6, "band {b} az {az}");
            prev = g;
        }
    }
}

#[test]
fn horn_coverage_widens_toward_low_frequencies() {
    let p = EmitterPattern::horn(60.0, 40.0);
    let g = p.band_gains(FWD, dir_at(30.0, 0.0));
    for b in 0..7 {
        assert!(g.0[b] >= g.0[b + 1] - 1e-6, "band {b} must be at least as wide as band {}", b + 1);
    }
    // The -6 dB point of band 0 sits at HORN_COVERAGE_WIDENING[0] x the nominal coverage.
    let wide = p.band_gains(FWD, dir_at(30.0 * HORN_COVERAGE_WIDENING[0], 0.0));
    assert!((db(wide.0[0]) + 6.0).abs() < 0.1, "{} dB", db(wide.0[0]));
}

#[test]
fn horn_vertical_axis_follows_the_up_vector() {
    // Same horn rotated 90 deg about the forward axis: vertical becomes horizontal.
    let tall = EmitterPattern::Horn { h_deg: 60.0, v_deg: 20.0, rear_db: -30.0, up: [0.0, 1.0, 0.0] };
    let rolled = EmitterPattern::Horn { h_deg: 60.0, v_deg: 20.0, rear_db: -30.0, up: [1.0, 0.0, 0.0] };
    let a = tall.band_gains(FWD, dir_at(30.0, 0.0));
    let b = rolled.band_gains(FWD, dir_at(0.0, 30.0));
    for i in 0..8 {
        assert!((a.0[i] - b.0[i]).abs() < 1e-5);
    }
    // Up parallel to forward must not blow up.
    let degenerate = EmitterPattern::Horn { h_deg: 60.0, v_deg: 40.0, rear_db: -30.0, up: FWD };
    let g = degenerate.band_gains(FWD, dir_at(20.0, 5.0));
    assert!(g.0.iter().all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0));
}

#[test]
fn sound_cone_reproduces_inner_outer_and_outer_gain_exactly() {
    let p = EmitterPattern::SoundCone { inner_deg: 60.0, outer_deg: 120.0, outer_gain_db: -20.0 };
    let at = |az: f32| p.band_gains(FWD, dir_at(az, 0.0)).0[3];
    assert_eq!(at(0.0), 1.0);
    assert!((at(30.0) - 1.0).abs() < 1e-5); // inner half-angle (float rounding at the boundary)
    assert!((db(at(60.0)) + 20.0).abs() < 1e-3); // outer half-angle
    assert!((db(at(90.0)) + 20.0).abs() < 1e-3);
    assert!((db(at(180.0)) + 20.0).abs() < 1e-3);
    // Halfway between inner and outer half-angles: half of the outer gain in dB.
    assert!((db(at(45.0)) + 10.0).abs() < 1e-3);
    // Frequency independent.
    let g = p.band_gains(FWD, dir_at(45.0, 0.0));
    assert!(g.0.iter().all(|v| (*v - g.0[0]).abs() < 1e-7));
}

#[test]
fn hyper_and_supercardioid_have_the_textbook_rear_lobes_and_nulls() {
    let sup = EmitterPattern::Supercardioid;
    let hyp = EmitterPattern::Hypercardioid;
    // At 1 kHz (band 4, exponent 1): |a + (1-a) cos|.
    assert!((sup.band_gains(FWD, dir_at(180.0, 0.0)).0[4] - 0.26).abs() < 1e-3);
    assert!((hyp.band_gains(FWD, dir_at(180.0, 0.0)).0[4] - 0.50).abs() < 1e-3);
    // Null near 125.3 deg (super) and 109.5 deg (hyper): well below -30 dB there.
    assert!(db(sup.band_gains(FWD, dir_at(125.3, 0.0)).0[4]) < -30.0);
    assert!(db(hyp.band_gains(FWD, dir_at(109.5, 0.0)).0[4]) < -30.0);
    // Narrower than the cardioid at the side (90 deg): 0.37 vs 0.5.
    assert!(sup.band_gains(FWD, dir_at(90.0, 0.0)).0[4] < 0.5);
    assert!(hyp.band_gains(FWD, dir_at(90.0, 0.0)).0[4] < sup.band_gains(FWD, dir_at(90.0, 0.0)).0[4]);
}

#[test]
fn diffuse_send_gain_orders_the_patterns_by_how_narrow_they_are() {
    let omni = EmitterPattern::Omni.diffuse_send_gain();
    let cardioid = EmitterPattern::CardioidFamily { directivity: 1.0 }.diffuse_send_gain();
    let horn = EmitterPattern::horn(60.0, 40.0).diffuse_send_gain();
    let wide_horn = EmitterPattern::horn(120.0, 90.0).diffuse_send_gain();
    let cone = EmitterPattern::SoundCone { inner_deg: 60.0, outer_deg: 120.0, outer_gain_db: -20.0 }
        .diffuse_send_gain();
    assert_eq!(omni, 1.0);
    assert!(cardioid < omni);
    assert!(horn < wide_horn && wide_horn < omni, "horn {horn} wide {wide_horn}");
    assert!(horn < cardioid, "a narrow horn radiates less total power than a cardioid");
    assert!(cone > 0.0 && cone < 1.0);
    // Independent of the up vector (frame invariant).
    let rolled = EmitterPattern::Horn { h_deg: 60.0, v_deg: 40.0, rear_db: -30.0, up: [1.0, 0.0, 0.0] };
    assert!((rolled.diffuse_send_gain() - horn).abs() < 1e-3);
}

#[test]
fn emitter_model_caches_the_diffuse_gain() {
    let m = EmitterModel::new(Some(EmitterPattern::horn(60.0, 40.0)), EmitterShape::Default);
    assert!((m.diffuse_gain - EmitterPattern::horn(60.0, 40.0).diffuse_send_gain()).abs() < 1e-7);
    let d = EmitterModel::default();
    assert_eq!((d.diffuse_gain, d.pattern.is_none()), (1.0, true));
}

#[test]
fn emitter_trace_is_all_ones_without_a_pattern() {
    let t = EmitterTrace { pattern: None, forward: FWD, shape: EmitterShape::Default };
    assert_eq!(t.band_gains(dir_at(120.0, 10.0)), Band8::splat(1.0));
    let t = EmitterTrace { pattern: Some(EmitterPattern::horn(60.0, 40.0)), forward: FWD, shape: EmitterShape::Default };
    assert!(t.band_gains(dir_at(180.0, 0.0)).0[5] < 0.1);
}

fn frame() -> ([f32; 3], [f32; 3], [f32; 3]) {
    // Emitter looks at a listener along +Z: probe plane spanned by X and Y.
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
}

#[test]
fn shapes_place_probes_inside_their_aperture() {
    let (u, w, view) = frame();
    let n = 12;
    // Default = the legacy 0.35 m disc.
    for i in 0..n {
        let o = EmitterShape::Default.probe_offset(i, n, u, w, view);
        let r = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
        assert!(r <= LEGACY_RADIUS + 1e-6 && o[2].abs() < 1e-7);
    }
    // Point: everything on the centre.
    assert_eq!(EmitterShape::Point.probe_offset(5, n, u, w, view), [0.0; 3]);
    // Disc of 2 m: within the radius, in the facing plane, and it reaches the rim region.
    let mut max_r = 0.0_f32;
    for i in 0..n {
        let o = EmitterShape::Disc { radius: 2.0 }.probe_offset(i, n, u, w, view);
        let r = (o[0] * o[0] + o[1] * o[1]).sqrt();
        assert!(r <= 2.0 + 1e-6 && o[2].abs() < 1e-6);
        max_r = max_r.max(r);
    }
    assert!(max_r > 1.7, "disc probes should span the aperture, max r {max_r}");
    // Sphere: inside the ball, and with depth (z) variation.
    let mut zs = Vec::new();
    for i in 0..n {
        let o = EmitterShape::Sphere { radius: 1.0 }.probe_offset(i, n, u, w, view);
        assert!((o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() <= 1.0 + 1e-6);
        zs.push(o[2]);
    }
    assert!(zs.iter().cloned().fold(f32::MIN, f32::max) > 0.3 && zs.iter().cloned().fold(f32::MAX, f32::min) < -0.3);
    // Line: collinear along the axis, symmetric, spanning +-half_length.
    let line = EmitterShape::Line { half_length: 3.0, axis: [0.0, 2.0, 0.0] };
    let mut ts = Vec::new();
    for i in 0..n {
        let o = line.probe_offset(i, n, u, w, view);
        assert!(o[0].abs() < 1e-6 && o[2].abs() < 1e-6);
        ts.push(o[1]);
    }
    assert!(ts.windows(2).all(|p| p[1] > p[0]));
    assert!((ts[0] + ts[n - 1]).abs() < 1e-5 && ts[n - 1] > 2.4 && ts[n - 1] <= 3.0);
}
