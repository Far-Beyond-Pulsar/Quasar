//! Constant-power VBAP panner tests (`quasar_dsp::vbap`).

use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::master_decoder::{
    layout_lfe, layout_panner, layout_positions, DecoderMode, MasterSpatialDecoderNode, SpeakerLayout,
};
use quasar_dsp::node_graph::AudioNode;
use quasar_dsp::vbap::VbapPanner;
use std::f32::consts::PI;

fn deg(d: f32) -> f32 {
    d.to_radians()
}

/// 7.1.4-like custom layout: ear-level ring of 7 + 4 height speakers.
fn custom_714_heights() -> (Vec<[f32; 3]>, Vec<usize>) {
    let dir = |az: f32, el: f32| {
        let (a, e) = (deg(az), deg(el));
        [a.sin() * e.cos(), e.sin(), -a.cos() * e.cos()]
    };
    let v = vec![
        dir(-30.0, 0.0),
        dir(30.0, 0.0),
        dir(0.0, 0.0),
        dir(-90.0, 0.0),
        dir(90.0, 0.0),
        dir(-150.0, 0.0),
        dir(150.0, 0.0),
        dir(-45.0, 45.0),
        dir(45.0, 45.0),
        dir(-135.0, 45.0),
        dir(135.0, 45.0),
    ];
    (v, vec![])
}

/// Demo-style 8-speaker custom layout with world-space (non-unit) positions.
fn demo_layout() -> (Vec<[f32; 3]>, Vec<usize>) {
    (
        vec![
            [-7.0, 5.5, -12.0],
            [7.0, 5.5, -12.0],
            [0.0, 3.0, -12.0],
            [0.0, 0.3, -7.0],
            [-7.0, 2.0, 12.0],
            [7.0, 2.0, 12.0],
            [-7.0, 0.5, -12.0],
            [7.0, 0.5, -12.0],
        ],
        vec![],
    )
}

fn named(l: SpeakerLayout) -> (Vec<[f32; 3]>, Vec<usize>) {
    (layout_positions(&l), layout_lfe(&l).to_vec())
}

fn all_layouts() -> Vec<(&'static str, Vec<[f32; 3]>, Vec<usize>)> {
    let mut v = Vec::new();
    for (n, l) in [
        ("stereo", SpeakerLayout::Stereo),
        ("quad", SpeakerLayout::Quad),
        ("5.1", SpeakerLayout::Surround51),
        ("7.1", SpeakerLayout::Surround714),
    ] {
        let (p, lfe) = named(l);
        v.push((n, p, lfe));
    }
    let (p, lfe) = custom_714_heights();
    v.push(("7.1.4", p, lfe));
    let (p, lfe) = demo_layout();
    v.push(("demo8", p, lfe));
    v
}

fn gains_of(p: &VbapPanner, az: f32, el: f32, n: usize) -> Vec<f32> {
    let mut g = vec![0.0; n];
    p.gains(az, el, &mut g);
    g
}

fn power(g: &[f32]) -> f32 {
    g.iter().map(|x| x * x).sum()
}

#[test]
fn constant_power_everywhere() {
    for (name, pos, lfe) in all_layouts() {
        let p = VbapPanner::new(&pos, &lfe);
        let mut az = -PI;
        while az <= PI {
            let mut el = -PI / 2.0;
            while el <= PI / 2.0 + 1e-6 {
                let g = gains_of(&p, az, el, pos.len());
                assert!(g.iter().all(|x| *x >= 0.0 && x.is_finite()), "{name}: bad gain at az={az} el={el}: {g:?}");
                assert!((power(&g) - 1.0).abs() < 1e-4, "{name}: power {} at az={az} el={el}", power(&g));
                el += deg(7.0);
            }
            az += deg(1.3);
        }
    }
}

#[test]
fn source_at_speaker_gives_unit_gain_there() {
    for (name, pos, lfe) in all_layouts() {
        let p = VbapPanner::new(&pos, &lfe);
        for (i, s) in pos.iter().enumerate() {
            if lfe.contains(&i) {
                continue;
            }
            let az = s[0].atan2(-s[2]);
            let el = s[1].atan2(s[0].hypot(s[2]));
            let g = gains_of(&p, az, el, pos.len());
            for (j, v) in g.iter().enumerate() {
                if j == i {
                    assert!(*v > 0.9995, "{name}: speaker {i} gain {v}");
                } else {
                    assert!(v.abs() < 2e-3, "{name}: speaker {i} leaks {v} into {j}");
                }
            }
        }
    }
}

#[test]
fn gains_are_continuous_over_az_and_el() {
    let step = deg(0.1);
    for (name, pos, lfe) in all_layouts() {
        let p = VbapPanner::new(&pos, &lfe);
        let n = pos.len();
        for &el in &[-80.0f32, -45.0, -20.0, 0.0, 20.0, 45.0, 80.0] {
            let mut prev = gains_of(&p, -PI, deg(el), n);
            let mut az = -PI + step;
            while az <= PI + 1e-6 {
                let g = gains_of(&p, az, deg(el), n);
                for k in 0..n {
                    assert!(
                        (g[k] - prev[k]).abs() < 0.05,
                        "{name}: az jump {} on slot {k} at az={az} el={el}",
                        (g[k] - prev[k]).abs()
                    );
                }
                prev = g;
                az += step;
            }
            // +-pi seam must match (same direction).
            let a = gains_of(&p, -PI, deg(el), n);
            let b = gains_of(&p, PI, deg(el), n);
            for k in 0..n {
                assert!((a[k] - b[k]).abs() < 1e-3, "{name}: wrap seam slot {k} el={el}");
            }
        }
        for &az in &[-170.0f32, -120.0, -90.0, -30.0, 0.0, 30.0, 119.9, 150.0, 180.0] {
            let mut prev = gains_of(&p, deg(az), -PI / 2.0, n);
            let mut el = -PI / 2.0 + step;
            while el <= PI / 2.0 {
                let g = gains_of(&p, deg(az), el, n);
                for k in 0..n {
                    assert!((g[k] - prev[k]).abs() < 0.05, "{name}: el jump on slot {k} at az={az} el={el}");
                }
                prev = g;
                el += step;
            }
        }
    }
}

#[test]
fn stereo_grazing_angle_has_no_cliff_and_center_is_constant_power() {
    let (pos, lfe) = named(SpeakerLayout::Stereo);
    let p = VbapPanner::new(&pos, &lfe);
    let a = gains_of(&p, deg(119.9), 0.0, 2);
    let b = gains_of(&p, deg(120.1), 0.0, 2);
    for k in 0..2 {
        assert!((a[k] - b[k]).abs() < 0.01, "cliff at 120 deg: {a:?} vs {b:?}");
    }
    let c = gains_of(&p, 0.0, 0.0, 2);
    assert!((power(&c) - 1.0).abs() < 1e-5, "centre power {}", power(&c));
    assert!((c[0] - c[1]).abs() < 1e-5);
}

#[test]
fn stereo_rear_is_audible_and_symmetric() {
    let (pos, lfe) = named(SpeakerLayout::Stereo);
    let p = VbapPanner::new(&pos, &lfe);
    for &d in &[100.0f32, 135.0, 170.0, 180.0] {
        let r = gains_of(&p, deg(d), 0.0, 2);
        let l = gains_of(&p, deg(-d), 0.0, 2);
        assert!((power(&r) - 1.0).abs() < 1e-4, "rear {d} power {}", power(&r));
        assert!((r[0] - l[1]).abs() < 1e-4 && (r[1] - l[0]).abs() < 1e-4, "asymmetric at {d}");
    }
    let r = gains_of(&p, PI, 0.0, 2);
    assert!((r[0] - r[1]).abs() < 1e-4);
}

#[test]
fn lfe_slot_is_always_silent() {
    for l in [SpeakerLayout::Surround51, SpeakerLayout::Surround714] {
        let (pos, lfe) = named(l);
        assert_eq!(lfe, vec![3]);
        let p = VbapPanner::new(&pos, &lfe);
        let mut az = -PI;
        while az <= PI {
            for &el in &[-1.0f32, 0.0, 0.7] {
                let g = gains_of(&p, az, el, pos.len());
                assert_eq!(g[3], 0.0, "LFE leaked at az={az} el={el}");
                assert!((power(&g) - 1.0).abs() < 1e-4);
            }
            az += deg(2.0);
        }
        // Straight below centre (the old LFE position) must not feed it either.
        assert_eq!(gains_of(&p, 0.0, -0.78, pos.len())[3], 0.0);
    }
}

#[test]
fn surround714_slot_order_is_standard() {
    let (pos, lfe) = named(SpeakerLayout::Surround714);
    let p = VbapPanner::new(&pos, &lfe);
    let peak = |g: &[f32]| g.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(peak(&gains_of(&p, deg(150.0), 0.0, 8)), 5, "+150 deg => BR");
    assert_eq!(peak(&gains_of(&p, deg(-150.0), 0.0, 8)), 4, "-150 deg => BL");
    assert_eq!(peak(&gains_of(&p, deg(90.0), 0.0, 8)), 7, "+90 deg => SR");
    assert_eq!(peak(&gains_of(&p, deg(-90.0), 0.0, 8)), 6, "-90 deg => SL");
    assert_eq!(peak(&gains_of(&p, 0.0, 0.0, 8)), 2, "front => C");
    let g = gains_of(&p, deg(150.0), 0.0, 8);
    assert!(g[5] > 0.999 && g[7] < 1e-3);
}

#[test]
fn mono_and_empty_layouts() {
    let p = VbapPanner::new(&[[0.0, 0.0, -1.0]], &[]);
    assert_eq!(gains_of(&p, 2.0, 0.3, 1), vec![1.0]);
    let p = VbapPanner::new(&[], &[]);
    let mut g: [f32; 0] = [];
    p.gains(0.0, 0.0, &mut g);
}

fn coeffs(az: f32, el: f32) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        direct_azimuth: az,
        direct_elevation: el,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        early_late_split_secs: 0.0,
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version: 0,
    }
}

/// The decoder node ramps gains per sample, so a pan change between blocks
/// produces no step at the block boundary (slope only).
#[test]
fn decoder_node_ramps_across_block_boundary() {
    const N: u16 = 256;
    let mut node = MasterSpatialDecoderNode::new(DecoderMode::Vbap { layout: SpeakerLayout::Stereo }, 48_000.0);
    let mut input = AudioBuffer::new(1, N);
    for i in 0..N {
        input.set(0, i, 1.0);
    }
    let mut stream: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for az in [-30.0f32, -30.0, 30.0, 30.0] {
        let mut out = AudioBuffer::new(2, N);
        node.process(&input, &mut out, &coeffs(deg(az), 0.0));
        for ch in 0..2u16 {
            stream[ch as usize].extend_from_slice(out.channel(ch));
        }
    }
    // First block is exactly the target (no ramp from garbage): left = 1.
    assert!((stream[0][0] - 1.0).abs() < 1e-4 && stream[1][0].abs() < 1e-4);
    let mut max_step = 0.0f32;
    for ch in 0..2 {
        for w in stream[ch].windows(2) {
            max_step = max_step.max((w[1] - w[0]).abs());
        }
    }
    assert!(max_step < 1.5 / N as f32, "ramp step {max_step} exceeds slope bound");
    // And the ramp did reach the new pan.
    assert!(stream[1][4 * N as usize - 1] > 0.99);
}

#[test]
fn layout_panner_matches_layout_outputs() {
    for l in [SpeakerLayout::Stereo, SpeakerLayout::Quad, SpeakerLayout::Surround51, SpeakerLayout::Surround714] {
        let n = MasterSpatialDecoderNode::output_channels_for_mode(&DecoderMode::Vbap { layout: l.clone() });
        assert_eq!(layout_panner(&l).num_outputs(), n as usize);
    }
}

// ── L3: planar layouts and elevation ─────────────────────────────────────

fn active_count(pos: &[[f32; 3]], lfe: &[usize]) -> usize {
    (0..pos.len()).filter(|i| !lfe.contains(i)).count()
}

#[test]
fn planar_elevation_is_ignored_inside_the_planar_band() {
    let (pos, lfe) = named(SpeakerLayout::Stereo);
    let p = VbapPanner::new(&pos, &lfe);
    let a = gains_of(&p, deg(20.0), 0.0, 2);
    let b = gains_of(&p, deg(20.0), deg(4.0), 2);
    for k in 0..2 {
        assert!((a[k] - b[k]).abs() < 1e-6, "gain moved inside the planar band");
    }
}

#[test]
fn planar_elevation_widens_the_image_and_converges_at_the_poles() {
    for l in [SpeakerLayout::Stereo, SpeakerLayout::Quad, SpeakerLayout::Surround51, SpeakerLayout::Surround714] {
        let (pos, lfe) = named(l);
        let m = active_count(&pos, &lfe) as f32;
        let p = VbapPanner::new(&pos, &lfe);
        let n = pos.len();
        for &pole in &[PI / 2.0, -PI / 2.0] {
            let reference = gains_of(&p, 0.0, pole, n);
            let mut az = -PI;
            while az <= PI {
                let g = gains_of(&p, az, pole, n);
                for k in 0..n {
                    assert!((g[k] - reference[k]).abs() < 1e-4, "pole gain depends on az (slot {k}, az {az})");
                    if !lfe.contains(&k) {
                        assert!((g[k] - 1.0 / m.sqrt()).abs() < 1e-4, "pole gain {} != 1/sqrt(M)", g[k]);
                    } else {
                        assert_eq!(g[k], 0.0);
                    }
                }
                az += deg(9.0);
            }
        }
        // The panned image flattens monotonically as the source rises: the loudest
        // speaker for a source at the first active speaker's azimuth falls with elevation.
        let first = (0..n).find(|i| !lfe.contains(i)).unwrap();
        let az0 = pos[first][0].atan2(-pos[first][2]);
        let mut last = f32::INFINITY;
        let mut el = 0.0;
        while el <= PI / 2.0 + 1e-6 {
            let g = gains_of(&p, az0, el, n);
            assert!((power(&g) - 1.0).abs() < 1e-4);
            assert!(g[first] <= last + 1e-6, "peak gain rose with elevation");
            last = g[first];
            el += deg(2.0);
        }
    }
}

// ── L4: layouts that do not surround the listener ────────────────────────

fn dir3(az: f32, el: f32) -> [f32; 3] {
    let (a, e) = (deg(az), deg(el));
    [a.sin() * e.cos(), e.sin(), -a.cos() * e.cos()]
}

/// 3D layout confined to a +-30 degree front sector (lower + upper row).
fn front_only_3d() -> Vec<[f32; 3]> {
    vec![dir3(-30.0, 0.0), dir3(30.0, 0.0), dir3(0.0, 0.0), dir3(-30.0, 35.0), dir3(30.0, 35.0), dir3(0.0, 35.0)]
}

/// Front half dome: ear-level front half ring, an upper row and a zenith speaker.
fn half_dome() -> Vec<[f32; 3]> {
    vec![
        dir3(-90.0, 0.0),
        dir3(-45.0, 0.0),
        dir3(0.0, 0.0),
        dir3(45.0, 0.0),
        dir3(90.0, 0.0),
        dir3(-60.0, 45.0),
        dir3(0.0, 45.0),
        dir3(60.0, 45.0),
        dir3(0.0, 90.0),
    ]
}

/// Largest per-step gain change over dense azimuth sweeps (several elevations,
/// incl. the poles' neighbourhood) and elevation sweeps (several azimuths).
fn max_sweep_step<F: Fn(f32, f32) -> Vec<f32>>(f: F) -> f32 {
    let step = deg(0.1);
    let mut worst = 0.0f32;
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    for &el in &[-85.0f32, -60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0, 85.0] {
        let mut prev = f(-PI, deg(el));
        let mut az = -PI + step;
        while az <= PI + 1e-6 {
            let g = f(az, deg(el));
            worst = worst.max(diff(&g, &prev));
            prev = g;
            az += step;
        }
    }
    for &az in &[-180.0f32, -150.0, -100.0, -60.0, -30.0, 0.0, 15.0, 45.0, 90.0, 135.0, 179.0] {
        let mut prev = f(deg(az), -PI / 2.0);
        let mut el = -PI / 2.0 + step;
        while el <= PI / 2.0 {
            let g = f(deg(az), el);
            worst = worst.max(diff(&g, &prev));
            prev = g;
            el += step;
        }
    }
    worst
}

#[test]
fn non_surrounding_3d_layouts_have_full_coverage() {
    for (name, pos) in [("front-only", front_only_3d()), ("half-dome", half_dome())] {
        let p = VbapPanner::new(&pos, &[]);
        let n = pos.len();
        let mut az = -PI;
        while az <= PI {
            let mut el = -PI / 2.0;
            while el <= PI / 2.0 + 1e-6 {
                let g = gains_of(&p, az, el, n);
                assert!(g.iter().all(|x| *x >= 0.0 && x.is_finite()), "{name}: bad gain {g:?} at az={az} el={el}");
                assert!((power(&g) - 1.0).abs() < 1e-4, "{name}: power {} at az={az} el={el}", power(&g));
                el += deg(5.0);
            }
            az += deg(3.0);
        }
        // Directly behind and below are in the formerly uncovered region: still audible.
        for &(a, e) in &[(180.0f32, 0.0f32), (180.0, 60.0), (0.0, -60.0), (120.0, -20.0)] {
            let g = gains_of(&p, deg(a), deg(e), n);
            assert!((power(&g) - 1.0).abs() < 1e-4, "{name}: silent at az={a} el={e}");
        }
        // Real speakers still win exactly where they sit.
        for (i, s) in pos.iter().enumerate() {
            let g = gains_of(&p, s[0].atan2(-s[2]), s[1].atan2(s[0].hypot(s[2])), n);
            assert!(g[i] > 0.9995, "{name}: speaker {i} gain {}", g[i]);
        }
    }
}

#[test]
fn non_surrounding_3d_layouts_are_continuous_over_dense_sweeps() {
    for (name, pos) in [("front-only", front_only_3d()), ("half-dome", half_dome())] {
        let p = VbapPanner::new(&pos, &[]);
        let n = pos.len();
        let worst = max_sweep_step(|az, el| gains_of(&p, az, el, n));
        assert!(worst < 0.05, "{name}: max gain step {worst} per 0.1 degree");
    }
}

/// Sanity check of the continuity metric itself: an artificially discontinuous
/// gain function (hard swap of two speakers past az = 0.3 rad) must be flagged
/// by the same sweep that passes the real panner.
#[test]
fn continuity_metric_detects_an_artificial_discontinuity() {
    let (pos, lfe) = named(SpeakerLayout::Stereo);
    let p = VbapPanner::new(&pos, &lfe);
    let good = max_sweep_step(|az, el| gains_of(&p, az, el, 2));
    let bad = max_sweep_step(|az, el| {
        let mut g = gains_of(&p, az, el, 2);
        if az > 0.3 && az < 2.0 {
            g.swap(0, 1);
        }
        g
    });
    assert!(good < 0.05, "real panner step {good}");
    assert!(bad > 0.2, "metric failed to flag the discontinuity (step {bad})");
}

// ---- #126: lateral sources on gap layouts go to the nearer speaker -----------

#[test]
fn stereo_lateral_sources_do_not_leak_into_the_far_speaker() {
    let (pos, lfe) = named(SpeakerLayout::Stereo);
    let p = VbapPanner::new(&pos, &lfe);
    for &d in &[90.0f32, 100.0] {
        let r = gains_of(&p, deg(d), 0.0, 2); // hard right: left must be ~0
        let l = gains_of(&p, deg(-d), 0.0, 2);
        assert!(r[0] < 0.05 && r[1] > 0.99, "+{d}: {r:?}");
        assert!(l[1] < 0.05 && l[0] > 0.99, "-{d}: {l:?}");
        assert!((power(&r) - 1.0).abs() < 1e-5);
    }
    // The rear still plays on both speakers and the far gain rises monotonically toward it.
    let rear = gains_of(&p, PI, 0.0, 2);
    assert!(rear[0] > 0.7 && rear[0] < 0.72 && (rear[0] - rear[1]).abs() < 1e-5, "{rear:?}");
    let mut prev = 0.0f32;
    for d in (90..=180).step_by(5) {
        let g = gains_of(&p, deg(d as f32), 0.0, 2)[0];
        assert!(g >= prev - 1e-6, "far gain not monotonic at {d}");
        prev = g;
    }
}

#[test]
fn stereo_gap_crossfade_is_continuous_and_constant_power_at_fine_steps() {
    let (pos, lfe) = named(SpeakerLayout::Stereo);
    let p = VbapPanner::new(&pos, &lfe);
    let step = deg(0.1);
    let mut prev = gains_of(&p, -PI, 0.0, 2);
    let mut az = -PI + step;
    while az <= PI {
        let g = gains_of(&p, az, 0.0, 2);
        assert!((power(&g) - 1.0).abs() < 1e-4);
        assert!((g[0] - prev[0]).abs() < 0.05 && (g[1] - prev[1]).abs() < 0.05, "step at {az}");
        prev = g;
        az += step;
    }
}

#[test]
fn front_only_planar_layouts_with_wide_gaps_stay_well_behaved() {
    // Front-only planar layouts: a >= 180 degree rear gap between the outermost speakers.
    let layouts: [&[f32]; 3] = [&[-30.0, 0.0, 30.0], &[-90.0, 90.0], &[-60.0, -20.0, 20.0, 60.0]];
    for az_list in layouts {
        let pos: Vec<[f32; 3]> =
            az_list.iter().map(|&a| [deg(a).sin(), 0.0, -deg(a).cos()]).collect();
        let p = VbapPanner::new(&pos, &[]);
        let n = pos.len();
        let mut prev = gains_of(&p, -PI, 0.0, n);
        let mut az = -PI;
        while az <= PI {
            let g = gains_of(&p, az, 0.0, n);
            assert!((power(&g) - 1.0).abs() < 1e-4, "{az_list:?} power at {az}");
            for k in 0..n {
                assert!((g[k] - prev[k]).abs() < 0.05, "{az_list:?} step at {az}");
            }
            prev = g;
            az += deg(0.1);
        }
        // Each speaker direction is still reproduced exactly by that speaker.
        for (i, &a) in az_list.iter().enumerate() {
            let g = gains_of(&p, deg(a), 0.0, n);
            assert!((g[i] - 1.0).abs() < 1e-4, "{az_list:?} speaker {i}: {g:?}");
        }
        // Rear is audible and mirror symmetric for symmetric layouts.
        let (r, l) = (gains_of(&p, deg(170.0), 0.0, n), gains_of(&p, deg(-170.0), 0.0, n));
        assert!(power(&r) > 0.99 && power(&l) > 0.99);
        for k in 0..n {
            assert!((r[k] - l[n - 1 - k]).abs() < 1e-4);
        }
    }
}
