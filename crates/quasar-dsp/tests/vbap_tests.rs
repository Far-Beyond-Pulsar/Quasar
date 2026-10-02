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
