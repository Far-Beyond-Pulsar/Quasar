//! #77: the direct chain (propagation delay, band EQ / occlusion, distance gain) is computed per
//! (listener, emitter) pair, so every listener gets its own distance, occlusion and filtering.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const C: f32 = 343.0;
const BLOCK: usize = 256;
const SETTLE: usize = 12;
const BLOCKS: usize = 40;

/// Emitter at the origin; one stereo listener per entry of `listeners` (all facing the emitter
/// along the z axis). `panel`: a panel across the line to a listener at z = -4 that passes lows
/// and blocks highs.
fn build(listeners: &[[f32; 3]], cpu_with_panel: bool) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    if cpu_with_panel {
        e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
        let panel_mat = e.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(
                Band8::splat(0.3),
                Band8::zeros(),
                // Transmission: lows pass, highs are blocked (a thick, soft panel).
                Band8::new([0.9, 0.85, 0.7, 0.4, 0.2, 0.08, 0.03, 0.01]),
            ),
        ));
        let mut scene = AcousticScene::new();
        // 4 m x 4 m panel across the line emitter -> (0, 0, -4), at z = -2.
        scene.add_mesh(AcousticMesh::new(
            1,
            vec![[-2.0, -2.0, -2.0], [2.0, -2.0, -2.0], [2.0, 2.0, -2.0], [-2.0, 2.0, -2.0]],
            vec![0, 1, 2, 0, 2, 3],
            panel_mat,
        ));
        let cfg = CpuSimdConfig { max_reflection_order: 0, ..CpuSimdConfig::default() };
        e.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
    } else {
        e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    }
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    e.debug_audio_stage = 2; // direct chain only
    let src = e.load_source(SourceConfig { path: "imp.wav".into(), channels: 1 }).expect("source");
    let out = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    for p in listeners {
        // Face the emitter (centre pan for everyone).
        let to_emitter = [-p[0], -p[1], -p[2]];
        e.add_listener(ListenerConfig {
            position: *p,
            heading: to_emitter,
            physical_layout: PhysicalOutputLayout::Stereo,
        });
    }
    e.update_scene_spatial();
    e
}

/// Render an impulse; returns the mono (L+R) response per listener over `BLOCKS` blocks.
fn impulse_responses(e: &mut SpatialAudioEngine, n_lis: usize) -> Vec<Vec<f32>> {
    let silence = AudioBuffer::new(1, BLOCK as u16);
    let mut imp = AudioBuffer::new(1, BLOCK as u16);
    imp.set(0, 0, 1.0);
    let mut outs: Vec<AudioBuffer> = (0..n_lis).map(|_| AudioBuffer::new(2, BLOCK as u16)).collect();
    let mut resp = vec![Vec::new(); n_lis];
    for b in 0..(SETTLE + BLOCKS) {
        for o in outs.iter_mut() {
            o.clear();
        }
        e.process_audio_scene(&[if b == SETTLE { &imp } else { &silence }], &mut outs);
        if b >= SETTLE {
            for (l, o) in outs.iter().enumerate() {
                for i in 0..BLOCK {
                    resp[l].push(o.channel(0)[i] + o.channel(1)[i]);
                }
            }
        }
    }
    resp
}

fn peak(x: &[f32]) -> (usize, f32) {
    let mut best = (0, 0.0f32);
    for (i, v) in x.iter().enumerate() {
        if v.abs() > best.1 {
            best = (i, v.abs());
        }
    }
    best
}

#[test]
fn two_listeners_get_their_own_delay_and_level() {
    // ~1 m and ~8 m from the emitter, stub backend (inverse-distance law, flat spectrum), at
    // distances giving INTEGER delays (140 and 1120 samples) so the fractional-delay interpolation
    // does not colour the level comparison.
    let (na, nb) = (140.0_f32, 1120.0_f32);
    let (pa, pb) = ([0.0, 0.0, na * C / SR], [0.0, 0.0, nb * C / SR]);
    let mut both = build(&[pa, pb], false);
    let r = impulse_responses(&mut both, 2);
    let (ia, va) = peak(&r[0]);
    let (ib, vb) = peak(&r[1]);
    let (da, db) = (na, nb);
    assert!((ia as f32 - da).abs() <= 1.5, "listener 0 delay {ia}, expected {da}");
    assert!((ib as f32 - db).abs() <= 1.5, "listener 1 delay {ib}, expected {db}");
    let ratio = vb / va;
    assert!((ratio - 1.0 / 8.0).abs() < 0.01, "level ratio {ratio}, expected 1/8 (distance law)");

    // Each listener sounds exactly as it does in an engine where it is the only listener.
    let ra = impulse_responses(&mut build(&[pa], false), 1);
    let rb = impulse_responses(&mut build(&[pb], false), 1);
    assert_eq!(r[0], ra[0], "listener 0 must be independent of listener 1");
    assert_eq!(r[1], rb[0], "listener 1 must not inherit listener 0's coefficients");
}

#[test]
fn occlusion_and_filtering_are_independent_per_listener() {
    // A: clear line at 3 m (z = +3). B: 4 m away behind the panel (z = -4).
    let (pa, pb) = ([0.0, 0.0, 3.0], [0.0, 0.0, -4.0]);
    let mut both = build(&[pa, pb], true);
    let r = impulse_responses(&mut both, 2);

    // Bit-exact against single-listener engines: neither listener's chain depends on the other.
    let ra = impulse_responses(&mut build(&[pa], true), 1);
    let rb = impulse_responses(&mut build(&[pb], true), 1);
    assert_eq!(r[0], ra[0], "listener 0 (clear) changed by the presence of listener 1");
    assert_eq!(r[1], rb[0], "listener 1 (occluded) is not rendered with its own occluded path");

    // Delays follow each listener's distance.
    let (ia, va) = peak(&r[0]);
    let (ib, vb) = peak(&r[1]);
    // The occluded path has a low-pass filter, which delays/smears the peak slightly.
    assert!((ia as f32 - 3.0 * SR / C).abs() <= 1.5, "A delay {ia}");
    assert!((ib as f32 - 4.0 * SR / C).abs() <= 12.0, "B delay {ib}");

    // Level and tilt: the panel removes level and (mostly) highs from B only.
    // Spectral tilt: energy of the first difference (a +6 dB/oct high-pass) relative to energy.
    let tilt = |x: &[f32]| {
        let e: f32 = x.iter().map(|v| v * v).sum();
        let d: f32 = x.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])).sum();
        d / e.max(1e-20)
    };
    let (ta, tb) = (tilt(&r[0]), tilt(&r[1]));
    assert!(vb < 0.6 * va * 3.0 / 4.0, "B must be quieter than the distance law alone ({vb} vs {va})");
    assert!(tb < 0.5 * ta, "B (behind the panel) must be low-passed: tilt {tb} vs clear {ta}");
}

#[test]
fn removing_a_listener_does_not_disturb_the_other() {
    let (pa, pb) = ([0.0, 0.0, 3.0], [0.0, 0.0, -4.0]);
    let mut a = build(&[pa, pb], true);
    let mut b = build(&[pa, pb], true);
    let silence = AudioBuffer::new(1, BLOCK as u16);
    let mut imp = AudioBuffer::new(1, BLOCK as u16);
    imp.set(0, 0, 1.0);
    let mut o2: Vec<AudioBuffer> = (0..2).map(|_| AudioBuffer::new(2, BLOCK as u16)).collect();
    let mut o1: Vec<AudioBuffer> = vec![AudioBuffer::new(2, BLOCK as u16)];
    let mut o2b: Vec<AudioBuffer> = (0..2).map(|_| AudioBuffer::new(2, BLOCK as u16)).collect();
    for blk in 0..30 {
        let input = if blk == 3 { &imp } else { &silence };
        a.process_audio_scene(&[input], &mut o2);
        b.process_audio_scene(&[input], &mut o2b);
    }
    // Remove listener 1 (occluded) from `a` while its response from the impulse is in flight.
    let imp2 = imp.clone();
    a.remove_listener(ListenerId(1));
    for blk in 0..20 {
        let input = if blk == 0 { &imp2 } else { &silence };
        a.process_audio_scene(&[input], &mut o1);
        b.process_audio_scene(&[input], &mut o2b);
        for c in 0..2 {
            assert_eq!(o1[0].channel(c), o2b[0].channel(c), "block {blk}: listener 0 disturbed by the removal");
        }
    }
}
