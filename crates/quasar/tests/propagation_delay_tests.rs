//! Engine-level propagation delay / timing tests (#48, #52, #57 convention).
//!
//! The direct path is delayed by its propagation time (`dist * fs / c`); early
//! reflections carry the full emission -> listener path delay. Both are on the
//! same clock, so a reflection arrives `(L_refl - L_direct) * fs / c` samples
//! after the direct sound.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const BLOCK: usize = 256;
const C: f32 = 343.0;
/// Blocks of silence rendered first so crossfades and delay ramps have settled.
const SETTLE: usize = 24;
const TAIL_BLOCKS: usize = 90;

fn run(
    engine: &mut SpatialAudioEngine,
    stage: u8,
) -> Vec<f32> {
    engine.debug_audio_stage = stage;
    let src = AudioBuffer::new(1, BLOCK as u16);
    let mut imp = AudioBuffer::new(1, BLOCK as u16);
    imp.set(0, 0, 1.0);
    let mut out = AudioBuffer::new(2, BLOCK as u16);
    let mut left = Vec::new();
    for b in 0..(SETTLE + TAIL_BLOCKS) {
        out.clear();
        let input = if b == SETTLE { &imp } else { &src };
        engine.process_audio_scene(&[input], std::slice::from_mut(&mut out));
        if b >= SETTLE {
            left.extend_from_slice(&out.channel(0)[..BLOCK]);
        }
    }
    left
}

/// Sub-sample peak position (parabolic interpolation) of the largest |value| in `x[lo..hi]`.
fn peak_pos(x: &[f32], lo: usize, hi: usize) -> f32 {
    let mut best = lo.max(1);
    for i in lo.max(1)..hi.min(x.len() - 1) {
        if x[i].abs() > x[best].abs() {
            best = i;
        }
    }
    let (a, b, c) = (x[best - 1].abs(), x[best].abs(), x[best + 1].abs());
    let denom = a - 2.0 * b + c;
    let off = if denom.abs() > 1e-12 { 0.5 * (a - c) / denom } else { 0.0 };
    best as f32 + off
}

fn engine_for(sr: f32, scene: AcousticScene) -> SpatialAudioEngine {
    let mut engine = SpatialAudioEngine::new(0, sr, 15.0);
    // The engine hands its own sample rate to the backend when it is installed.
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    engine
}

fn add_ahead_listener(engine: &mut SpatialAudioEngine, listener_z: f32) {
    let src = engine
        .load_source(SourceConfig { path: "imp.wav".into(), channels: 1 })
        .expect("source");
    let out = engine.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    // Facing +Z, the emitter at the origin is dead ahead.
    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, listener_z],
        heading: [0.0, 0.0, 1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    engine.update_scene_spatial();
}

#[test]
fn direct_sound_arrives_after_its_propagation_time_at_any_device_rate() {
    for sr in [44_100.0_f32, 48_000.0, 96_000.0] {
        for dist in [5.0_f32, 20.0, 60.0] {
            let mut engine = engine_for(sr, AcousticScene::new());
            add_ahead_listener(&mut engine, -dist);
            let left = run(&mut engine, 2); // direct path only
            let expected = dist * sr / C;
            let got = peak_pos(&left, 0, left.len());
            assert!(
                (got - expected).abs() <= 1.0,
                "sr {sr} dist {dist}: peak at {got}, expected {expected}"
            );
        }
    }
}

#[test]
fn reflection_follows_direct_by_the_path_length_difference() {
    for sr in [44_100.0_f32, 48_000.0] {
        // Wall at z = -10, listener at z = -4, emitter at the origin: the
        // reflected path is 10 m (to the wall) + 6 m (back) = 16 m, the direct
        // path 4 m, so the reflection trails by 12 m.
        let mut scene = AcousticScene::new();
        // Material handle is only known after registration, so build the engine first.
        let mut engine = SpatialAudioEngine::new(0, sr, 15.0);
        engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
        let wall = engine.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::zeros(), Band8::zeros(), Band8::zeros()),
        ));
        scene.add_mesh(AcousticMesh::new(
            1,
            vec![[-50.0, -50.0, -10.0], [50.0, -50.0, -10.0], [50.0, 50.0, -10.0], [-50.0, 50.0, -10.0]],
            vec![0, 1, 2, 0, 2, 3],
            wall,
        ));
        engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())));
        engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
        add_ahead_listener(&mut engine, -4.0);

        let left = run(&mut engine, 3); // direct + early reflections

        let d_direct = 4.0 * sr / C;
        let d_refl = 16.0 * sr / C;
        let p_direct = peak_pos(&left, 0, (d_direct + 200.0) as usize);
        let p_refl = peak_pos(&left, (d_direct + 600.0) as usize, (d_refl + 400.0) as usize);
        assert!((p_direct - d_direct).abs() <= 1.0, "direct at {p_direct}, expected {d_direct}");
        assert!(
            ((p_refl - p_direct) - (d_refl - d_direct)).abs() <= 1.0,
            "sr {sr}: reflection trails direct by {}, expected {}",
            p_refl - p_direct,
            d_refl - d_direct
        );
        assert!(left[p_refl as usize].abs() > 1e-4, "reflection must be audible");
    }
}

/// Like [`run`] but returns the energy envelope `sqrt(L^2 + R^2)` (a reflection
/// that is panned hard to one side still shows up).
fn run_envelope(engine: &mut SpatialAudioEngine, stage: u8) -> Vec<f32> {
    engine.debug_audio_stage = stage;
    let src = AudioBuffer::new(1, BLOCK as u16);
    let mut imp = AudioBuffer::new(1, BLOCK as u16);
    imp.set(0, 0, 1.0);
    let mut out = AudioBuffer::new(2, BLOCK as u16);
    let mut env = Vec::new();
    for b in 0..(SETTLE + TAIL_BLOCKS) {
        out.clear();
        let input = if b == SETTLE { &imp } else { &src };
        engine.process_audio_scene(&[input], std::slice::from_mut(&mut out));
        if b >= SETTLE {
            for i in 0..BLOCK {
                let (l, r) = (out.channel(0)[i], out.channel(1)[i]);
                env.push((l * l + r * r).sqrt());
            }
        }
    }
    env
}

/// #57 end to end with the image-source tracer: an OFF-AXIS side-wall reflection
/// (the path is not collinear with the direct one) trails the direct sound by
/// `(L_refl - L_direct) * fs / c` samples at every device rate.
#[test]
fn side_wall_reflection_trails_direct_by_the_path_length_difference() {
    for sr in [44_100.0_f32, 48_000.0, 96_000.0] {
        let mut engine = SpatialAudioEngine::new(0, sr, 15.0);
        engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
        let wall = engine.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::zeros(), Band8::zeros(), Band8::zeros()),
        ));
        let mut scene = AcousticScene::new();
        // Wall at x = 5: emitter (0,0,0), listener (0,0,-4); bounce at (5,0,-2).
        scene.add_mesh(AcousticMesh::new(
            1,
            vec![[5.0, -50.0, -50.0], [5.0, 50.0, -50.0], [5.0, 50.0, 50.0], [5.0, -50.0, 50.0]],
            vec![0, 1, 2, 0, 2, 3],
            wall,
        ));
        let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
        engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
        engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
        add_ahead_listener(&mut engine, -4.0);

        let env = run_envelope(&mut engine, 3);
        let l_direct = 4.0_f32;
        let l_refl = 2.0 * (25.0_f32 + 4.0).sqrt();
        let d_direct = l_direct * sr / C;
        let d_refl = l_refl * sr / C;
        let p_direct = peak_pos(&env, 0, (d_direct + 200.0) as usize);
        let p_refl = peak_pos(&env, (d_direct + 300.0) as usize, (d_refl + 300.0) as usize);
        assert!((p_direct - d_direct).abs() <= 1.0, "sr {sr}: direct at {p_direct}, expected {d_direct}");
        assert!(
            ((p_refl - p_direct) - (d_refl - d_direct)).abs() <= 1.0,
            "sr {sr}: reflection trails direct by {}, expected {}",
            p_refl - p_direct,
            d_refl - d_direct
        );
        assert!(env[p_refl as usize] > 1e-3, "reflection must be audible");
    }
}
