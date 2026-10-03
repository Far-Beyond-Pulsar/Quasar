//! Engine-level distance model (#51): engine-wide model and per-output override.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::distance::{DistanceCurve, DistanceModel};
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const BLOCK: usize = 256;

/// Steady-state DC level of the left speaker for an emitter dead ahead at `dist`.
fn dc_level(
    dist: f32,
    engine_model: Option<DistanceModel>,
    output_model: Option<DistanceModel>,
) -> f32 {
    let mut engine = SpatialAudioEngine::new(0, 48_000.0, 15.0);
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default())));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    if let Some(m) = engine_model {
        engine.set_distance_model(m);
    }
    let src = engine.load_source(SourceConfig { path: "dc.wav".into(), channels: 1 }).unwrap();
    let out = engine.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    if let Some(m) = output_model {
        engine.set_scene_output_distance_model(out, Some(m));
    }
    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, -dist],
        heading: [0.0, 0.0, 1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    engine.update_scene_spatial();
    engine.debug_audio_stage = 2; // direct path only

    let mut dc = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        dc.set(0, i as u16, 1.0);
    }
    let mut o = AudioBuffer::new(2, BLOCK as u16);
    for _ in 0..120 {
        o.clear();
        engine.process_audio_scene(&[&dc], std::slice::from_mut(&mut o));
    }
    o.channel(0)[BLOCK - 1]
}

#[test]
fn default_model_halves_the_level_per_doubling() {
    // DC sits in the lowest band, where air absorption is negligible.
    let near = dc_level(4.0, None, None);
    let far = dc_level(8.0, None, None);
    let db = 20.0 * (far / near).log10();
    assert!((db + 6.02).abs() < 0.1, "{db} dB per doubling");
}

#[test]
fn engine_wide_model_changes_the_level() {
    let default = dc_level(8.0, None, None);
    let steep = dc_level(
        8.0,
        Some(DistanceModel { curve: DistanceCurve::Exponential, rolloff_factor: 2.0, ..DistanceModel::default() }),
        None,
    );
    // 1/8 vs 1/64.
    assert!((default / steep - 8.0).abs() < 0.2, "default {default} steep {steep}");
}

#[test]
fn per_output_override_replaces_the_engine_model() {
    let flat = DistanceModel { rolloff_factor: 0.0, ..DistanceModel::default() };
    let base = dc_level(1.0, None, None);
    let overridden = dc_level(16.0, None, Some(flat));
    assert!((overridden / base - 1.0).abs() < 0.02, "override should remove distance falloff: {overridden} vs {base}");
    let normal = dc_level(16.0, None, None);
    assert!((base / normal - 16.0).abs() < 0.5);
}
