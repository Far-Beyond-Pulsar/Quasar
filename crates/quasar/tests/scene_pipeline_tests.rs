//! End-to-end scene-pipeline tests (P2: channel pulling → spatial render → VBAP decode).
//!
//! Exercises `SpatialAudioEngine`'s zero-alloc audio thread
//! (`process_audio_scene`) against the real-time CPU SIMD backend with an
//! empty scene, so the direct path has no occlusion and there are no early
//! reflections — the dominant signal is a panned direct path.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::AcousticScene;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
/// Blocks rendered per scenario. Covers the 15 ms crossfade (~3 blocks) plus
/// the direct-path delay (~1 block) and the lowpass settling tail.
const RENDER_BLOCKS: usize = 12;

fn source(path: &str, channels: usize) -> SourceConfig {
    SourceConfig {
        path: path.to_string(),
        channels,
    }
}

/// Mono DC buffer (constant `value`).
fn dc(value: f32) -> AudioBuffer {
    let mut buf = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        buf.set(0, i as u16, value);
    }
    buf
}

/// Two-channel (Stereo) listener output buffer.
fn stereo_out() -> AudioBuffer {
    AudioBuffer::new(2, BLOCK as u16)
}

fn engine_with_backend() -> SpatialAudioEngine {
    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(
        AcousticScene::new(),
        CpuSimdConfig::default(),
    )));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    engine
}

/// Render `blocks` blocks into a stereo buffer and return the last block.
fn render_blocks(engine: &mut SpatialAudioEngine, sources: &[&AudioBuffer], blocks: usize) -> AudioBuffer {
    let mut out = stereo_out();
    for _ in 0..blocks {
        out.clear();
        engine.process_audio_scene(sources, std::slice::from_mut(&mut out));
    }
    out
}

fn sum_channel(buf: &AudioBuffer, ch: u16) -> f32 {
    buf.channel(ch).iter().sum()
}

// ── pull_renders_and_pans_to_listener ────────────────────────────────

#[test]
fn pull_renders_and_pans_to_listener() {
    let mut engine = engine_with_backend();

    let src = engine.load_source(source("mono.wav", 1)).expect("load source");

    // Emitter to the RIGHT of the listener.
    let out = engine.add_scene_output(SceneOutputConfig::new([2.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));

    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    // Resolve geometry once; fades converge over the first few blocks.
    engine.update_scene_spatial();

    let source_buf = dc(1.0);
    let sources = [&source_buf];
    let last = render_blocks(&mut engine, &sources, RENDER_BLOCKS);

    // Output shape + non-silence.
    assert_eq!(last.channels(), 2, "stereo listener must produce 2 channels");
    assert!(last.peak() > 0.0, "scene pipeline must render audio");

    // Emitter right of the listener → right channel must dominate.
    let left = sum_channel(&last, 0);
    let right = sum_channel(&last, 1);
    assert!(
        right > left,
        "source to the right must pan right (L={left}, R={right})"
    );

    // Move the emitter behind-left; a fresh resolve + fades must flip the pan.
    engine.set_scene_output_position(out, [-2.0, 0.0, 1.0]);
    engine.update_scene_spatial();
    let last2 = render_blocks(&mut engine, &sources, RENDER_BLOCKS);

    let left2 = sum_channel(&last2, 0);
    let right2 = sum_channel(&last2, 1);
    assert!(
        left2 >= right2,
        "source behind-left must pan left (L={left2}, R={right2})"
    );
}

// ── pull_gain_controls_output_level ──────────────────────────────────

#[test]
fn pull_gain_controls_output_level() {
    let mut engine = engine_with_backend();

    let src = engine.load_source(source("mono.wav", 1)).expect("load source");
    let out = engine.add_scene_output(SceneOutputConfig::new([5.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    engine.add_listener(ListenerConfig::default());

    engine.update_scene_spatial();
    let source_buf = dc(1.0);
    let sources = [&source_buf];

    let ref_out = render_blocks(&mut engine, &sources, RENDER_BLOCKS);
    let ref_level = ref_out.peak();
    assert!(ref_level > 0.0, "reference render must not be silent");

    // Drop the pull to -40 dB → the entire rendered signal scales down ~100×.
    engine.set_pull_gain(out, src, 0, -40.0);
    engine.update_scene_spatial();
    let quiet_out = render_blocks(&mut engine, &sources, RENDER_BLOCKS);
    let quiet_level = quiet_out.peak();

    assert!(
        quiet_level < ref_level * 0.05,
        "-40 dB pull gain must cut the level (ref={ref_level}, quiet={quiet_level})"
    );
}

// ── no_pulls_renders_silence ─────────────────────────────────────────

#[test]
fn no_pulls_renders_silence() {
    let mut engine = engine_with_backend();

    // A source exists, but nothing pulls from it.
    let _src = engine.load_source(source("unused.wav", 1)).expect("load source");
    let _out = engine.add_scene_output(SceneOutputConfig::new([2.0, 0.0, 0.0], Movability::Static));
    engine.add_listener(ListenerConfig::default());

    engine.update_scene_spatial();
    let source_buf = dc(1.0);
    let sources = [&source_buf];
    let last = render_blocks(&mut engine, &sources, RENDER_BLOCKS);

    assert_eq!(last.peak(), 0.0, "no pulls → no audible content");
}
