//! Noise-detection tests for the scene pipeline (`process_audio_scene`).
//!
//! Exercises the zero-alloc audio thread with realistic conditions that expose
//! clipping, NaN/Inf, tonal imbalance, self‑oscillation, and channel bleed.
//! Tests use an empty scene (no mesh) so the backend returns default reverb
//! params and no early reflections — the signal path is pure
//! (pull → occlusion → reverb → VBAP). An additional test uses a simple mesh
//! to exercise the realistic late‑reverb loudness path.

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

fn source(path: &str, channels: usize) -> SourceConfig {
    SourceConfig {
        path: path.to_string(),
        channels,
    }
}

fn dc(value: f32, channels: usize) -> AudioBuffer {
    let mut buf = AudioBuffer::new(channels as u16, BLOCK as u16);
    for c in 0..channels {
        for i in 0..BLOCK {
            buf.set(c as u16, i as u16, value);
        }
    }
    buf
}

fn silence(channels: usize) -> AudioBuffer {
    AudioBuffer::new(channels as u16, BLOCK as u16)
}

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

/// Render `blocks` blocks of the scene pipeline, returning the final buffer.
fn render_blocks(
    engine: &mut SpatialAudioEngine,
    sources: &[&AudioBuffer],
    listener_outputs: &mut [AudioBuffer],
    blocks: usize,
) {
    for _ in 0..blocks {
        for l in listener_outputs.iter_mut() {
            l.clear();
        }
        engine.process_audio_scene(sources, listener_outputs);
    }
}

// ── Test 1: no_clipping_with_full_level_source ────────────────────────────

#[test]
fn no_clipping_with_full_level_source() {
    let mut engine = engine_with_backend();
    let src = engine.load_source(source("test.wav", 1)).expect("load source");

    // Scene output 25 m from the listener → distance_attenuation ≈ 0.038.
    // (37 m would exceed the occlusion node's 0.1 s max delay at 48 kHz.)
    let out = engine.add_scene_output(SceneOutputConfig::new(
        [0.0, 0.0, -25.0],
        Movability::Static,
    ));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));

    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    engine.update_scene_spatial();

    let src_buf = dc(0.9, 1);
    let sources = [&src_buf];
    let mut out_buf = stereo_out();
    // 12 blocks — enough for the crossfader (~3 blocks @ 15 ms) and the
    // FDN to stabilise from its initial ring-up transient.
    render_blocks(&mut engine, &sources, std::slice::from_mut(&mut out_buf), 12);

    assert!(
        out_buf.peak() < 1.0,
        "output peak must be < 1.0 (no clipping); got {}",
        out_buf.peak()
    );
}

// ── Test 2: no_nan_or_inf_in_output ───────────────────────────────────────

#[test]
fn no_nan_or_inf_in_output() {
    let mut engine = engine_with_backend();
    let src = engine.load_source(source("test.wav", 1)).expect("load source");

    let out = engine.add_scene_output(SceneOutputConfig::new(
        [0.0, 0.0, -25.0],
        Movability::Static,
    ));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));

    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    engine.update_scene_spatial();

    let src_buf = dc(0.9, 1);
    let sources = [&src_buf];
    let mut out_buf = stereo_out();
    render_blocks(&mut engine, &sources, std::slice::from_mut(&mut out_buf), 12);

    for ch_idx in 0..out_buf.channels() {
        let ch = out_buf.channel(ch_idx);
        for (i, &sample) in ch.iter().enumerate() {
            assert!(
                sample.is_finite(),
                "sample {} in channel {} is non-finite (NaN or Inf); value={}",
                i,
                ch_idx,
                sample
            );
        }
    }
}

// ── Test 3: dry_tonal_content_dominates_reverb_tail ───────────────────────

#[test]
fn dry_tonal_content_dominates_reverb_tail() {
    let mut engine = engine_with_backend();
    let src = engine.load_source(source("test.wav", 1)).expect("load source");

    // One scene output 10 m from the listener → atten ≈ 0.09.
    let out = engine.add_scene_output(SceneOutputConfig::new(
        [0.0, 0.0, -10.0],
        Movability::Static,
    ));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));

    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    engine.update_scene_spatial();

    // Build a 480 Hz sine tone at 0.9 FS
    let mut sine = AudioBuffer::new(1, BLOCK as u16);
    let phase_inc = 480.0 / SR * std::f32::consts::TAU;
    for i in 0..BLOCK {
        sine.set(0, i as u16, 0.9 * (phase_inc * i as f32).sin());
    }
    let sources = [&sine];
    let mut out_bufs = [stereo_out()];

    // Render 100 blocks (sustained tone, more than enough for the FDN to fill).
    render_blocks(&mut engine, &sources, &mut out_bufs, 100);

    // Check the last block: dry tonal content should dominate over reverb.
    // A pure sine has peak-to-RMS ≈ √2 ≈ 1.414. Noise-like signals have
    // higher P2R (~3-4 for Gaussian).  If P2R < 2.5 the output is clearly
    // tonal (dry path dominates the noise-like reverb).
    let last_buf = &out_bufs[0];
    let peak = last_buf.peak();
    let rms = last_buf.rms();
    let p2r = if rms > 1e-10 { peak / rms } else { 0.0 };
    assert!(
        p2r < 2.5,
        "peak-to-RMS ratio must be < 2.5 for tonal content; got peak={peak}, rms={rms}, ratio={p2r}"
    );
}

// ── Test 4: no_acoustic_artifact_on_silent_input ──────────────────────────

#[test]
fn no_acoustic_artifact_on_silent_input() {
    let mut engine = engine_with_backend();
    let src = engine.load_source(source("test.wav", 1)).expect("load source");

    let out = engine.add_scene_output(SceneOutputConfig::new(
        [0.0, 0.0, -10.0],
        Movability::Static,
    ));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));

    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    engine.update_scene_spatial();

    let src_buf = silence(1);
    let sources = [&src_buf];
    let mut out_buf = stereo_out();
    render_blocks(&mut engine, &sources, std::slice::from_mut(&mut out_buf), 10);

    let peak = out_buf.peak();
    assert_eq!(
        peak, 0.0,
        "silent input must produce silence; got peak={}",
        peak
    );
}

// ── Test 5: channel_separation_in_multi_output_scenario ───────────────────

#[test]
fn channel_separation_in_multi_output_scenario() {
    let mut engine = engine_with_backend();

    // Two-channel source: ch0 gets a tone, ch1 is silent.
    let src = engine
        .load_source(source("stereo.wav", 2))
        .expect("load source");

    // Two scene outputs on opposite sides of the listener.
    let out_left = engine.add_scene_output(SceneOutputConfig::new(
        [-5.0, 0.0, 0.0],
        Movability::Static,
    ));
    engine.connect_pull(out_left, ChannelPull::new(src, 0, 0.0));

    let out_right = engine.add_scene_output(SceneOutputConfig::new(
        [5.0, 0.0, 0.0],
        Movability::Static,
    ));
    engine.connect_pull(out_right, ChannelPull::new(src, 1, 0.0));

    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    engine.update_scene_spatial();

    // Build a 480 Hz tone at 0.9 FS for ch0; ch1 is silence.
    let mut src_buf = AudioBuffer::new(2, BLOCK as u16);
    let phase_inc = 480.0 / SR * std::f32::consts::TAU;
    for i in 0..BLOCK {
        src_buf.set(0, i as u16, 0.9 * (phase_inc * i as f32).sin());
        src_buf.set(1, i as u16, 0.0);
    }
    let sources = [&src_buf];
    let mut out_buf = stereo_out();
    render_blocks(&mut engine, &sources, std::slice::from_mut(&mut out_buf), 12);

    // out_left (ch0, tone) is at x=-5 (left of center) → VBAP puts it mostly
    // in the left speaker.  out_right (ch1, silence) contributes nothing.
    let left_l = out_buf.channel(0).iter().map(|&s| s.abs()).sum::<f32>();
    let right_l = out_buf.channel(1).iter().map(|&s| s.abs()).sum::<f32>();

    assert!(
        left_l > 0.0 || right_l > 0.0,
        "output must contain audio from the active channel"
    );
    // Left speaker dominates (tone is on left-side output).
    assert!(
        left_l > right_l * 0.5,
        "left speaker must carry the tone (left_l={left_l}, right_l={right_l})"
    );
}

// ── Test 6: no_clipping_with_hybrid_blend_strategy ───────────────────────
//
// The cathedral demo uses `HybridBlend`, which hardcodes `late_loudness_db =
// -10` dB (wet ≈ 0.316).  With the FDN's internal gain ≈ 1.67× the reverb
// output reaches ~53 % of the filtered (dry) signal level.  When 3+ WAV
// channels overlap on one output AND 8 outputs VBAP-decode onto the listener,
// the combined + master gain (+18 dB ≙ ×7.94) clips.
//
// This test reproduces the cathedral's most aggressive scenario: 3 channel
// pulls into one output at a realistic distance, processed through HybridBlend.

#[test]
fn no_clipping_with_hybrid_blend_strategy() {
    use quasar_audio::quasar_core::bands::Band8;
    use quasar_audio::quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};

    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    let mut scene = AcousticScene::new();

    // Small scene with absorption 0.35 (back end estimate will be quiet but
    // HybridBlend overrides it to -10 dB).
    {
        use quasar_audio::quasar_core::scene::AcousticMesh;
        use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
        use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};

        engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
        let mat = engine.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::splat(0.35), Band8::zeros(), Band8::zeros()),
        ));
        scene.add_mesh(AcousticMesh::new(
            1,
            vec![
                [-5.0, 0.0, -5.0], [ 5.0, 0.0, -5.0], [ 5.0, 0.0,  5.0], [-5.0, 0.0,  5.0],
                [-5.0, 4.0, -5.0], [ 5.0, 4.0, -5.0], [ 5.0, 4.0,  5.0], [-5.0, 4.0,  5.0],
                [-5.0, 0.0, -5.0], [-5.0, 4.0, -5.0], [-5.0, 4.0,  5.0], [-5.0, 0.0,  5.0],
                [ 5.0, 0.0, -5.0], [ 5.0, 4.0, -5.0], [ 5.0, 4.0,  5.0], [ 5.0, 0.0,  5.0],
                [-5.0, 0.0, -5.0], [ 5.0, 0.0, -5.0], [ 5.0, 4.0, -5.0], [-5.0, 4.0, -5.0],
                [-5.0, 0.0,  5.0], [ 5.0, 0.0,  5.0], [ 5.0, 4.0,  5.0], [-5.0, 4.0,  5.0],
            ],
            vec![
                0,1,2, 0,2,3,    4,6,5, 4,7,6,
                8,9,10, 8,10,11,  12,14,13, 12,15,14,
                16,17,18, 16,18,19, 20,22,21, 20,23,22,
            ],
            mat,
        ));
    }
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())));

    // Probe grid: 2×2×2 = 8 probes so the listener position (0, 1.6, 0) is
    // inside the grid and the HybridBlend resolve succeeds.  Short t60 (0.3 s)
    // lets the crossfade transient decay fast.
    let grid_origin = [-10.0, -5.0, -10.0];
    let grid_spacing = [10.0, 10.0, 10.0];
    let grid_dims = [2u32, 2, 2];
    let mut probes = Vec::with_capacity(8);
    for z in 0..grid_dims[2] {
        for y in 0..grid_dims[1] {
            for x in 0..grid_dims[0] {
                probes.push(AcousticProbe {
                    position: [
                        grid_origin[0] + x as f32 * grid_spacing[0],
                        grid_origin[1] + y as f32 * grid_spacing[1],
                        grid_origin[2] + z as f32 * grid_spacing[2],
                    ],
                    rir_samples: Vec::new(),
                    sample_rate: 48000,
                    t60: Band8::splat(0.3),
                    broadband_t60: 0.3,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }
    engine.set_probe_grid(
        AcousticProbeGrid::new(probes, grid_origin, grid_spacing, grid_dims)
            .expect("probe grid"),
    );
    engine.set_strategy(HybridSamplingStrategy::HybridBlend);

    // 3-channel source to simulate overlapping WAV channels (cathedral's s4-s5).
    let src = engine
        .load_source(source("multi.wav", 3))
        .expect("load source");

    // Three scene outputs at 12 m (atten ≈ 0.077).  With HybridBlend's -10 dB
    // wet gain + FDN 1.67×, reverb ≈ 0.53 × dry → combined clips after VBAP
    // + master gain.  After fix (wet × 0.3), reverb ≈ 0.16 × dry (safe).
    for ch in 0..3 {
        let out = engine.add_scene_output(SceneOutputConfig::new(
            [0.0, 1.6, -12.0],
            Movability::Static,
        ));
        engine.connect_pull(out, ChannelPull::new(src, ch, 0.0));
    }

    engine.add_listener(ListenerConfig {
        position: [0.0, 1.6, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });

    engine.update_scene_spatial();

    // All 3 channels get DC 0.9 (worst-case sweep overlap).
    let mut src_buf = AudioBuffer::new(3, BLOCK as u16);
    for c in 0..3 {
        for i in 0..BLOCK {
            src_buf.set(c as u16, i as u16, 0.9);
        }
    }
    let sources = [&src_buf];
    let mut out_buf = stereo_out();
    // 400 blocks (≈ 2.1 s) ensures the crossfade transient and FDN ring-up
    // from the initial coefficients have fully decayed.
    render_blocks(&mut engine, &sources, std::slice::from_mut(&mut out_buf), 400);

    let peak = out_buf.peak();
    assert!(
        peak < 1.0,
        "no_clipping_with_hybrid_blend_strategy: output peak {peak} >= 1.0\n\
         HybridBlend's hardcoded -10 dB wet gain + FDN 1.67× gain + 3 overlapping\n\
         channels + VBAP produces clipping. The reverb wet gain must be reduced."
    );
}

// ── Test 7: reverb_level_with_realistic_absorption ────────────────────────
//
// A small "room" mesh with high-absorption walls forces the backend to compute
// realistic late_loudness_db (~ -4 dB).  The FDN then feeds the occlusion-
// attenuated signal, and the combined output must not clip even when multiple
// outputs drive the listener's speakers.

#[test]
fn reverb_level_with_realistic_absorption() {
    use quasar_audio::quasar_core::bands::Band8;
    use quasar_audio::quasar_core::scene::AcousticMesh;
    use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
    use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
    use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;

    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    let mut scene = AcousticScene::new();

    // Register a tabular material evaluator and a "wall" instance.
    engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let wall_mat = engine.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.35), Band8::zeros(), Band8::zeros()),
    ));

    // Small enclosure (10×5×4 m) to force a realistic late_loudness_db.
    // Two triangles per face.
    scene.add_mesh(AcousticMesh::new(
        1,
        vec![
            [-5.0, 0.0, -5.0], [ 5.0, 0.0, -5.0], [ 5.0, 0.0,  5.0], [-5.0, 0.0,  5.0], // floor
            [-5.0, 4.0, -5.0], [ 5.0, 4.0, -5.0], [ 5.0, 4.0,  5.0], [-5.0, 4.0,  5.0], // ceiling
            [-5.0, 0.0, -5.0], [-5.0, 4.0, -5.0], [-5.0, 4.0,  5.0], [-5.0, 0.0,  5.0], // left wall
            [ 5.0, 0.0, -5.0], [ 5.0, 4.0, -5.0], [ 5.0, 4.0,  5.0], [ 5.0, 0.0,  5.0], // right wall
            [-5.0, 0.0, -5.0], [ 5.0, 0.0, -5.0], [ 5.0, 4.0, -5.0], [-5.0, 4.0, -5.0], // front
            [-5.0, 0.0,  5.0], [ 5.0, 0.0,  5.0], [ 5.0, 4.0,  5.0], [-5.0, 4.0,  5.0], // back
        ],
        vec![
            0,1,2, 0,2,3,    4,6,5, 4,7,6,
            8,9,10, 8,10,11,  12,14,13, 12,15,14,
            16,17,18, 16,18,19, 20,22,21, 20,23,22,
        ],
        wall_mat,
    ));

    engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);

    let src = engine.load_source(source("test.wav", 1)).expect("load source");

    // Scene outputs at different distances covering realistic cathedral range.
    for dist in &[-5.0_f32, -10.0, -15.0, -25.0] {
        let out = engine.add_scene_output(SceneOutputConfig::new(
            [0.0, 0.0, *dist],
            Movability::Static,
        ));
        engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    }

    engine.add_listener(ListenerConfig {
        position: [0.0, 1.6, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Quad,
    });

    engine.update_scene_spatial();

    let src_buf = dc(0.9, 1);
    let sources = [&src_buf];
    let mut out_buf = AudioBuffer::new(4, BLOCK as u16);
    render_blocks(&mut engine, &sources, std::slice::from_mut(&mut out_buf), 50);

    let peak = out_buf.peak();
    assert!(
        peak < 1.0,
        "reverb_level_with_realistic_absorption: output peak {peak} >= 1.0 (clipping) — the reverb wet gain is too high relative to dry attenuation"
    );
}
