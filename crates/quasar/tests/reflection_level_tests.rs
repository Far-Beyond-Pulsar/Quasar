//! Engine-level reflection level (#59, #125): the reflection stage is fed the
//! un-attenuated dry signal, its taps carry the FULL path attenuation and surface
//! reflection coefficients, and it does not depend on the direct path's occlusion.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::air::air_absorption_gain;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::distance::DistanceModel;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const C: f32 = 343.0;
const BLOCK: usize = 256;
const SETTLE: usize = 24;
const TAIL_BLOCKS: usize = 90;

/// Render a unit impulse (stage 3 = direct + reflections) and return (left, right).
fn impulse_response(engine: &mut SpatialAudioEngine) -> (Vec<f32>, Vec<f32>) {
    engine.debug_audio_stage = 3;
    let silence = AudioBuffer::new(1, BLOCK as u16);
    let mut imp = AudioBuffer::new(1, BLOCK as u16);
    imp.set(0, 0, 1.0);
    let mut out = AudioBuffer::new(2, BLOCK as u16);
    let (mut l, mut r) = (Vec::new(), Vec::new());
    for b in 0..(SETTLE + TAIL_BLOCKS) {
        out.clear();
        let input = if b == SETTLE { &imp } else { &silence };
        engine.process_audio_scene(&[input], std::slice::from_mut(&mut out));
        if b >= SETTLE {
            l.extend_from_slice(&out.channel(0)[..BLOCK]);
            r.extend_from_slice(&out.channel(1)[..BLOCK]);
        }
    }
    (l, r)
}

/// sqrt(sum of squares over both channels) in `centre +- half` samples.
fn window_level(l: &[f32], r: &[f32], centre: f32, half: usize) -> f32 {
    let c = centre.round() as usize;
    let (lo, hi) = (c.saturating_sub(half), (c + half).min(l.len()));
    let e: f32 = (lo..hi).map(|i| l[i] * l[i] + r[i] * r[i]).sum();
    e.sqrt()
}

fn db(ratio: f32) -> f32 {
    20.0 * ratio.max(1e-12).log10()
}

/// Wall at x = `wx` (absorption `alpha`), optionally a small panel on the direct line.
/// Emitter at the origin, listener at (0, 0, -4) facing +z, so the wall bounce is
/// off the side: B = (wx, 0, -2).
fn build(wx: f32, alpha: f32, with_panel: bool) -> SpatialAudioEngine {
    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let wall_mat = engine.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(alpha), Band8::zeros(), Band8::zeros()),
    ));
    let mut scene = AcousticScene::new();
    scene.add_mesh(AcousticMesh::new(
        1,
        vec![[wx, -30.0, -30.0], [wx, 30.0, -30.0], [wx, 30.0, 30.0], [wx, -30.0, 30.0]],
        vec![0, 1, 2, 0, 2, 3],
        wall_mat,
    ));
    if with_panel {
        // Opaque 2 m x 4 m panel across the direct line only (z = -2, x in -1..1).
        let panel_mat = engine.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::splat(0.5), Band8::zeros(), Band8::zeros()),
        ));
        scene.add_mesh(AcousticMesh::new(
            2,
            vec![[-1.0, -2.0, -2.0], [1.0, -2.0, -2.0], [1.0, 2.0, -2.0], [-1.0, 2.0, -2.0]],
            vec![0, 1, 2, 0, 2, 3],
            panel_mat,
        ));
    }
    let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);

    let src = engine.load_source(SourceConfig { path: "imp.wav".into(), channels: 1 }).expect("source");
    let out = engine.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, -4.0],
        heading: [0.0, 0.0, 1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    engine.update_scene_spatial();
    engine
}

/// Wall distance giving a reflected path of exactly `n` samples at `SR` / `C`
/// (so the tap sits on a sample and the fractional-delay interpolation loses nothing).
fn wall_x_for_path_samples(n: f32) -> f32 {
    let path = (n as f64) * (C as f64) / (SR as f64);
    // path = 2 sqrt(wx^2 + 2^2)
    ((path / 2.0).powi(2) - 4.0).sqrt() as f32
}

#[test]
fn occluding_the_direct_path_leaves_the_reflection_unchanged() {
    let n = 1500.0_f32;
    let wx = wall_x_for_path_samples(n);
    let (cl, cr) = impulse_response(&mut build(wx, 0.0, false));
    let (bl, br) = impulse_response(&mut build(wx, 0.0, true));

    let direct_delay = 4.0 * SR / C;
    let clear_direct = window_level(&cl, &cr, direct_delay, 64);
    let blocked_direct = window_level(&bl, &br, direct_delay, 64);
    assert!(
        db(blocked_direct / clear_direct) < -6.0,
        "the panel must attenuate the direct path: {} dB",
        db(blocked_direct / clear_direct)
    );

    let clear_refl = window_level(&cl, &cr, n, 64);
    let blocked_refl = window_level(&bl, &br, n, 64);
    assert!(clear_refl > 1e-3, "reflection must be audible: {clear_refl}");
    assert!(
        db(blocked_refl / clear_refl).abs() < 0.5,
        "reflection level changed by {} dB when only the direct path was occluded",
        db(blocked_refl / clear_refl)
    );
}

#[test]
fn reflection_level_is_path_attenuation_times_reflection_coefficient() {
    let n = 1500.0_f32;
    let wx = wall_x_for_path_samples(n);
    let len = n * C / SR; // total path length (m)
    let dm = DistanceModel::default();
    for alpha in [0.0_f32, 0.19, 0.5] {
        let (l, r) = impulse_response(&mut build(wx, alpha, false));
        let got = window_level(&l, &r, n, 64);
        // Band-mean of the per-band path gain (the stage folds bands; broadband impulse).
        let air = air_absorption_gain(len, 20.0, 50.0);
        let band_mean = air.0.iter().sum::<f32>() / 8.0;
        let want = dm.gain(len) * (1.0 - alpha).sqrt() * band_mean;
        assert!(
            db(got / want).abs() < 1.0,
            "alpha {alpha}: reflection level {got} vs analytic {want} ({} dB)",
            db(got / want)
        );
    }
}

#[test]
fn reflection_level_does_not_follow_the_direct_gain() {
    // Direct gain 1/4 (distance law), reflection 1/length: their ratio is the length
    // ratio, not its square (double attenuation would make the reflection ~ 1/(4 x length)).
    let n = 1500.0_f32;
    let wx = wall_x_for_path_samples(n);
    let (l, r) = impulse_response(&mut build(wx, 0.0, false));
    let refl = window_level(&l, &r, n, 64);
    let direct = window_level(&l, &r, 4.0 * SR / C, 64);
    let ratio = direct / refl;
    let len = n * C / SR;
    assert!(ratio > 0.7 * len / 4.0 && ratio < 1.3 * len / 4.0, "direct/reflection {ratio}, length ratio {}", len / 4.0);
}
