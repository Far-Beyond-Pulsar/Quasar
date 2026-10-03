//! #74: source directivity at engine level: orientation + directivity change the level at the
//! listener (direct path, early reflections along their own departure direction, reverb send).

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId, SourceConfig,
};
use quasar_audio::quasar_core::source_directivity::{diffuse_send_gain, pattern_band_gains};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const C: f32 = 343.0;
const BLOCK: usize = 256;

/// Stub-backend engine: emitter at the origin, stereo listener `dist` metres ahead (at -Z,
/// facing the emitter), unity-ish direct gain `1/dist`. Returns the engine and the output id.
fn stub_engine(dist: f32) -> (SpatialAudioEngine, SceneOutputId) {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "t.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([0.0; 3], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig {
        position: [0.0, 0.0, -dist],
        heading: [0.0, 0.0, 1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    (e, o)
}

fn tone(freq: f32, amp: f32, burst: Option<usize>) -> Vec<f32> {
    (0..BLOCK * 80)
        .map(|i| {
            let w = match burst {
                Some(n) if i < n => 0.5 - 0.5 * (std::f32::consts::TAU * (i as f32 + 0.5) / n as f32).cos(),
                Some(_) => 0.0,
                None => 1.0,
            };
            amp * w * (std::f32::consts::TAU * freq * i as f32 / SR).sin()
        })
        .collect()
}

/// Render `input` (looped by the caller as needed) in blocks, return the L+R mono output.
fn render(e: &mut SpatialAudioEngine, input: &[f32], blocks: usize) -> Vec<f32> {
    let mut out = [AudioBuffer::new(2, BLOCK as u16)];
    let mut y = Vec::new();
    let mut src = AudioBuffer::new(1, BLOCK as u16);
    for b in 0..blocks {
        for i in 0..BLOCK {
            src.set(0, i as u16, input.get(b * BLOCK + i).copied().unwrap_or(0.0));
        }
        e.process_audio_scene(&[&src], &mut out);
        for i in 0..BLOCK {
            y.push(out[0].channel(0)[i] + out[0].channel(1)[i]);
        }
    }
    y
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Steady-state RMS of a continuous tone after the crossfade has settled.
fn steady_level(e: &mut SpatialAudioEngine, freq: f32) -> f32 {
    let input = tone(freq, 0.4, None);
    let y = render(e, &input, 40);
    rms(&y[30 * BLOCK..])
}

#[test]
fn rotating_the_emitter_by_180_degrees_applies_the_rear_attenuation() {
    let d = 0.6_f32;
    let (mut e, o) = stub_engine(3.0);
    e.debug_audio_stage = 2;
    // Facing the listener (-Z): on axis, no attenuation.
    e.set_scene_output_directivity(o, Some([0.0, 0.0, -1.0]), d);
    e.update_scene_spatial();
    let front = steady_level(&mut e, 1000.0);
    // Rotated by 180 degrees: the listener is now behind the emitter.
    e.set_scene_output_directivity(o, Some([0.0, 0.0, 1.0]), d);
    e.update_scene_spatial();
    let rear = steady_level(&mut e, 1000.0);

    let want = pattern_band_gains(d, -1.0).0[4] / pattern_band_gains(d, 1.0).0[4]; // = 1 - d
    assert!((want - 0.4).abs() < 1e-6);
    let got = rear / front;
    assert!((got / want - 1.0).abs() < 0.05, "rear/front level {got}, pattern's rear attenuation {want}");

    // Omni is unchanged: directivity 0 (any orientation) and no orientation at all both equal
    // the engine that never heard of directivity, bit for bit.
    let (mut plain, _) = stub_engine(3.0);
    plain.debug_audio_stage = 2;
    plain.update_scene_spatial();
    let (mut a, oa) = stub_engine(3.0);
    a.debug_audio_stage = 2;
    a.set_scene_output_directivity(oa, Some([0.3, 0.2, 1.0]), 0.0);
    a.update_scene_spatial();
    let (mut b, ob) = stub_engine(3.0);
    b.debug_audio_stage = 2;
    b.set_scene_output_directivity(ob, None, 1.0);
    b.update_scene_spatial();
    let input = tone(1000.0, 0.4, None);
    let (yp, ya, yb) = (render(&mut plain, &input, 20), render(&mut a, &input, 20), render(&mut b, &input, 20));
    assert_eq!(yp, ya, "directivity 0 must be exactly omnidirectional");
    assert_eq!(yp, yb, "no orientation must be exactly omnidirectional");
}

#[test]
fn per_band_cone_is_tighter_at_high_frequencies() {
    // Listener at 90 degrees off axis: gain = p^n_b with p = 0.5 at d = 1 (cardioid side).
    let (mut e, o) = stub_engine(3.0);
    e.debug_audio_stage = 2;
    e.set_scene_output_directivity(o, Some([1.0, 0.0, 0.0]), 1.0); // facing +X, listener at -Z
    e.update_scene_spatial();
    let lo = steady_level(&mut e, 250.0); // band 2 centre
    let hi = steady_level(&mut e, 4000.0); // band 6 centre
    // Reference levels without directivity (flat stub spectrum).
    let (mut plain, _) = stub_engine(3.0);
    plain.debug_audio_stage = 2;
    plain.update_scene_spatial();
    let (lo0, hi0) = (steady_level(&mut plain, 250.0), steady_level(&mut plain, 4000.0));
    let g = pattern_band_gains(1.0, 0.0).0;
    let (got_lo, got_hi) = (lo / lo0, hi / hi0);
    assert!((got_lo / g[2] - 1.0).abs() < 0.08, "250 Hz: {got_lo} vs pattern {}", g[2]);
    assert!((got_hi / g[6] - 1.0).abs() < 0.08, "4 kHz: {got_hi} vs pattern {}", g[6]);
    assert!(got_hi < got_lo * 0.8, "the high band must be attenuated more than the low band");
}

/// Wall at x = 4, emitter at the origin, listener at (0, 0, -4) facing +Z, order-1 reflections.
/// The single reflection leaves the emitter toward the bounce point P = (4, 0, -2).
fn wall_engine() -> (SpatialAudioEngine, SceneOutputId) {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let mat = e.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::zeros(), Band8::zeros(), Band8::zeros()),
    ));
    let mut scene = AcousticScene::new();
    scene.add_mesh(AcousticMesh::new(
        1,
        vec![[4.0, -30.0, -30.0], [4.0, 30.0, -30.0], [4.0, 30.0, 30.0], [4.0, -30.0, 30.0]],
        vec![0, 1, 2, 0, 2, 3],
        mat,
    ));
    let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
    e.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "t.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([0.0; 3], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig {
        position: [0.0, 0.0, -4.0],
        heading: [0.0, 0.0, 1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e.debug_audio_stage = 3; // direct + early reflections
    (e, o)
}

/// RMS of the direct arrival and of the reflection arrival of a 1 kHz Hann tone burst.
fn direct_and_reflection_levels(orientation: [f32; 3], d: f32) -> (f32, f32) {
    let (mut e, o) = wall_engine();
    e.set_scene_output_directivity(o, Some(orientation), d);
    e.update_scene_spatial();
    let burst = tone(1000.0, 0.5, Some(240));
    // Settle the crossfade with silence, then send the burst.
    let mut input = vec![0.0f32; 30 * BLOCK];
    input.extend_from_slice(&burst);
    let y = render(&mut e, &input, 30 + 20);
    let y = &y[30 * BLOCK..];
    let direct_delay = (4.0 * SR / C) as usize;
    let path = 2.0 * (16.0_f32 + 4.0).sqrt();
    let refl_delay = (path * SR / C) as usize;
    let win = 300;
    let (dl, rl) = (rms(&y[direct_delay..direct_delay + win]), rms(&y[refl_delay..refl_delay + win]));
    (dl, rl)
}

#[test]
fn reflections_leave_the_emitter_in_their_own_direction() {
    let d = 0.8;
    // Bounce point P = (4, 0, -2): the emitter->P axis, and the emitter->listener axis (-Z).
    let to_p = [4.0, 0.0, -2.0];
    let to_l = [0.0, 0.0, -4.0];
    let (dir_a, refl_a) = direct_and_reflection_levels(to_p, d); // reflection on axis
    let (dir_b, refl_b) = direct_and_reflection_levels(to_l, d); // direct on axis

    // cos of the angle between the two axes:
    let cos = (to_p[2] * to_l[2]) / ((16.0f32 + 4.0).sqrt() * 4.0); // 2/sqrt(20) = 0.447
    let g_off = pattern_band_gains(d, cos).0[4];
    // Orientation A has the reflection on axis and the direct path off axis; B is the reverse.
    let refl_ratio = refl_a / refl_b; // expected 1 / g_off
    let direct_ratio = dir_a / dir_b; // expected g_off
    assert!(refl_ratio > 1.1, "the reflection must be louder when the emitter faces the bounce point ({refl_ratio})");
    assert!(direct_ratio < 0.9, "the direct path must be quieter when the emitter faces away ({direct_ratio})");
    assert!((refl_ratio * g_off - 1.0).abs() < 0.08, "reflection ratio {refl_ratio}, expected {}", 1.0 / g_off);
    assert!((direct_ratio / g_off - 1.0).abs() < 0.08, "direct ratio {direct_ratio}, expected {g_off}");
}

#[test]
fn reverb_send_follows_the_diffuse_field_average_not_the_listener_direction() {
    // Stub backend: late_gain_db is a fixed -10 dB. The listener is on axis for BOTH engines, so
    // the direct path is identical; only the late field differs.
    let d = 1.0;
    let run = |directivity: f32| {
        let (mut e, o) = stub_engine(3.0);
        e.debug_audio_stage = 4;
        e.set_scene_output_directivity(o, Some([0.0, 0.0, -1.0]), directivity);
        e.update_scene_spatial();
        // The diffuse tail starts `early_late_split_secs` (0.05 s = 2400 frames for the stub) after the
        // direct sound (#120 wired the split into the renderer), so render long enough to see it.
        let mut imp = vec![0.0f32; 70 * BLOCK];
        imp[20 * BLOCK] = 0.5;
        let y = render(&mut e, &imp, 70);
        // Energy well after the direct sound: the reverberant tail only.
        y[23 * BLOCK..].iter().map(|v| v * v).sum::<f32>()
    };
    let (omni, directional) = (run(0.0), run(d));
    let want = diffuse_send_gain(d).powi(2);
    let got = directional / omni;
    assert!(want < 0.7, "the pattern must radiate less total power than omni ({want})");
    assert!((got / want - 1.0).abs() < 0.05, "late energy ratio {got}, diffuse-field power {want}");
}
