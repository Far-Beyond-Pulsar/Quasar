//! #156 at engine level: spherical and directional speaker patterns change the level at the
//! listener as specified, and the legacy directivity path is unchanged.

use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::emitter_pattern::EmitterPattern;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Emitter at the origin, stereo listener 4 m away at -Z facing it. Stage 2 = direct path only.
fn engine() -> (SpatialAudioEngine, SceneOutputId) {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "t.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([0.0; 3], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig {
        position: [0.0, 0.0, -4.0],
        heading: [0.0, 0.0, 1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e.debug_audio_stage = 2;
    (e, o)
}

/// Steady-state RMS (L+R) of a continuous tone.
fn level(e: &mut SpatialAudioEngine, freq: f32) -> f32 {
    e.update_scene_spatial();
    let mut out = [AudioBuffer::new(2, BLOCK as u16)];
    let mut src = AudioBuffer::new(1, BLOCK as u16);
    let mut acc = 0.0_f64;
    let mut n = 0usize;
    for b in 0..60 {
        for i in 0..BLOCK {
            let t = (b * BLOCK + i) as f32;
            src.set(0, i as u16, 0.4 * (std::f32::consts::TAU * freq * t / SR).sin());
        }
        e.process_audio_scene(&[&src], &mut out);
        if b >= 40 {
            for i in 0..BLOCK {
                let v = out[0].channel(0)[i] + out[0].channel(1)[i];
                acc += (v as f64) * (v as f64);
                n += 1;
            }
        }
    }
    ((acc / n as f64).sqrt()) as f32
}

fn db(a: f32, b: f32) -> f32 {
    20.0 * (a / b).log10()
}

/// Orientation pointing `az` degrees to the side / `el` degrees up from the emitter -> listener
/// direction (0, 0, -1).
fn aim(az_deg: f32, el_deg: f32) -> [f32; 3] {
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    [az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()]
}

#[test]
fn horn_is_minus_6_db_at_its_edge_and_deep_at_the_rear() {
    // 60 x 40 degree horn.
    let freq = 2000.0; // pattern control above ~1 kHz
    let reference = {
        let (mut e, _) = engine();
        level(&mut e, freq)
    };
    let at = |az: f32, el: f32| {
        let (mut e, o) = engine();
        e.set_scene_output_directivity(o, Some(aim(az, el)), 0.0);
        e.set_scene_output_pattern(o, Some(EmitterPattern::horn(60.0, 40.0)));
        level(&mut e, freq)
    };
    // On axis the horn is transparent.
    assert!(db(at(0.0, 0.0), reference).abs() < 0.2, "on axis {} dB", db(at(0.0, 0.0), reference));
    // -6 dB at the horizontal (30 deg) and vertical (20 deg) half-angles.
    let h = db(at(30.0, 0.0), reference);
    let v = db(at(0.0, 20.0), reference);
    assert!((h + 6.0).abs() < 0.6, "horizontal edge {h} dB");
    assert!((v + 6.0).abs() < 0.6, "vertical edge {v} dB");
    // Narrower vertically than horizontally at the same off-axis angle.
    assert!(db(at(0.0, 30.0), reference) < db(at(30.0, 0.0), reference) - 2.0);
    // At least -20 dB at the rear.
    let rear = db(at(180.0, 0.0), reference);
    assert!(rear <= -20.0, "rear {rear} dB");
}

#[test]
fn sound_cone_is_exact_at_engine_level() {
    let freq = 1000.0;
    let reference = {
        let (mut e, _) = engine();
        level(&mut e, freq)
    };
    let at = |az: f32| {
        let (mut e, o) = engine();
        e.set_scene_output_directivity(o, Some(aim(az, 0.0)), 0.0);
        e.set_scene_output_pattern(
            o,
            Some(EmitterPattern::SoundCone { inner_deg: 60.0, outer_deg: 120.0, outer_gain_db: -20.0 }),
        );
        db(level(&mut e, freq), reference)
    };
    assert!(at(25.0).abs() < 0.2, "inside the inner cone {}", at(25.0));
    assert!((at(90.0) + 20.0).abs() < 0.6, "outside the outer cone {}", at(90.0));
    assert!((at(45.0) + 10.0).abs() < 0.6, "halfway {}", at(45.0));
}

#[test]
fn explicit_omni_and_cardioid_family_match_the_legacy_directivity_path() {
    let freq = 1000.0;
    // Explicit Omni with an orientation == no pattern at all.
    let plain = {
        let (mut e, _) = engine();
        level(&mut e, freq)
    };
    let omni = {
        let (mut e, o) = engine();
        e.set_scene_output_directivity(o, Some(aim(70.0, 0.0)), 0.0);
        e.set_scene_output_pattern(o, Some(EmitterPattern::Omni));
        level(&mut e, freq)
    };
    assert_eq!(plain, omni, "explicit omni is bit-identical to an emitter without a pattern");
    // CardioidFamily == legacy `directivity`.
    for d in [0.5_f32, 1.0] {
        let legacy = {
            let (mut e, o) = engine();
            e.set_scene_output_directivity(o, Some(aim(70.0, 0.0)), d);
            level(&mut e, freq)
        };
        let family = {
            let (mut e, o) = engine();
            e.set_scene_output_directivity(o, Some(aim(70.0, 0.0)), 0.0);
            e.set_scene_output_pattern(o, Some(EmitterPattern::CardioidFamily { directivity: d }));
            level(&mut e, freq)
        };
        assert_eq!(legacy, family, "d={d}");
        assert!(db(legacy, plain) < -1.0, "the pattern must actually attenuate at 70 deg: {}", db(legacy, plain));
    }
}

#[test]
fn clearing_the_pattern_returns_to_the_legacy_behaviour() {
    let freq = 1000.0;
    let (mut e, o) = engine();
    e.set_scene_output_directivity(o, Some(aim(70.0, 0.0)), 1.0);
    let legacy = level(&mut e, freq);
    e.set_scene_output_pattern(o, Some(EmitterPattern::horn(30.0, 30.0)));
    let horn = level(&mut e, freq);
    assert!(horn < legacy, "a narrow horn is further down than a cardioid at 70 deg");
    e.set_scene_output_pattern(o, None);
    let back = level(&mut e, freq);
    assert!((back - legacy).abs() / legacy < 1e-3, "{back} vs {legacy}");
}

#[test]
fn without_an_orientation_every_pattern_is_omnidirectional() {
    let freq = 1000.0;
    let plain = {
        let (mut e, _) = engine();
        level(&mut e, freq)
    };
    let (mut e, o) = engine();
    e.set_scene_output_pattern(o, Some(EmitterPattern::horn(30.0, 20.0)));
    assert_eq!(level(&mut e, freq), plain);
}

// ── Reflections leave the emitter in their own direction (#156) ─────────────

mod reflections {
    use super::*;
    use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
    use quasar_audio::quasar_backends::CpuSimdComputeBackend;
    use quasar_audio::quasar_core::bands::Band8;
    use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene};
    use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
    use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};

    const C: f32 = 343.0;
    const SRC: [f32; 3] = [10.0, 1.7, 8.0];
    const LIS: [f32; 3] = [10.0, 1.7, 25.0];

    /// Empty 20 x 20 x 30 m box (inward-facing quads), emitter at z = 8, listener at z = 25. The
    /// ceiling is high so its reflection (40 m) does not share a window with the far wall (27 m).
    fn room_engine() -> (SpatialAudioEngine, SceneOutputId) {
        let mut e = SpatialAudioEngine::new(0, SR, 15.0);
        e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
        let mat = e.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::splat(0.1), Band8::zeros(), Band8::zeros()),
        ));
        let (lo, hi) = ([0.0_f32, 0.0, 0.0], [20.0_f32, 20.0, 30.0]);
        let p = vec![
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ];
        let mut idx: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
        ];
        for t in idx.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        let mut scene = AcousticScene::new();
        scene.add_mesh(AcousticMesh::new(1, p, idx, mat));
        let cfg = CpuSimdConfig { max_reflection_order: 1, max_reflections: 16, ..CpuSimdConfig::default() };
        e.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
        e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
        let s = e.load_source(SourceConfig { path: "imp.wav".into(), channels: 1 }).expect("source");
        let o = e.add_scene_output(SceneOutputConfig::new(SRC, Movability::Static));
        e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
        e.add_listener(ListenerConfig {
            position: LIS,
            heading: [0.0, 0.0, 1.0],
            physical_layout: PhysicalOutputLayout::Stereo,
        });
        e.debug_audio_stage = 3; // direct + early reflections
        (e, o)
    }

    /// Impulse response (L+R) after the engine has settled.
    fn impulse(e: &mut SpatialAudioEngine) -> Vec<f32> {
        e.update_scene_spatial();
        let silence = AudioBuffer::new(1, BLOCK as u16);
        let mut imp = AudioBuffer::new(1, BLOCK as u16);
        imp.set(0, 0, 1.0);
        let mut out = [AudioBuffer::new(2, BLOCK as u16)];
        let mut y = Vec::new();
        for b in 0..(24 + 110) {
            let input = if b == 24 { &imp } else { &silence };
            e.process_audio_scene(&[input], &mut out);
            if b >= 24 {
                for i in 0..BLOCK {
                    y.push(out[0].channel(0)[i] + out[0].channel(1)[i]);
                }
            }
        }
        y
    }

    fn window(y: &[f32], metres: f32) -> f32 {
        let c = (metres * SR / C).round() as usize;
        let (lo, hi) = (c.saturating_sub(48), (c + 48).min(y.len()));
        y[lo..hi].iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn a_horn_keeps_the_on_axis_reflection_and_cuts_the_one_behind_the_speaker() {
        // Far wall (z = 30): path 22 + 5 = 27 m, leaves the emitter on axis (toward +Z).
        // Rear wall (z = 0): path 8 + 25 = 33 m, leaves it at 180 degrees.
        let omni = {
            let (mut e, _) = room_engine();
            impulse(&mut e)
        };
        let horn = {
            let (mut e, o) = room_engine();
            e.set_scene_output_directivity(o, Some([0.0, 0.0, 1.0]), 0.0);
            e.set_scene_output_pattern(o, Some(EmitterPattern::horn(60.0, 40.0)));
            impulse(&mut e)
        };
        let far = db(window(&horn, 27.0), window(&omni, 27.0));
        let rear = db(window(&horn, 33.0), window(&omni, 33.0));
        assert!(window(&omni, 27.0) > 1e-4 && window(&omni, 33.0) > 1e-4, "reflections must exist");
        assert!(far.abs() < 1.0, "on-axis far-wall reflection must be unchanged: {far} dB");
        assert!(rear < -6.0, "rear-wall reflection must be cut by the horn's rear: {rear} dB");
        // And the direct sound (17 m, on axis) is unchanged.
        let direct = db(window(&horn, 17.0), window(&omni, 17.0));
        assert!(direct.abs() < 1.0, "on-axis direct sound must be unchanged: {direct} dB");
    }
}
