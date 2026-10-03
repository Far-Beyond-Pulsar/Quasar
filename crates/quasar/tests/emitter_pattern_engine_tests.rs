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
