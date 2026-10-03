//! #119: a pair renders silence until its first REAL coefficients are published, then
//! snaps to them (no blip or delay glide from the defaults).

use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn engine() -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let src = e.load_source(SourceConfig { path: "dc.wav".into(), channels: 1 }).expect("source");
    // 16 m ahead: direct delay 16 * 48000 / 343 = 2239 samples (8.7 blocks).
    let out = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, -16.0], Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    e.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e
}

fn dc() -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        b.set(0, i as u16, 0.5);
    }
    b
}

#[test]
fn silent_before_the_first_update_and_no_default_energy_after_it() {
    let mut e = engine();
    let src = dc();
    let mut out = AudioBuffer::new(2, BLOCK as u16);

    // No update published yet: silence (the default coefficients are NOT audible).
    for _ in 0..4 {
        out.clear();
        e.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        assert_eq!(out.peak(), 0.0, "pair without real coefficients must be silent");
    }

    // First update: the first blocks have no default-coefficient energy. With the default
    // delay of 0 the DC would sound immediately; with the real 2239-sample propagation delay
    // (snapped, not glided from 0) nothing may be heard before the sound arrives.
    e.update_scene_spatial();
    for b in 0..8 {
        out.clear();
        e.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        assert_eq!(out.peak(), 0.0, "block {b} after the first update must be silent (delay not elapsed)");
    }
    // ... and the sound does arrive afterwards, at roughly the stub's distance gain 1/16.
    let mut peak = 0.0_f32;
    for _ in 0..8 {
        out.clear();
        e.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        peak = peak.max(out.peak());
    }
    assert!(peak > 0.01, "the direct sound must arrive: {peak}");
}
