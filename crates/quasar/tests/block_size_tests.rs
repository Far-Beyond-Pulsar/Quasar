//! The renderer must produce the same signal whatever block size the device callback uses.
//!
//! Regression: the per-output scratch buffers were fixed at the 256-sample capacity, so a
//! shorter block (e.g. the 224-frame tail of a 480-frame WASAPI callback) pushed zero padding into
//! the direct-path delay line and the early-reflection line every callback: dropouts in the
//! direct sound while the reverb tail stayed intact ("only echoes").

use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const FREQ: f64 = 700.0;

/// Render `total` frames of a steady tone with the given repeating block pattern; returns the
/// left channel.
fn render(stage: u8, pattern: &[usize], total: usize) -> Vec<f32> {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "t.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([-3.0, 0.0, -4.0], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig { position: [0.0; 3], heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.update_scene_spatial();
    e.debug_audio_stage = stage;
    let (mut left, mut pos, mut k) = (Vec::new(), 0usize, 0usize);
    while pos < total {
        let block = pattern[k % pattern.len()].min(total - pos);
        k += 1;
        let mut input = AudioBuffer::new(1, block as u16);
        for i in 0..block {
            input.set(0, i as u16, (0.5 * (2.0 * std::f64::consts::PI * FREQ * (pos + i) as f64 / SR as f64).sin()) as f32);
        }
        let mut out = AudioBuffer::new(2, block as u16);
        e.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
        left.extend_from_slice(&out.channel(0)[..block]);
        pos += block;
    }
    left
}

#[test]
fn direct_sound_is_identical_for_any_block_size() {
    let total = 12_000;
    let reference = render(2, &[256], total);
    let peak = reference[4_000..].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.05, "the direct sound must be present: {peak}");
    for pattern in [&[64usize][..], &[224, 256], &[173, 91, 256, 17], &[224, 224, 33]] {
        let y = render(2, pattern, total);
        let err = y.iter().zip(&reference).skip(2_000).fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "block pattern {pattern:?}: max deviation {err} from the 256-block render");
    }
}

#[test]
fn early_line_time_base_is_independent_of_block_size() {
    // Stage 3 adds the early-reflection taps (none with the stub backend), so this mainly pins
    // that the dry delay line used by every stage advances by the block, not by the capacity.
    let a = render(3, &[256], 9_000);
    let b = render(3, &[100, 77], 9_000);
    let err = a.iter().zip(&b).skip(2_000).fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(err < 1e-4, "deviation {err}");
}
