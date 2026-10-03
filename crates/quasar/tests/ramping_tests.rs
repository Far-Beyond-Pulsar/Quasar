//! Engine-level per-sample ramping (#72): a gain or delay change published by
//! the compute thread must not produce a block-boundary step in the output.
//!
//! A tone is rendered through the direct path while the parameter changes.
//! The largest sample-to-sample step must stay within the tone's own slope
//! (plus the ramp slope), far below what a block-constant parameter would give.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::distance::DistanceModel;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const FREQ: f32 = 500.0;

struct Rig {
    engine: SpatialAudioEngine,
    out: SceneOutputId,
    pos: usize,
}

impl Rig {
    fn new(dist: f32) -> Self {
        let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
        engine.set_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default())));
        engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
        // No distance falloff: gain changes then only come from what the test changes.
        engine.set_distance_model(DistanceModel { rolloff_factor: 0.0, ..DistanceModel::default() });
        let src = engine.load_source(SourceConfig { path: "tone.wav".into(), channels: 1 }).unwrap();
        let out = engine.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Dynamic));
        engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
        engine.add_listener(ListenerConfig {
            position: [0.0, 0.0, -dist],
            heading: [0.0, 0.0, 1.0],
            physical_layout: PhysicalOutputLayout::Stereo,
        });
        engine.update_scene_spatial();
        engine.debug_audio_stage = 2; // direct path only
        Self { engine, out, pos: 0 }
    }

    fn blocks(&mut self, n: usize) -> Vec<f32> {
        let mut left = Vec::new();
        let mut input = AudioBuffer::new(1, BLOCK as u16);
        let mut o = AudioBuffer::new(2, BLOCK as u16);
        for _ in 0..n {
            for i in 0..BLOCK {
                let ph = 2.0 * std::f64::consts::PI * FREQ as f64 * (self.pos + i) as f64 / SR as f64;
                input.set(0, i as u16, ph.sin() as f32);
            }
            self.pos += BLOCK;
            o.clear();
            self.engine.process_audio_scene(&[&input], std::slice::from_mut(&mut o));
            left.extend_from_slice(&o.channel(0)[..BLOCK]);
        }
        left
    }
}

fn max_step(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
}

#[test]
fn gain_change_is_ramped_per_sample() {
    let mut rig = Rig::new(8.0);
    let _ = rig.blocks(40); // settle
    // Drop the level of this output by 20 dB (override: reference distance far away
    // gives a constant, lower gain; here via a steeper rolloff at the same distance).
    let m = DistanceModel { rolloff_factor: 1.0, ..DistanceModel::default() }; // 1/8 at 8 m
    rig.engine.set_scene_output_distance_model(rig.out, Some(m));
    rig.engine.update_scene_spatial();
    // The spatial crossfader spans the measured interval between updates (here 40 blocks
    // ~ 213 ms, capped at 2x the 50 ms minimum fade = 100 ms), so give the glide time to complete before the level check.
    let tail = rig.blocks(60);

    let level_before = 0.707_f32; // unity gain, centre pan
    let natural = level_before * 2.0 * std::f32::consts::PI * FREQ / SR;
    let step = max_step(&tail);
    // A block-constant gain would step by ~(gain change per block) * amplitude ~ 0.1.
    assert!(step <= natural * 1.15, "step {step} vs tone slope {natural}");
    // The level really dropped to 1/8.
    let end = tail[tail.len() - BLOCK..].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!((end - 0.707 / 8.0).abs() < 0.01, "settled amplitude {end}");
}

#[test]
fn delay_change_is_ramped_per_sample() {
    let mut rig = Rig::new(8.0);
    let _ = rig.blocks(40);
    // Move the emitter 0.3 m farther away: the delay grows by ~42 samples over the
    // 15 ms crossfade (slope 0.06 samples/sample, i.e. a Doppler glide).
    rig.engine.set_scene_output_position(rig.out, [0.0, 0.0, 0.3]);
    // (listener at z = -8, emitter now at z = +0.3 -> 8.3 m)
    rig.engine.update_scene_spatial();
    let tail = rig.blocks(12);

    let natural = 0.707 * 2.0 * std::f32::consts::PI * FREQ / SR;
    let slope = 0.06_f32;
    let step = max_step(&tail);
    // Delay ramp adds at most slope * natural to the instantaneous frequency.
    assert!(step <= natural * (1.0 + slope) * 1.1, "step {step} vs tone slope {natural}");
    // Without per-sample ramping the 42-sample change would be applied in ~14-sample
    // jumps per block, each a step of ~14 * natural.
}
