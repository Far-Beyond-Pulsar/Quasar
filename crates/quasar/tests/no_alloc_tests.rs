//! #79: the audio-thread entry points allocate nothing.
//!
//! Uses the counting global allocator from `common` (own test binary, per-thread counting).

mod common;

use common::count_allocs;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::backend::SpatialQuery;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::occlusion::AirAbsorptionOcclusionNode;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn noise(seed: u32) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    let mut s = seed;
    for i in 0..BLOCK {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        b.set(0, i as u16, (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5);
    }
    b
}

/// Legacy `process_audio` + `AudioNodeGraph::process` (two sources, one routed connection).
#[test]
fn legacy_process_audio_does_not_allocate() {
    let mut engine = SpatialAudioEngine::new(2, SR, 15.0);
    engine.set_backend(Box::new(HardwareAcceleratorStub::new()));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    {
        let g = engine.dsp_graph();
        let a = g.add_node(Box::new(AirAbsorptionOcclusionNode::new(1, SR, 0.5)));
        let b = g.add_node(Box::new(AirAbsorptionOcclusionNode::new(1, SR, 0.5)));
        g.connect_with_source(a, 0, b, 0, 0.5, 1);
    }
    for id in 0..2u32 {
        engine.update_spatial(&SpatialQuery {
            source_position: [id as f32 * 3.0 - 1.0, 0.0, -4.0],
            listener_position: [0.0; 3],
            source_id: id,
        });
    }
    let (in0, in1) = (noise(1), noise(2));
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    // Warm-up (first block retargets the crossfaders).
    engine.process_audio(&[&in0, &in1], &mut out);
    let (_, n) = count_allocs(|| {
        for _ in 0..50 {
            engine.process_audio(&[&in0, &in1], &mut out);
        }
    });
    assert_eq!(n, 0, "legacy process_audio allocated {n} time(s)");
    assert!(out.peak() > 0.0, "the legacy path must still produce audio");
}

/// The scene path (`process_audio_scene`) is allocation-free too.
#[test]
fn scene_process_does_not_allocate() {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let src = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).expect("source");
    let out = e.add_scene_output(SceneOutputConfig::new([2.0, 0.0, -3.0], Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    e.add_listener(ListenerConfig {
        position: [0.0; 3],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e.update_scene_spatial();
    let input = noise(3);
    let mut o = AudioBuffer::new(2, BLOCK as u16);
    e.process_audio_scene(&[&input], std::slice::from_mut(&mut o));
    let mut n = 0;
    for round in 0..10 {
        // Compute thread (not counted): move the emitter and publish new coefficients, so the
        // audio thread keeps re-targeting its crossfaders inside the counted region.
        e.set_scene_output_position(out, [2.0 - round as f32 * 0.5, 0.0, -3.0]);
        e.update_scene_spatial();
        let (_, k) = count_allocs(|| {
            for _ in 0..5 {
                e.process_audio_scene(&[&input], std::slice::from_mut(&mut o));
            }
        });
        n += k;
    }
    assert_eq!(n, 0, "process_audio_scene allocated {n} time(s)");
    assert!(o.peak() > 0.0);
}

/// Sanity check of the harness itself: it must actually see allocations.
#[test]
fn harness_detects_allocations() {
    let (v, n) = count_allocs(|| {
        let mut v = Vec::new();
        v.push(1u32);
        v
    });
    assert!(n >= 1 && v.len() == 1);
}
