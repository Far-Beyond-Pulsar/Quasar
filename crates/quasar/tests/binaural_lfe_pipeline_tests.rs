//! Engine-level tests for the HRTF (binaural) listener path and the LFE bus.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::AcousticScene;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn engine_with_backend() -> SpatialAudioEngine {
    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(
        AcousticScene::new(),
        CpuSimdConfig::default(),
    )));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    engine
}

/// Render `blocks` blocks; `src_of(block)` supplies the mono source block.
/// Returns one sample stream per output channel.
fn render_stream(
    engine: &mut SpatialAudioEngine,
    out_channels: u16,
    blocks: usize,
    src_of: impl Fn(usize) -> AudioBuffer,
) -> Vec<Vec<f32>> {
    let mut out = AudioBuffer::new(out_channels, BLOCK as u16);
    let mut streams = vec![Vec::new(); out_channels as usize];
    for b in 0..blocks {
        let src = src_of(b);
        out.clear();
        engine.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        for ch in 0..out_channels {
            streams[ch as usize].extend_from_slice(out.channel(ch));
        }
    }
    streams
}

/// Phase-continuous mono sine block `b` (block index) at `freq` Hz.
fn sine_block(freq: f32, b: usize, amp: f32) -> AudioBuffer {
    let mut buf = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        let n = (b * BLOCK + i) as f32;
        buf.set(0, i as u16, amp * (2.0 * std::f32::consts::PI * freq * n / SR).sin());
    }
    buf
}

fn rms_of(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn source() -> SourceConfig {
    SourceConfig { path: "mono.wav".to_string(), channels: 1 }
}

fn engine_with(pos: [f32; 3], layout: PhysicalOutputLayout) -> (SpatialAudioEngine, SceneOutputId) {
    let mut engine = engine_with_backend();
    let src = engine.load_source(source()).expect("load source");
    let out = engine.add_scene_output(SceneOutputConfig::new(pos, Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: layout,
    });
    (engine, out)
}

// ── HRTF listener (parametric binaural path) ─────────────────────────

fn hrtf_engine(pos: [f32; 3]) -> SpatialAudioEngine {
    let (mut engine, _) = engine_with(pos, PhysicalOutputLayout::Hrtf);
    engine.update_scene_spatial();
    engine
}

#[test]
fn hrtf_listener_renders_two_channels_lateralised_louder_and_earlier() {
    // Level: 4 kHz tone from the right. Head shadow makes the right ear louder.
    let mut engine = hrtf_engine([2.0, 0.0, 0.0]);
    let s = render_stream(&mut engine, 2, 40, |b| sine_block(4000.0, b, 0.5));
    assert_eq!(s.len(), 2, "Hrtf listener must produce 2 channels");
    let tail = |x: &Vec<f32>| rms_of(&x[x.len() - 10 * BLOCK..]);
    let (l, r) = (tail(&s[0]), tail(&s[1]));
    assert!(l.is_finite() && r.is_finite() && r > 1e-4, "no audio (L={l}, R={r})");
    assert!(r > l * 1.5, "source at the right must be clearly louder on the right ear (L={l}, R={r})");

    // Timing: a single click from the right arrives at the right ear first (ITD ~ 31 samples).
    let mut engine = hrtf_engine([2.0, 0.0, 0.0]);
    let s = render_stream(&mut engine, 2, 26, |b| {
        let mut buf = AudioBuffer::new(1, BLOCK as u16);
        if b == 20 {
            buf.set(0, 5, 1.0);
        }
        buf
    });
    let argmax = |x: &[f32]| {
        let mut best = 20 * BLOCK;
        for i in 20 * BLOCK..x.len() {
            if x[i].abs() > x[best].abs() {
                best = i;
            }
        }
        best as i32
    };
    let lead = argmax(&s[0]) - argmax(&s[1]);
    assert!((15..=50).contains(&lead), "right ear must lead by ~31 samples, measured {lead}");

    // Mirror: emitter on the left flips the level cue.
    let mut engine = hrtf_engine([-2.0, 0.0, 0.0]);
    let s = render_stream(&mut engine, 2, 40, |b| sine_block(4000.0, b, 0.5));
    assert!(tail(&s[0]) > tail(&s[1]) * 1.5, "source at the left must be louder on the left ear");
}

#[test]
fn hrtf_listener_follows_heading() {
    // Emitter to the world +X; listener turned to face +X: the source is now ahead,
    // so the two ears must be (nearly) balanced.
    let (mut engine, _) = engine_with([2.0, 0.0, 0.0], PhysicalOutputLayout::Hrtf);
    let id = quasar_audio::quasar_core::scene_output::ListenerId(0);
    engine.update_listener(id, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    engine.update_scene_spatial();
    let s = render_stream(&mut engine, 2, 40, |b| sine_block(4000.0, b, 0.5));
    let tail = |x: &Vec<f32>| rms_of(&x[x.len() - 10 * BLOCK..]);
    let (l, r) = (tail(&s[0]), tail(&s[1]));
    assert!(r > 1e-4 && (l / r - 1.0).abs() < 0.1, "facing the source must balance the ears (L={l}, R={r})");
}

// ── LFE bus ──────────────────────────────────────────────────────────

/// Engine with one emitter 3 m ahead and the given listener layout (5.1 / 7.1: LFE = slot 3).
fn lfe_engine(layout: PhysicalOutputLayout, send: Option<f32>) -> (SpatialAudioEngine, SceneOutputId) {
    let (mut engine, out) = engine_with([0.0, 0.0, -3.0], layout);
    if let Some(g) = send {
        engine.set_scene_output_lfe_send(out, g);
    }
    engine.update_scene_spatial();
    (engine, out)
}

#[test]
fn lfe_send_defaults_to_zero_and_leaves_the_mix_untouched() {
    for (layout, ch) in [(PhysicalOutputLayout::Surround51, 6u16), (PhysicalOutputLayout::Surround714, 8)] {
        let src = |b: usize| sine_block(60.0, b, 0.5);
        let (mut default_engine, out) = lfe_engine(layout.clone(), None);
        assert_eq!(default_engine.scene_output_lfe_send(out), 0.0);
        let (mut zero_engine, _) = lfe_engine(layout.clone(), Some(0.0));
        let (mut sent_engine, _) = lfe_engine(layout, Some(1.0));
        let a = render_stream(&mut default_engine, ch, 30, src);
        let z = render_stream(&mut zero_engine, ch, 30, src);
        let s = render_stream(&mut sent_engine, ch, 30, src);
        assert_eq!(a, z, "explicit zero send must be bit-identical to the default");
        assert!(a[3].iter().all(|v| *v == 0.0), "LFE must stay silent with no send");
        assert!(a[2].iter().any(|v| v.abs() > 1e-3), "front centre must still carry the panned mix");
        for c in (0..ch as usize).filter(|c| *c != 3) {
            assert_eq!(a[c], s[c], "an LFE send must only touch the LFE slot (channel {c})");
        }
        assert!(rms_of(&s[3][s[3].len() / 2..]) > 1e-3, "send 1.0 must feed the LFE channel");
    }
}

#[test]
fn lfe_bus_low_passes_the_send() {
    // 4th-order Butterworth @ 120 Hz: ~0 dB at 60 Hz, about -74 dB at 1 kHz (theory).
    // Require a conservative 40 dB gap through the whole engine.
    let level = |freq: f32| {
        let (mut engine, _) = lfe_engine(PhysicalOutputLayout::Surround51, Some(1.0));
        let s = render_stream(&mut engine, 6, 60, |b| sine_block(freq, b, 0.5));
        rms_of(&s[3][s[3].len() / 2..])
    };
    let low = level(60.0);
    let high = level(1000.0);
    assert!(low > 1e-3, "60 Hz must reach the LFE channel (rms {low})");
    let margin_db = 20.0 * (low / high.max(1e-12)).log10();
    assert!(margin_db > 40.0, "60 Hz vs 1 kHz margin only {margin_db} dB (low={low}, high={high})");
}

#[test]
fn lfe_send_is_ignored_by_layouts_without_an_lfe_slot() {
    let src = |b: usize| sine_block(60.0, b, 0.5);
    let (mut a, _) = lfe_engine(PhysicalOutputLayout::Stereo, None);
    let (mut b, _) = lfe_engine(PhysicalOutputLayout::Stereo, Some(1.0));
    assert_eq!(render_stream(&mut a, 2, 20, src), render_stream(&mut b, 2, 20, src));
}

#[test]
fn lfe_send_survives_registry_changes() {
    // The send lives beside the registry, so adding/removing outputs keeps indices aligned.
    let (mut engine, out) = lfe_engine(PhysicalOutputLayout::Surround51, Some(0.5));
    let second = engine.add_scene_output(SceneOutputConfig::new([1.0, 0.0, -2.0], Movability::Static));
    assert_eq!(engine.scene_output_lfe_send(out), 0.5);
    assert_eq!(engine.scene_output_lfe_send(second), 0.0);
    engine.remove_scene_output(out);
    assert_eq!(engine.scene_output_lfe_send(SceneOutputId(0)), 0.0);
}
