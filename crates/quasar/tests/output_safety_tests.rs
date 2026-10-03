//! #80: per-listener output safety stage at engine level (ceiling, NaN scrub, latency, meters,
//! FTZ, allocation-free).

mod common;

use common::count_allocs;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::limiter::{flush_to_zero_enabled, OutputSafetyConfig};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const CEIL: f32 = 0.891_250_9; // -1 dBFS

/// One emitter 1 m ahead of a stereo listener (stub backend: unity direct gain).
fn engine(pull_db: f32) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    e.debug_audio_stage = 2;
    let s = e.load_source(SourceConfig { path: "s.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, -1.0], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, pull_db));
    e.add_listener(ListenerConfig {
        position: [0.0; 3],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e.update_scene_spatial();
    e
}

fn sine_block(phase: &mut f32, amp: f32) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        b.set(0, i as u16, amp * phase.sin());
        *phase += std::f32::consts::TAU * 440.0 / SR;
    }
    b
}

fn render(e: &mut SpatialAudioEngine, input: &AudioBuffer) -> AudioBuffer {
    let mut out = [AudioBuffer::new(2, BLOCK as u16)];
    e.process_audio_scene(&[input], &mut out);
    let [o] = out;
    o
}

#[test]
fn an_overdriven_mix_never_exceeds_the_ceiling_and_the_meters_see_it() {
    // +24 dB on a full-scale-ish sine: ~11 x over the ceiling before the stage.
    let mut e = engine(24.0);
    let meter = e.output_meter(ListenerId(0));
    let mut phase = 0.0;
    let mut peak = 0.0f32;
    for _ in 0..60 {
        let out = render(&mut e, &sine_block(&mut phase, 1.0));
        peak = peak.max(out.peak());
    }
    assert!(peak <= CEIL, "output peak {peak} above the -1 dBFS ceiling");
    assert!(peak > 0.8, "the limiter holds the level at the ceiling instead of muting ({peak})");
    assert!(meter.peak() <= CEIL && meter.peak() > 0.8);
    assert!(meter.limited_samples() > 0);
    assert!(meter.gain_reduction_db() > 6.0, "gr {}", meter.gain_reduction_db());
    assert_eq!(meter.hard_clipped_samples(), 0);
    assert_eq!(meter.nonfinite_samples(), 0);
}

#[test]
fn default_stage_is_transparent_below_the_ceiling_and_has_no_latency() {
    let mut on = engine(0.0);
    let mut off = engine(0.0);
    off.set_output_safety(ListenerId(0), OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
    assert_eq!(on.output_latency_samples(ListenerId(0)), 0);
    assert_eq!(off.output_latency_samples(ListenerId(0)), 0);
    let (mut p1, mut p2) = (0.0, 0.0);
    for blk in 0..30 {
        // 0.5 amplitude through the centre pan (0.707 per channel): far below -1 dBFS.
        let (a, b) = (render(&mut on, &sine_block(&mut p1, 0.5)), render(&mut off, &sine_block(&mut p2, 0.5)));
        for c in 0..2 {
            assert_eq!(a.channel(c), b.channel(c), "block {blk}: the default stage altered a quiet signal");
        }
    }
    assert_eq!(on.output_meter(ListenerId(0)).limited_samples(), 0);
}

#[test]
fn look_ahead_latency_reported_equals_measured() {
    let mut a = engine(0.0);
    let mut b = engine(0.0);
    b.set_output_safety(ListenerId(0), OutputSafetyConfig { lookahead_ms: 1.0, ..OutputSafetyConfig::default() });
    assert_eq!(b.output_latency_samples(ListenerId(0)), 48);
    // Impulse after the start-up fades; first non-zero output sample position, plain vs look-ahead.
    let (mut imp, silence) = (AudioBuffer::new(1, BLOCK as u16), AudioBuffer::new(1, BLOCK as u16));
    imp.set(0, 0, 0.5);
    let first = |e: &mut SpatialAudioEngine| {
        for _ in 0..8 {
            render(e, &silence);
        }
        let mut pos = None;
        for blk in 0..6 {
            let out = render(e, if blk == 0 { &imp } else { &silence });
            if pos.is_none() {
                pos = out.channel(0).iter().position(|&v| v != 0.0).map(|p| p + blk * BLOCK);
            }
        }
        pos.expect("impulse heard")
    };
    let (pa, pb) = (first(&mut a), first(&mut b));
    assert_eq!(pb - pa, 48, "measured latency {} vs reported 48", pb - pa);
}

#[test]
fn nan_and_inf_sources_yield_silence_and_count_errors_not_noise() {
    let mut e = engine(0.0);
    let meter = e.output_meter(ListenerId(0));
    let mut phase = 0.0;
    for _ in 0..8 {
        render(&mut e, &sine_block(&mut phase, 0.3));
    }
    assert_eq!(meter.nonfinite_samples(), 0);
    // A poisoned source block: every sample NaN / inf.
    let mut bad = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        bad.set(0, i as u16, if i % 2 == 0 { f32::NAN } else { f32::INFINITY });
    }
    let mut late_max = 0.0f32;
    for blk in 0..8 {
        let out = render(&mut e, &bad);
        assert!(out.channel(0).iter().chain(out.channel(1)).all(|v| v.is_finite()), "NaN / inf reached the output");
        if blk >= 2 {
            // Past the ~140 sample propagation delay only poisoned samples are left.
            late_max = late_max.max(out.peak());
        }
    }
    assert_eq!(late_max, 0.0, "poisoned input must produce silence, not noise");
    assert!(meter.nonfinite_samples() > 0, "error counter must record the scrubbed samples");
    assert!(meter.peak() <= CEIL);
}

#[test]
fn renderer_enables_flush_to_zero_on_the_audio_thread() {
    let mut e = engine(0.0);
    let mut r = e.audio_handle();
    let (ftz, supported) = std::thread::spawn(move || {
        let mut out = [AudioBuffer::new(2, BLOCK as u16)];
        let input = AudioBuffer::new(1, BLOCK as u16);
        r.process_audio_scene(&[&input], &mut out);
        // The meters are readable from the handle too.
        assert!(r.output_meter(0).is_some());
        (flush_to_zero_enabled(), r.prepare_audio_thread())
    })
    .join()
    .unwrap();
    assert_eq!(ftz.is_some(), supported);
    if let Some(on) = ftz {
        assert!(on, "the first block must set FTZ/DAZ on the audio thread");
    }
}

#[test]
fn limiting_path_does_not_allocate() {
    let mut e = engine(24.0);
    e.set_output_safety(ListenerId(0), OutputSafetyConfig { lookahead_ms: 2.0, ..OutputSafetyConfig::default() });
    let mut phase = 0.0;
    let input = sine_block(&mut phase, 1.0);
    let mut out = [AudioBuffer::new(2, BLOCK as u16)];
    for _ in 0..4 {
        e.process_audio_scene(&[&input], &mut out);
    }
    let ((), n) = count_allocs(|| {
        for _ in 0..20 {
            e.process_audio_scene(&[&input], &mut out);
        }
    });
    assert_eq!(n, 0, "the output safety stage allocated");
    assert!(out[0].peak() <= CEIL);
}
