//! #84: engine-level speaker calibration (delay / trim / EQ per physical speaker).

mod common;

use common::count_allocs;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::speaker_calibration::{CalibrationConfig, CalibrationError, EqBand};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const FL_POS: [f32; 3] = [-0.5, 0.0, -0.866_025_4];

fn engine() -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    e.debug_audio_stage = 2;
    let s = e.load_source(SourceConfig { path: "s.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new(FL_POS, Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig {
        position: [0.0; 3],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Surround51,
    });
    e.update_scene_spatial();
    e
}

fn signal(n: usize) -> Vec<f32> {
    // Deterministic broadband-ish signal (two tones + a chirp-like term).
    (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() + 0.2 * (2.0 * std::f64::consts::PI * (900.0 + 2000.0 * t) * t).sin()) as f32
        })
        .collect()
}

/// Render `input` through the engine in blocks, returning the 6 channels.
fn render_all(e: &mut SpatialAudioEngine, input: &[f32], mut hook: impl FnMut(usize, &mut SpatialAudioEngine)) -> Vec<Vec<f32>> {
    let mut cap = vec![Vec::new(); 6];
    for (blk, chunk) in input.chunks(BLOCK).enumerate() {
        hook(blk, e);
        let mut inp = AudioBuffer::new(1, chunk.len() as u16);
        inp.channel_mut(0).copy_from_slice(chunk);
        let mut out = [AudioBuffer::new(6, chunk.len() as u16)];
        e.process_audio_scene(&[&inp], &mut out);
        for c in 0..6 {
            cap[c].extend_from_slice(out[0].channel(c as u16));
        }
    }
    cap
}

#[test]
fn default_and_flat_calibration_are_bit_identical() {
    let x = signal(24_000);
    let mut plain = engine();
    let mut flat = engine();
    flat.set_listener_calibration(ListenerId(0), None).unwrap();
    let a = render_all(&mut plain, &x, |_, _| {});
    let b = render_all(&mut flat, &x, |_, _| {});
    assert_eq!(a, b);
}

#[test]
fn distances_delay_and_trim_the_nearer_speakers() {
    // FL 2.000 m, the other speakers 2.343 m: 0.343 m = exactly 48 samples at 48 kHz.
    let dist = [2.0, 2.343, 2.343, 2.343, 2.343, 2.343];
    let x = signal(40_000);
    let mut plain = engine();
    let mut cal = engine();
    cal.set_listener_speaker_distances(ListenerId(0), &dist).unwrap();
    let r = render_all(&mut plain, &x, |_, _| {});
    let c = render_all(&mut cal, &x, |_, _| {});
    let gain = 2.0f32 / 2.343;
    let mut worst = 0.0f32;
    for n in 20_000..40_000 {
        let want = gain * r[0][n - 48];
        worst = worst.max((c[0][n] - want).abs());
    }
    eprintln!("FL: calibrated vs 48-sample delayed reference x {gain:.4}: max error {worst:.2e} (-1.37 dB expected)");
    assert!(worst < 1e-4, "max error {worst}");
    // Channels that carry nothing stay silent.
    for ch in [1usize, 2, 4, 5] {
        assert!(c[ch].iter().all(|&s| s.abs() < 1e-3));
    }
}

#[test]
fn eq_in_the_engine_changes_the_level_of_the_band() {
    let mut cfg = CalibrationConfig::flat(6);
    cfg.channels[0].eq[0] = EqBand::Peaking { hz: 440.0, q: 2.0, gain_db: 6.0 };
    cfg.ramp_ms = 5.0;
    let x: Vec<f32> = (0..48_000).map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR).sin()).collect();
    let mut plain = engine();
    let mut cal = engine();
    cal.set_listener_calibration(ListenerId(0), Some(cfg)).unwrap();
    let r = render_all(&mut plain, &x, |_, _| {});
    let c = render_all(&mut cal, &x, |_, _| {});
    let rms = |v: &[f32]| (v.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    let db = 20.0 * (rms(&c[0][24_000..]) / rms(&r[0][24_000..])).log10();
    eprintln!("engine EQ bell +6 dB at 440 Hz: measured {db:.3} dB");
    assert!((db - 6.0).abs() < 0.2, "{db}");
}

#[test]
fn invalid_calibration_is_an_error_and_changes_nothing() {
    let mut e = engine();
    assert!(matches!(
        e.set_listener_speaker_distances(ListenerId(0), &[1.0, 2.0, 3.0]),
        Err(CalibrationError::ChannelMismatch { .. })
    ));
    assert!(matches!(
        e.set_listener_speaker_distances(ListenerId(0), &[1.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
        Err(CalibrationError::BadDistance(5))
    ));
    assert!(matches!(
        e.set_listener_speaker_distances(ListenerId(0), &[1.0, 1.0, 1.0, 1.0, 1.0, 50.0]),
        Err(CalibrationError::DelayTooLong { .. })
    ));
    let x = signal(8_000);
    let mut plain = engine();
    assert_eq!(render_all(&mut e, &x, |_, _| {}), render_all(&mut plain, &x, |_, _| {}));
}

#[test]
fn changing_the_calibration_is_click_free() {
    let x: Vec<f32> = (0..48_000).map(|i| 0.4 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / SR).sin()).collect();
    let mut e = engine();
    let c = render_all(&mut e, &x, |blk, e| {
        if blk == 60 {
            e.set_listener_speaker_distances(ListenerId(0), &[1.0, 2.0, 2.0, 2.0, 2.0, 2.0]).unwrap();
        }
    });
    let max_jump = c[0].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    let plain = 0.4 * 2.0 * std::f32::consts::PI * 300.0 / SR;
    eprintln!("max sample jump {max_jump:.4} vs sine slope {plain:.4} (FL is the loudest panned channel)");
    assert!(max_jump < 2.0 * plain, "click {max_jump}");
}

#[test]
fn calibration_does_not_allocate_on_the_audio_thread() {
    let mut e = engine();
    let mut renderer = e.audio_handle();
    let x = signal(BLOCK * 80);
    let blocks: Vec<AudioBuffer> = x
        .chunks(BLOCK)
        .map(|c| {
            let mut b = AudioBuffer::new(1, BLOCK as u16);
            b.channel_mut(0).copy_from_slice(c);
            b
        })
        .collect();
    let mut out = [AudioBuffer::new(6, BLOCK as u16)];
    for b in blocks.iter().take(4) {
        renderer.process_audio_scene(&[b], &mut out);
    }
    let mut cfg = CalibrationConfig::flat(6);
    cfg.channels[0].delay_samples = 100.3;
    cfg.channels[0].eq[1] = EqBand::HighShelf { hz: 5000.0, gain_db: -3.0 };
    cfg.channels[0].highpass_hz = Some(60.0);
    e.set_listener_calibration(ListenerId(0), Some(cfg)).unwrap();
    let ((), n) = count_allocs(|| {
        for b in blocks.iter() {
            renderer.process_audio_scene(&[b], &mut out);
        }
    });
    assert_eq!(n, 0, "audio thread allocated {n} times with calibration (install, fade, steady)");
    e.set_listener_calibration(ListenerId(0), None).unwrap();
    let ((), n) = count_allocs(|| {
        for b in blocks.iter() {
            renderer.process_audio_scene(&[b], &mut out);
        }
    });
    assert_eq!(n, 0);
}
