//! #85: engine-level bass management (LR4 crossover, small-speaker bass to the LFE channel).

mod common;

use common::count_allocs;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::bass_management::{BassError, BassManagementConfig};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const LFE: usize = 3;
/// 51 layout: FL is at -30 degrees; the emitter exactly there lands on FL alone.
const FL_POS: [f32; 3] = [-0.5, 0.0, -0.866_025_4];
const FR_POS: [f32; 3] = [0.5, 0.0, -0.866_025_4];

fn engine(layout: PhysicalOutputLayout, pos: [f32; 3]) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    e.debug_audio_stage = 2;
    let s = e.load_source(SourceConfig { path: "s.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new(pos, Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig { position: [0.0; 3], heading: [0.0, 0.0, -1.0], physical_layout: layout });
    e.update_scene_spatial();
    e
}

fn tone_block(phase: &mut f64, hz: f64, amp: f32) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        b.set(0, i as u16, amp * phase.sin() as f32);
        *phase += 2.0 * std::f64::consts::PI * hz / SR as f64;
    }
    b
}

/// Render `blocks` blocks of a sine; returns every channel's samples of the last `keep` blocks.
fn run(e: &mut SpatialAudioEngine, hz: f64, blocks: usize, keep: usize) -> Vec<Vec<f32>> {
    let mut ph = 0.0;
    let mut cap = vec![Vec::new(); 6];
    for blk in 0..blocks {
        let mut out = [AudioBuffer::new(6, BLOCK as u16)];
        e.process_audio_scene(&[&tone_block(&mut ph, hz, 0.4)], &mut out);
        if blk >= blocks - keep {
            for c in 0..6 {
                cap[c].extend_from_slice(out[0].channel(c as u16));
            }
        }
    }
    cap
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / x.len().max(1) as f64).sqrt() as f32
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

fn cfg_all_small() -> BassManagementConfig {
    BassManagementConfig::all_small(6, &[LFE])
}

#[test]
fn default_is_bit_identical_and_the_lfe_is_silent() {
    let mut plain = engine(PhysicalOutputLayout::Surround51, FL_POS);
    let mut none = engine(PhysicalOutputLayout::Surround51, FL_POS);
    none.set_listener_bass_management(ListenerId(0), None).unwrap();
    let a = run(&mut plain, 60.0, 30, 10);
    let b = run(&mut none, 60.0, 30, 10);
    assert_eq!(a, b);
    assert!(a[LFE].iter().all(|&s| s == 0.0));
    assert!(rms(&a[0]) > 0.1, "the test source reaches FL");
}

#[test]
fn small_speaker_bass_moves_to_the_lfe_and_the_coherent_sum_is_flat() {
    for &hz in &[30.0, 60.0, 80.0, 120.0, 400.0, 2000.0] {
        let mut reference = engine(PhysicalOutputLayout::Surround51, FL_POS);
        let mut managed = engine(PhysicalOutputLayout::Surround51, FL_POS);
        managed.set_listener_bass_management(ListenerId(0), Some(cfg_all_small())).unwrap();
        let r = run(&mut reference, hz, 120, 60);
        let m = run(&mut managed, hz, 120, 60);
        let sum: Vec<f32> = m[0].iter().zip(&m[LFE]).map(|(a, b)| a + b).collect();
        let dev = db(rms(&sum)) - db(rms(&r[0]));
        eprintln!(
            "{hz:>6} Hz: FL ref {:.2} dB | FL {:.2} dB, LFE {:.2} dB | coherent sum deviation {dev:+.3} dB",
            db(rms(&r[0])),
            db(rms(&m[0])),
            db(rms(&m[LFE]))
        );
        assert!(dev.abs() < 0.2, "{hz} Hz: coherent sum off by {dev} dB");
        // Other channels are not touched by the stage (FL source: FR etc. stay silent).
        for c in [1usize, 2, 4, 5] {
            assert!(rms(&m[c]) < 1e-3, "channel {c} got {}", db(rms(&m[c])));
        }
    }
    // Band split: 30 Hz is in the LFE and gone from FL; 2 kHz is only in FL.
    let mut e = engine(PhysicalOutputLayout::Surround51, FL_POS);
    e.set_listener_bass_management(ListenerId(0), Some(cfg_all_small())).unwrap();
    let low = run(&mut e, 30.0, 120, 60);
    assert!(db(rms(&low[LFE])) > db(rms(&low[0])) + 15.0);
    let mut e = engine(PhysicalOutputLayout::Surround51, FL_POS);
    e.set_listener_bass_management(ListenerId(0), Some(cfg_all_small())).unwrap();
    let high = run(&mut e, 2000.0, 120, 60);
    assert!(db(rms(&high[LFE])) < db(rms(&high[0])) - 60.0);
}

#[test]
fn large_speakers_are_bit_identical_and_get_no_redirect() {
    let mut cfg = cfg_all_small();
    cfg.small[1] = false; // FR is large
    let mut reference = engine(PhysicalOutputLayout::Surround51, FR_POS);
    let mut managed = engine(PhysicalOutputLayout::Surround51, FR_POS);
    managed.set_listener_bass_management(ListenerId(0), Some(cfg)).unwrap();
    let r = run(&mut reference, 50.0, 60, 20);
    let m = run(&mut managed, 50.0, 60, 20);
    assert_eq!(r[1], m[1], "a large speaker must be bit-identical");
    eprintln!("LFE level with only a large speaker active: {:.1} dB", db(rms(&m[LFE])));
    assert!(db(rms(&m[LFE])) < -80.0, "nothing to redirect from a large speaker");
}

#[test]
fn unsupported_configurations_are_errors() {
    let mut stereo = engine(PhysicalOutputLayout::Stereo, FL_POS);
    assert_eq!(
        stereo.set_listener_bass_management(ListenerId(0), Some(BassManagementConfig::all_small(2, &[]))),
        Err(BassError::NoLfeChannel)
    );
    let mut e = engine(PhysicalOutputLayout::Surround51, FL_POS);
    let mut bad = cfg_all_small();
    bad.small.pop();
    assert!(matches!(e.set_listener_bass_management(ListenerId(0), Some(bad)), Err(BassError::BadSpeakerFlags { .. })));
    let mut bad = cfg_all_small();
    bad.crossover_hz = 1000.0;
    assert!(matches!(e.set_listener_bass_management(ListenerId(0), Some(bad)), Err(BassError::BadCrossover(_))));
    // Nothing was installed.
    let mut plain = engine(PhysicalOutputLayout::Surround51, FL_POS);
    assert_eq!(run(&mut e, 60.0, 20, 5), run(&mut plain, 60.0, 20, 5));
}

#[test]
fn bass_management_does_not_allocate_on_the_audio_thread() {
    let mut e = engine(PhysicalOutputLayout::Surround51, FL_POS);
    let mut renderer = e.audio_handle();
    let mut ph = 0.0;
    let mut out = [AudioBuffer::new(6, BLOCK as u16)];
    for _ in 0..4 {
        renderer.process_audio_scene(&[&tone_block(&mut ph, 50.0, 0.4)], &mut out);
    }
    e.set_listener_bass_management(ListenerId(0), Some(cfg_all_small())).unwrap();
    let inputs: Vec<AudioBuffer> = (0..60).map(|_| tone_block(&mut ph, 50.0, 0.4)).collect();
    let ((), n) = count_allocs(|| {
        for inp in &inputs {
            renderer.process_audio_scene(&[inp], &mut out);
        }
    });
    assert_eq!(n, 0, "audio thread allocated {n} times (install + processing)");
    // Reconfigure (replaces the stage, adopting its state) and disable.
    let mut cfg = cfg_all_small();
    cfg.crossover_hz = 120.0;
    e.set_listener_bass_management(ListenerId(0), Some(cfg)).unwrap();
    let ((), n1) = count_allocs(|| {
        for inp in &inputs {
            renderer.process_audio_scene(&[inp], &mut out);
        }
    });
    e.set_listener_bass_management(ListenerId(0), None).unwrap();
    let ((), n2) = count_allocs(|| {
        for inp in &inputs {
            renderer.process_audio_scene(&[inp], &mut out);
        }
    });
    assert_eq!((n1, n2), (0, 0));
    assert!(out[0].channel(LFE as u16).iter().all(|&s| s == 0.0), "after the ramp-out the LFE is silent again");
}
