//! #75: the audio thread owns an `AudioRenderer` and never takes a lock shared with the compute
//! / configuration side; configuration reaches it through a lock-free SPSC command queue.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::count_alloc_free;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId, SourceConfig,
    SourceId,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::{AudioRenderer, SpatialAudioEngine};

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn build_engine() -> (SpatialAudioEngine, SourceId, [SceneOutputId; 2]) {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "s.wav".into(), channels: 2 }).expect("source");
    let o0 = e.add_scene_output(SceneOutputConfig::new([2.0, 0.0, -3.0], Movability::Static));
    let o1 = e.add_scene_output(SceneOutputConfig::new([-3.0, 1.0, -2.0], Movability::Static));
    e.connect_pull(o0, ChannelPull::new(s, 0, 0.0));
    e.connect_pull(o1, ChannelPull::new(s, 1, -3.0));
    e.add_listener(ListenerConfig {
        position: [0.0; 3],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e.update_scene_spatial();
    (e, s, [o0, o1])
}

fn sine(phase: &mut f32, freq: f32, amp: f32) -> AudioBuffer {
    let mut b = AudioBuffer::new(2, BLOCK as u16);
    for i in 0..BLOCK {
        let v = amp * (*phase).sin();
        b.set(0, i as u16, v);
        b.set(1, i as u16, -v * 0.5);
        *phase += std::f32::consts::TAU * freq / SR;
    }
    *phase = phase.rem_euclid(std::f32::consts::TAU);
    b
}

#[test]
fn renderer_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<AudioRenderer>();
}

#[test]
#[should_panic(expected = "split off")]
fn combined_process_panics_after_split() {
    let (mut e, _, _) = build_engine();
    let _r = e.audio_handle();
    let mut out = vec![AudioBuffer::new(2, BLOCK as u16)];
    e.process_audio_scene(&[], &mut out);
}

/// Split and combined engines, driven identically (including live edits), render bit-identical
/// audio: the command queue changes WHERE edits are applied, not WHAT they do.
#[test]
fn split_engine_matches_combined_engine_bit_for_bit() {
    let (mut comb, s, [o0, _o1]) = build_engine();
    let (mut split, _, _) = build_engine();
    let mut r = split.audio_handle();

    let (mut pc, mut ps) = (0.0f32, 0.0f32);
    let mut oc = vec![AudioBuffer::new(2, BLOCK as u16)];
    let mut os = vec![AudioBuffer::new(2, BLOCK as u16)];
    for blk in 0..120 {
        match blk {
            20 => {
                comb.set_pull_gain(o0, s, 0, -9.0);
                split.set_pull_gain(o0, s, 0, -9.0);
            }
            40 => {
                comb.disconnect_pull(o0, s, 0);
                split.disconnect_pull(o0, s, 0);
            }
            50 => {
                comb.connect_pull(o0, ChannelPull::new(s, 1, 0.0));
                split.connect_pull(o0, ChannelPull::new(s, 1, 0.0));
            }
            60 => {
                for e in [&mut comb, &mut split] {
                    e.add_listener(ListenerConfig {
                        position: [1.0, 0.0, 0.0],
                        heading: [1.0, 0.0, 0.0],
                        physical_layout: PhysicalOutputLayout::Surround51,
                    });
                    e.update_scene_spatial();
                }
                oc.push(AudioBuffer::new(6, BLOCK as u16));
                os.push(AudioBuffer::new(6, BLOCK as u16));
            }
            80 => {
                for e in [&mut comb, &mut split] {
                    let n = e.add_scene_output(SceneOutputConfig::new([0.0, 2.0, -4.0], Movability::Static));
                    e.connect_pull(n, ChannelPull::new(s, 0, -6.0));
                    e.update_scene_spatial();
                }
            }
            100 => {
                comb.update_listener(ListenerId(0), [0.0; 3], [1.0, 0.0, -1.0]);
                split.update_listener(ListenerId(0), [0.0; 3], [1.0, 0.0, -1.0]);
                comb.remove_scene_output(SceneOutputId(1));
                split.remove_scene_output(SceneOutputId(1));
            }
            _ => {}
        }
        let (ic, is) = (sine(&mut pc, 330.0, 0.4), sine(&mut ps, 330.0, 0.4));
        comb.process_audio_scene(&[&ic], &mut oc);
        r.process_audio_scene(&[&is], &mut os);
        for (l, (a, b)) in oc.iter().zip(os.iter()).enumerate() {
            for ch in 0..a.channels() {
                assert_eq!(a.channel(ch), b.channel(ch), "block {blk}, listener {l}, channel {ch}");
            }
        }
    }
    assert_eq!(r.num_listeners(), 2);
    assert_eq!(r.num_outputs(), 2);
}

/// THE acceptance test: the compute side sits on a lock (busy for 100 ms) and keeps editing, while
/// the audio thread, which holds only the `AudioRenderer`, keeps producing blocks on time and the
/// signal stays continuous.
#[test]
fn audio_thread_is_not_blocked_by_a_busy_compute_side() {
    let (mut engine, s, [o0, o1]) = build_engine();
    engine.set_debug_audio_stage(2); // direct path only: a smooth signal to check continuity on
    let mut renderer = engine.audio_handle();
    let engine = Arc::new(Mutex::new(engine));

    let stop = Arc::new(AtomicBool::new(false));
    let stop_a = stop.clone();
    let audio = std::thread::spawn(move || {
        let mut phase = 0.0f32;
        let mut out = vec![AudioBuffer::new(2, BLOCK as u16)];
        let mut worst = Duration::ZERO;
        let mut blocks = 0usize;
        let mut max_step = 0.0f32;
        let mut last = [0.0f32; 2];
        let mut peak = 0.0f32;
        while !stop_a.load(Ordering::Relaxed) {
            let input = sine(&mut phase, 200.0, 0.5);
            let t = Instant::now();
            renderer.process_audio_scene(&[&input], &mut out);
            worst = worst.max(t.elapsed());
            blocks += 1;
            if blocks > 40 {
                // After the start-up fades.
                for i in 0..BLOCK {
                    for c in 0..2 {
                        let v = out[0].channel(c as u16)[i];
                        max_step = max_step.max((v - last[c]).abs());
                        last[c] = v;
                        peak = peak.max(v.abs());
                    }
                }
            } else {
                last = [out[0].channel(0)[BLOCK - 1], out[0].channel(1)[BLOCK - 1]];
            }
            // Pace like a real device (a block is 5.3 ms at 48 kHz; run 4x faster to fit more).
            std::thread::sleep(Duration::from_micros(1300));
        }
        (worst, blocks, max_step, peak)
    });

    std::thread::sleep(Duration::from_millis(100)); // let the audio thread warm up
    {
        // Hold the compute-side lock and burn CPU for 100 ms, editing meanwhile.
        let mut guard = engine.lock().unwrap();
        let t0 = Instant::now();
        let mut i = 0u32;
        while t0.elapsed() < Duration::from_millis(100) {
            if i % 20_000 == 0 {
                guard.set_pull_gain(o0, s, 0, -(i as f32 % 7.0));
                guard.update_scene_spatial();
            }
            std::hint::black_box(i.wrapping_mul(2654435761));
            i = i.wrapping_add(1);
        }
        guard.disconnect_pull(o1, s, 1);
        guard.connect_pull(o1, ChannelPull::new(s, 1, -3.0));
    }
    std::thread::sleep(Duration::from_millis(100));
    stop.store(true, Ordering::Relaxed);
    let (worst, blocks, max_step, peak) = audio.join().unwrap();

    eprintln!("worst block {worst:?} over {blocks} blocks, max step {max_step}");
    assert!(blocks > 80, "the audio thread kept producing blocks during the 100 ms hold ({blocks})");
    assert!(
        worst < Duration::from_millis(20),
        "a block took {worst:?}: the audio thread must never wait on the compute side (hold was 100 ms)"
    );
    assert!(peak > 0.01, "audio is audible");
    // 200 Hz at 0.5 amplitude through the (<= 1) pan gains: <= 0.5*2*pi*200/48000 per sample = 0.013.
    assert!(max_step < 0.03, "output stayed continuous, max sample step {max_step}");
}

/// The audio thread allocates and frees NOTHING while commands (gain edits, connects, structural
/// adds and removes) are being applied; retired DSP state is dropped by the compute side.
#[test]
fn render_path_and_command_application_do_not_allocate_or_free() {
    let (mut engine, s, [o0, _o1]) = build_engine();
    let mut renderer = engine.audio_handle();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_a = stop.clone();
    let audio = std::thread::spawn(move || {
        let mut phase = 0.0f32;
        let mut out = vec![AudioBuffer::new(2, BLOCK as u16)];
        // Warm up (first blocks apply the setup commands; allocation there would also count, but
        // keep the measured region to the steady state + the live edits below).
        for _ in 0..4 {
            let input = sine(&mut phase, 440.0, 0.3);
            renderer.process_audio_scene(&[&input], &mut out);
        }
        let ((), allocs, frees) = count_alloc_free(|| {
            while !stop_a.load(Ordering::Relaxed) {
                let input_phase = phase;
                // Build the input without allocating (stack buffer).
                let mut b = AudioBuffer::new(2, BLOCK as u16);
                for i in 0..BLOCK {
                    let v = 0.3 * (input_phase + std::f32::consts::TAU * 440.0 * i as f32 / SR).sin();
                    b.set(0, i as u16, v);
                    b.set(1, i as u16, v);
                }
                phase = (input_phase + std::f32::consts::TAU * 440.0 * BLOCK as f32 / SR).rem_euclid(std::f32::consts::TAU);
                renderer.process_audio_scene(&[&b], &mut out);
                std::thread::sleep(Duration::from_micros(500));
            }
        });
        (allocs, frees, renderer.num_outputs(), renderer.num_listeners())
    });

    std::thread::sleep(Duration::from_millis(50));
    for round in 0..3 {
        engine.set_pull_gain(o0, s, 0, -6.0 - round as f32);
        engine.disconnect_pull(o0, s, 0);
        engine.connect_pull(o0, ChannelPull::new(s, 0, 0.0));
        let n = engine.add_scene_output(SceneOutputConfig::new([1.0, 1.0, -5.0], Movability::Static));
        engine.connect_pull(n, ChannelPull::new(s, 1, 0.0));
        engine.update_scene_spatial();
        std::thread::sleep(Duration::from_millis(30));
        let l = engine.add_listener(ListenerConfig {
            position: [0.5, 0.0, 0.0],
            heading: [0.0, 0.0, -1.0],
            physical_layout: PhysicalOutputLayout::Hrtf,
        });
        engine.update_scene_spatial();
        std::thread::sleep(Duration::from_millis(30));
        engine.remove_listener(l);
        engine.remove_scene_output(n);
        std::thread::sleep(Duration::from_millis(30));
        engine.reap_retired();
    }
    std::thread::sleep(Duration::from_millis(30));
    stop.store(true, Ordering::Relaxed);
    let (allocs, frees, n_out, n_lis) = audio.join().unwrap();
    assert_eq!(allocs, 0, "the audio thread allocated {allocs} time(s)");
    assert_eq!(frees, 0, "the audio thread freed memory {frees} time(s)");
    assert_eq!((n_out, n_lis), (2, 1), "all structural commands were applied");
    engine.reap_retired();
    assert_eq!(engine.garbage_overflow_count(), 0);
}
