//! #73: patch-bay / registry edits mutate the renderer in place (click-free ramps, no reset of
//! unrelated DSP state, no audio-thread allocation).

mod common;

use common::count_allocs;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId, SourceConfig,
    SourceId,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn new_engine() -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    e
}

fn mono_source(e: &mut SpatialAudioEngine) -> SourceId {
    e.load_source(SourceConfig { path: "s.wav".into(), channels: 1 }).expect("source")
}

fn listener(e: &mut SpatialAudioEngine, pos: [f32; 3]) {
    e.add_listener(ListenerConfig { position: pos, heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
}

fn buf(f: impl Fn(usize) -> f32) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        b.set(0, i as u16, f(i));
    }
    b
}

fn dc(v: f32) -> AudioBuffer {
    buf(|_| v)
}

fn render(e: &mut SpatialAudioEngine, srcs: &[&AudioBuffer], n_lis: usize) -> Vec<AudioBuffer> {
    let mut out: Vec<AudioBuffer> = (0..n_lis).map(|_| AudioBuffer::new(2, BLOCK as u16)).collect();
    e.process_audio_scene(srcs, &mut out);
    out
}

fn same(a: &AudioBuffer, b: &AudioBuffer) -> bool {
    (0..2u16).all(|c| a.channel(c) == b.channel(c))
}

#[test]
fn set_pull_gain_is_click_free_and_default_ramp_is_applied() {
    let mut e = new_engine();
    let s = mono_source(&mut e);
    let o = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, -1.0], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    listener(&mut e, [0.0; 3]);
    e.debug_audio_stage = 2; // direct chain only
    e.update_scene_spatial();
    let input = dc(0.5);
    for _ in 0..8 {
        render(&mut e, &[&input], 1);
    }
    let before = render(&mut e, &[&input], 1);
    let level_before = before[0].channel(0)[BLOCK - 1].abs() + before[0].channel(1)[BLOCK - 1].abs();
    assert!(level_before > 0.1);

    e.set_pull_gain(o, s, 0, -12.0);
    let mut y: Vec<[f32; 2]> = Vec::new();
    for _ in 0..8 {
        let out = render(&mut e, &[&input], 1);
        for i in 0..BLOCK {
            y.push([out[0].channel(0)[i], out[0].channel(1)[i]]);
        }
    }
    // Biggest allowed step: the whole 0 -> -12 dB change spread over the 15 ms ramp, times the
    // input level and the (<= 1) pan gain, with headroom for the band filters' ringing.
    let ramp = 0.015 * SR;
    let bound = (1.0 - 10f32.powf(-12.0 / 20.0)) * 0.5 / ramp * 1.5;
    for ch in 0..2 {
        let mut prev = y[0][ch];
        let mut max_step = 0.0f32;
        for s in &y[1..] {
            max_step = max_step.max((s[ch] - prev).abs());
            prev = s[ch];
        }
        assert!(max_step <= bound, "ch {ch}: step {max_step} > ramp-slope bound {bound}");
    }
    let last = y[y.len() - 1];
    let level_after = last[0].abs() + last[1].abs();
    let ratio = level_after / level_before;
    assert!((ratio - 10f32.powf(-12.0 / 20.0)).abs() < 0.02, "settles at -12 dB, ratio {ratio}");
    // It really ramped: the first samples after the edit are still near the old level.
    let first = y[0][0].abs() + y[0][1].abs();
    assert!(first > 0.9 * level_before, "not an instant jump: {first} vs {level_before}");
}

/// Two identically built engines; B never gets the edits. Everything the edit must not touch
/// stays BIT-identical.
fn pair_of_engines(extra: impl Fn(&mut SpatialAudioEngine)) -> (SpatialAudioEngine, SpatialAudioEngine, SourceId, SourceId) {
    let build = |extra: &dyn Fn(&mut SpatialAudioEngine)| {
        let mut e = new_engine();
        let s0 = mono_source(&mut e);
        let s1 = mono_source(&mut e);
        let o = e.add_scene_output(SceneOutputConfig::new([2.0, 0.0, -3.0], Movability::Static));
        e.connect_pull(o, ChannelPull::new(s0, 0, 0.0));
        listener(&mut e, [0.0; 3]);
        extra(&mut e);
        e.update_scene_spatial();
        (e, s0, s1)
    };
    let (a, s0, s1) = build(&extra);
    let (b, _, _) = build(&extra);
    (a, b, s0, s1)
}

#[test]
fn reverb_tail_continues_across_set_pull_gain() {
    let (mut a, mut b, s0, _) = pair_of_engines(|_| {});
    let impulse = buf(|i| if i == 0 { 1.0 } else { 0.0 });
    let silence = dc(0.0);
    let (oa, ob) = (render(&mut a, &[&impulse, &silence], 1), render(&mut b, &[&impulse, &silence], 1));
    assert!(same(&oa[0], &ob[0]));
    let mut tail_energy = 0.0f32;
    for blk in 0..60 {
        if blk == 10 {
            // The edit under test (config-side call on A only).
            a.set_pull_gain(SceneOutputId(0), s0, 0, -20.0);
        }
        let (oa, ob) = (render(&mut a, &[&silence, &silence], 1), render(&mut b, &[&silence, &silence], 1));
        assert!(same(&oa[0], &ob[0]), "block {blk}: set_pull_gain disturbed the renderer state");
        if blk >= 10 {
            tail_energy += oa[0].rms();
        }
    }
    assert!(tail_energy > 1e-5, "there must be a reverb tail to preserve ({tail_energy})");
}

#[test]
fn adding_an_output_leaves_existing_outputs_bit_identical() {
    let (mut a, mut b, _, s1) = pair_of_engines(|_| {});
    let noise = buf(|i| ((i * 7919 % 251) as f32 / 251.0 - 0.5) * 0.8);
    let silence = dc(0.0);
    for _ in 0..6 {
        let (oa, ob) = (render(&mut a, &[&noise, &silence], 1), render(&mut b, &[&noise, &silence], 1));
        assert!(same(&oa[0], &ob[0]));
    }
    // A gets a second emitter pulling the (silent) source 1 mid-stream.
    let new = a.add_scene_output(SceneOutputConfig::new([-2.0, 0.0, -3.0], Movability::Static));
    a.connect_pull(new, ChannelPull::new(s1, 0, 0.0));
    // Phase 1: the new pair is unready (silent): everything else must be untouched.
    for blk in 0..10 {
        let (oa, ob) = (render(&mut a, &[&noise, &silence], 1), render(&mut b, &[&noise, &silence], 1));
        assert!(same(&oa[0], &ob[0]), "unready phase, block {blk}");
    }
    // Phase 2: coefficients published for the new pair (and re-published, unchanged, for the old).
    a.update_scene_spatial();
    b.update_scene_spatial();
    for blk in 0..20 {
        let (oa, ob) = (render(&mut a, &[&noise, &silence], 1), render(&mut b, &[&noise, &silence], 1));
        // The new emitter carries silence and has the same room (T60) as the old one, so the
        // shared reverb bus (mean T60 over ready outputs) is unchanged too: still bit-identical.
        assert!(same(&oa[0], &ob[0]), "ready phase, block {blk}");
    }
}

#[test]
fn removing_a_listener_or_output_leaves_survivors_bit_identical() {
    let (mut a, mut b, _, s1) = pair_of_engines(|e| listener(e, [0.5, 0.0, 0.0]));
    // Also a second (silent) output so removing one output has a survivor to compare.
    for e in [&mut a, &mut b] {
        let o2 = e.add_scene_output(SceneOutputConfig::new([-2.0, 0.0, -3.0], Movability::Static));
        e.connect_pull(o2, ChannelPull::new(s1, 0, 0.0));
        e.update_scene_spatial();
    }
    let noise = buf(|i| ((i * 7919 % 251) as f32 / 251.0 - 0.5) * 0.8);
    let silence = dc(0.0);
    for _ in 0..8 {
        render(&mut a, &[&noise, &silence], 2);
        render(&mut b, &[&noise, &silence], 2);
    }
    // Remove listener 1 from A only: listener 0's output must not change at all.
    a.remove_listener(ListenerId(1));
    for blk in 0..20 {
        let oa = render(&mut a, &[&noise, &silence], 1);
        let ob = render(&mut b, &[&noise, &silence], 2);
        assert!(same(&oa[0], &ob[0]), "listener 0 disturbed by removing listener 1 (block {blk})");
    }
    // Remove the silent output 1 from A: output 0's render is untouched.
    a.remove_scene_output(SceneOutputId(1));
    for blk in 0..20 {
        let oa = render(&mut a, &[&noise, &silence], 1);
        let ob = render(&mut b, &[&noise, &silence], 2);
        assert!(same(&oa[0], &ob[0]), "output 0 disturbed by removing output 1 (block {blk})");
    }
}

#[test]
fn connect_and_disconnect_ramp_and_drop() {
    let mut e = new_engine();
    let s = mono_source(&mut e);
    let o = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, -1.0], Movability::Static));
    listener(&mut e, [0.0; 3]);
    e.debug_audio_stage = 2;
    e.update_scene_spatial();
    let input = dc(0.5);
    for _ in 0..4 {
        let out = render(&mut e, &[&input], 1);
        assert_eq!(out[0].peak(), 0.0, "nothing connected: silence");
    }
    // Connect while running: fades in (first samples ~0, then audible, no jump).
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    assert_eq!(e.scene_outputs()[0].pulls.len(), 1, "config model is updated at once");
    let mut peak_first = 0.0f32;
    let mut y = Vec::new();
    for blk in 0..10 {
        let out = render(&mut e, &[&input], 1);
        if blk == 0 {
            peak_first = out[0].peak();
        }
        y.extend_from_slice(out[0].channel(0));
    }
    // The propagation delay (~140 samples) hides the start; by the end it is at full level.
    let steady = y[y.len() - 1];
    assert!(steady.abs() > 0.1);
    assert!(peak_first < 0.5 * steady.abs(), "attack starts well below the steady level: {peak_first} vs {steady}");
    let max_step = y.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    assert!(max_step < 0.5 / (0.015 * SR) * 1.5, "attack step {max_step}");

    // Disconnect: fades out to silence without a jump.
    e.disconnect_pull(o, s, 0);
    assert!(e.scene_outputs()[0].pulls.is_empty());
    let mut y = Vec::new();
    for _ in 0..12 {
        let out = render(&mut e, &[&input], 1);
        y.extend_from_slice(out[0].channel(0));
    }
    let max_step = y.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    assert!(max_step < 0.5 / (0.015 * SR) * 1.5, "release step {max_step}");
    assert!(y[y.len() - 1].abs() < 1e-5, "released tap is silent: {}", y[y.len() - 1]);
}

#[test]
fn edits_do_not_allocate_on_the_audio_side() {
    let mut e = new_engine();
    let s = mono_source(&mut e);
    let o = e.add_scene_output(SceneOutputConfig::new([1.0, 0.0, -2.0], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    listener(&mut e, [0.0; 3]);
    e.update_scene_spatial();
    let input = dc(0.3);
    render(&mut e, &[&input], 1);
    let mut out = vec![AudioBuffer::new(2, BLOCK as u16)];

    // The gain / disconnect edits themselves allocate nothing (config thread, but in-place).
    let ((), n) = count_allocs(|| e.set_pull_gain(o, s, 0, -6.0));
    assert_eq!(n, 0, "set_pull_gain allocated");
    // Blocks rendered while ramps are running and after structural edits allocate nothing.
    let ((), n) = count_allocs(|| {
        for _ in 0..4 {
            e.process_audio_scene(&[&input], &mut out);
        }
    });
    assert_eq!(n, 0, "render during a gain ramp allocated {n}");
    let ((), n) = count_allocs(|| e.disconnect_pull(o, s, 0));
    assert_eq!(n, 0, "disconnect_pull allocated");
    let ((), n) = count_allocs(|| {
        for _ in 0..4 {
            e.process_audio_scene(&[&input], &mut out);
        }
    });
    assert_eq!(n, 0, "render during a release ramp allocated {n}");
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    let o2 = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, -2.0], Movability::Static));
    e.connect_pull(o2, ChannelPull::new(s, 0, 0.0));
    e.update_scene_spatial();
    let ((), n) = count_allocs(|| {
        for _ in 0..8 {
            e.process_audio_scene(&[&input], &mut out);
        }
    });
    assert_eq!(n, 0, "render after add_scene_output / connect allocated {n}");
}
