//! #83 / #149: optional output format conversion (render 7.1, play stereo / 5.1) as the last
//! channel stage of a listener, before the limiter.

mod common;

use common::count_allocs;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::channel_matrix::{downmix_gains, MatrixError, BS775_K};
use quasar_audio::quasar_dsp::limiter::OutputSafetyConfig;
use quasar_audio::quasar_dsp::master_decoder::SpeakerLayout;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const AMP: f32 = 0.3;

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

fn sine(phase: &mut f32, n: usize) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, n as u16);
    for i in 0..n {
        b.set(0, i as u16, AMP * phase.sin());
        *phase += std::f32::consts::TAU * 440.0 / SR;
    }
    b
}

fn render(e: &mut SpatialAudioEngine, input: &AudioBuffer, out_ch: usize) -> AudioBuffer {
    let mut out = [AudioBuffer::new(out_ch as u16, input.samples())];
    e.process_audio_scene(&[input], &mut out);
    let [o] = out;
    o
}

fn power(b: &AudioBuffer) -> f32 {
    (0..b.channels()).map(|c| b.channel(c).iter().map(|&s| s * s).sum::<f32>()).sum()
}

/// Matrix applied to a reference render, channel by channel.
fn apply(gains: &[f32], in_ch: usize, out_ch: usize, x: &AudioBuffer) -> Vec<Vec<f32>> {
    (0..out_ch)
        .map(|o| {
            (0..x.samples() as usize)
                .map(|j| (0..in_ch).map(|i| gains[o * in_ch + i] * x.channel(i as u16)[j]).sum())
                .collect()
        })
        .collect()
}

const FRONT: [f32; 3] = [0.0, 0.0, -1.0];
const REAR_RIGHT: [f32; 3] = [1.0, 0.0, 1.0];
const FRONT_LEFT: [f32; 3] = [-1.0, 0.0, -1.0];

fn check_conversion(device: PhysicalOutputLayout, to: SpeakerLayout, dev_ch: usize, pos: [f32; 3], block: usize) -> (f32, f32) {
    let mut reference = engine(PhysicalOutputLayout::Surround714, pos);
    let mut conv = engine(PhysicalOutputLayout::Surround714, pos);
    conv.set_listener_output_layout(ListenerId(0), Some(device)).expect("supported conversion");
    let gains = downmix_gains(&SpeakerLayout::Surround714, &to).unwrap();
    let (mut p1, mut p2) = (0.0, 0.0);
    let (mut pw_ref, mut pw_out) = (0.0f32, 0.0f32);
    let settled = 600 / block + 2; // the 10 ms (480 sample) ramp is over
    for blk in 0..settled + 12 {
        let r = render(&mut reference, &sine(&mut p1, block), 8);
        let c = render(&mut conv, &sine(&mut p2, block), dev_ch);
        assert_eq!(c.channels() as usize, dev_ch);
        if blk >= settled {
            // Ramp (10 ms = 480 samples) is over: the output IS the matrix applied to the 7.1 render.
            let want = apply(&gains, 8, dev_ch, &r);
            for ch in 0..dev_ch {
                for j in 0..block {
                    let (a, b) = (c.channel(ch as u16)[j], want[ch][j]);
                    assert!((a - b).abs() <= 1e-6, "blk {blk} ch {ch} j {j}: {a} vs {b}");
                }
            }
            pw_ref += power(&r);
            pw_out += power(&c);
        }
    }
    (pw_ref, pw_out)
}

#[test]
fn render_71_to_stereo_follows_the_bs775_matrix() {
    for pos in [FRONT, FRONT_LEFT, REAR_RIGHT] {
        check_conversion(PhysicalOutputLayout::Stereo, SpeakerLayout::Stereo, 2, pos, BLOCK);
    }
}

#[test]
fn render_71_to_51_follows_the_matrix() {
    for pos in [FRONT, FRONT_LEFT, REAR_RIGHT] {
        check_conversion(PhysicalOutputLayout::Surround51, SpeakerLayout::Surround51, 6, pos, BLOCK);
    }
}

#[test]
fn short_blocks_are_converted_too() {
    check_conversion(PhysicalOutputLayout::Stereo, SpeakerLayout::Stereo, 2, FRONT_LEFT, 100);
    check_conversion(PhysicalOutputLayout::Surround51, SpeakerLayout::Surround51, 6, REAR_RIGHT, 37);
}

#[test]
fn power_of_the_front_centre_is_kept_and_a_single_side_surround_loses_3_db() {
    // Front centre: the centre goes to L and R at -3 dB each: unit total power.
    let (pr, po) = check_conversion(PhysicalOutputLayout::Stereo, SpeakerLayout::Stereo, 2, FRONT, BLOCK);
    let db = 10.0 * (po / pr).log10();
    eprintln!("front source, 7.1 -> stereo power change {db:.3} dB");
    assert!(db.abs() < 0.3, "front centre power change {db} dB");
    // A source only in the right surround: all of it goes to the right channel at -3 dB.
    let (pr, po) = check_conversion(PhysicalOutputLayout::Stereo, SpeakerLayout::Stereo, 2, REAR_RIGHT, BLOCK);
    let db = 10.0 * (po / pr).log10();
    eprintln!("rear-right source, 7.1 -> stereo power change {db:.3} dB (matrix k^2 = {:.3} dB)", 10.0 * (BS775_K * BS775_K).log10());
    assert!(db < -2.0 && db > -4.5, "rear-right power change {db} dB");
}

#[test]
fn unsupported_conversions_error_and_change_nothing() {
    let mut e = engine(PhysicalOutputLayout::Stereo, FRONT);
    // Upmix is not defined.
    let r = e.set_listener_output_layout(ListenerId(0), Some(PhysicalOutputLayout::Surround51));
    assert!(matches!(r, Err(MatrixError::UnsupportedConversion { from_channels: 2, to_channels: 6 })), "{r:?}");
    let custom = PhysicalOutputLayout::Custom { positions: vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]] };
    assert!(e.set_listener_output_layout(ListenerId(0), Some(custom.clone())).is_err());
    let mut e71 = engine(PhysicalOutputLayout::Surround714, FRONT);
    assert!(e71.set_listener_output_layout(ListenerId(0), Some(custom)).is_err());
    // Nothing was configured: bit-identical to an untouched engine.
    let mut plain = engine(PhysicalOutputLayout::Stereo, FRONT);
    let (mut p1, mut p2) = (0.0, 0.0);
    for _ in 0..10 {
        let (a, b) = (render(&mut e, &sine(&mut p1, BLOCK), 2), render(&mut plain, &sine(&mut p2, BLOCK), 2));
        assert_eq!((a.channel(0), a.channel(1)), (b.channel(0), b.channel(1)));
    }
}

#[test]
fn default_and_removed_conversion_are_bit_identical_to_the_plain_engine() {
    let mut plain = engine(PhysicalOutputLayout::Surround714, FRONT_LEFT);
    let mut none = engine(PhysicalOutputLayout::Surround714, FRONT_LEFT);
    none.set_listener_output_layout(ListenerId(0), None).unwrap(); // nothing to remove
    let mut ident = engine(PhysicalOutputLayout::Surround714, FRONT_LEFT);
    ident.set_listener_output_layout(ListenerId(0), Some(PhysicalOutputLayout::Surround714)).unwrap();
    let mut toggled = engine(PhysicalOutputLayout::Surround714, FRONT_LEFT);
    let (mut p0, mut p1, mut p2, mut p3) = (0.0, 0.0, 0.0, 0.0);
    for blk in 0..40 {
        if blk == 5 {
            toggled.set_listener_output_layout(ListenerId(0), Some(PhysicalOutputLayout::Stereo)).unwrap();
        }
        if blk == 15 {
            toggled.set_listener_output_layout(ListenerId(0), None).unwrap();
        }
        let r = render(&mut plain, &sine(&mut p0, BLOCK), 8);
        let n = render(&mut none, &sine(&mut p1, BLOCK), 8);
        let i = render(&mut ident, &sine(&mut p2, BLOCK), 8);
        let t = render(&mut toggled, &sine(&mut p3, BLOCK), 8);
        for ch in 0..8u16 {
            assert_eq!(r.channel(ch), n.channel(ch), "never configured vs removed, blk {blk} ch {ch}");
            if blk >= 3 {
                assert_eq!(r.channel(ch), i.channel(ch), "identity stage, blk {blk} ch {ch}");
            }
            if blk >= 25 {
                assert_eq!(r.channel(ch), t.channel(ch), "after removing the conversion, blk {blk} ch {ch}");
            }
        }
    }
}

#[test]
fn switching_the_conversion_on_and_off_is_click_free() {
    // Front-centre source: before the switch only the centre channel (2) is active, channels 0/1
    // are silent. After it L / R carry the centre at -3 dB. A hard switch would step channel 0
    // from 0 to ~0.7 * AMP * sin(phase) in one sample; the cross-fade must not.
    let mut e = engine(PhysicalOutputLayout::Surround714, FRONT);
    let mut ph = 0.0;
    let mut last: Option<f32> = None;
    let mut max_jump = 0.0f32;
    let mut max_before = 0.0f32;
    for blk in 0..60 {
        if blk == 20 {
            e.set_listener_output_layout(ListenerId(0), Some(PhysicalOutputLayout::Stereo)).unwrap();
        }
        if blk == 40 {
            e.set_listener_output_layout(ListenerId(0), None).unwrap();
        }
        // 8-channel buffer on every call: the transition cross-fades 7.1 and stereo.
        let o = render(&mut e, &sine(&mut ph, BLOCK), 8);
        for j in 0..BLOCK {
            let v = o.channel(0)[j];
            if blk >= 5 {
                if let Some(l) = last {
                    max_jump = max_jump.max((v - l).abs());
                }
            }
            if blk < 20 {
                max_before = max_before.max(v.abs());
            }
            last = Some(v);
        }
    }
    let hard_step = BS775_K * AMP * 0.9;
    eprintln!("max sample-to-sample jump on channel 0 across both switches: {max_jump:.4} (a hard switch would jump up to ~{hard_step:.3})");
    assert!(max_before < 1e-3, "the centre source leaks into channel 0 before the switch ({max_before})");
    assert!(max_jump < 0.06, "click: sample jump {max_jump}");
}

#[test]
fn limiter_runs_after_the_conversion_so_the_ceiling_holds_on_device_channels() {
    // Two coherent front channels summed into one: a downmix that sums can exceed what each
    // 7.1 channel carried. Drive a front-left source hard, convert to stereo and require the
    // ceiling at the output.
    let ceil = 0.891_250_9;
    let mut e = engine(PhysicalOutputLayout::Surround714, FRONT);
    e.set_listener_output_layout(ListenerId(0), Some(PhysicalOutputLayout::Stereo)).unwrap();
    e.connect_pull(
        quasar_audio::quasar_core::scene_output::SceneOutputId(0),
        ChannelPull::new(quasar_audio::quasar_core::scene_output::SourceId(0), 0, 18.0),
    );
    let mut ph: f32 = 0.0;
    let mut peak = 0.0f32;
    for _ in 0..40 {
        let mut b = AudioBuffer::new(1, BLOCK as u16);
        for i in 0..BLOCK {
            b.set(0, i as u16, ph.sin());
            ph += std::f32::consts::TAU * 440.0 / SR;
        }
        let o = render(&mut e, &b, 2);
        peak = peak.max(o.peak());
    }
    assert!(peak <= ceil, "peak {peak} above the ceiling after conversion");
    assert!(peak > 0.7, "limited, not muted: {peak}");
    assert!(e.output_meter(ListenerId(0)).limited_samples() > 0);
    let _ = OutputSafetyConfig::default();
}

#[test]
fn the_audio_thread_never_allocates_with_a_conversion_configured() {
    let mut e = engine(PhysicalOutputLayout::Surround714, REAR_RIGHT);
    let mut renderer = e.audio_handle();
    let mut ph = 0.0;
    // Warm up, then switch (the command is applied INSIDE the counted region) and keep rendering.
    for _ in 0..4 {
        let mut out = [AudioBuffer::new(8, BLOCK as u16)];
        renderer.process_audio_scene(&[&sine(&mut ph, BLOCK)], &mut out);
    }
    e.set_listener_output_layout(ListenerId(0), Some(PhysicalOutputLayout::Stereo)).unwrap();
    let inputs: Vec<AudioBuffer> = (0..40).map(|_| sine(&mut ph, BLOCK)).collect();
    let mut outs = [AudioBuffer::new(8, BLOCK as u16)];
    let ((), n) = count_allocs(|| {
        for inp in &inputs {
            renderer.process_audio_scene(&[inp], &mut outs);
        }
    });
    assert_eq!(n, 0, "audio thread allocated {n} times with the conversion switching on");
    // And switching it off again (identity fade-in, retirement of the stage) is allocation-free too.
    e.set_listener_output_layout(ListenerId(0), None).unwrap();
    let ((), n) = count_allocs(|| {
        for inp in &inputs {
            renderer.process_audio_scene(&[inp], &mut outs);
        }
    });
    assert_eq!(n, 0, "audio thread allocated {n} times while removing the conversion");
}
