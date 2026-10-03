//! Unit tests of the per-tap reflection decoder (#58).

use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::early_reflections::EarlyReflectionDelayNode;
use quasar_dsp::master_decoder::{layout_panner, SpeakerLayout};
use quasar_dsp::reflection_decoder::{ReflectionDecoder, TapTarget, REFLECTION_SLOTS};

const SR: f32 = 48_000.0;
const N: usize = 256;

fn tap(delay: f32, gain: f32, az: f32) -> TapTarget {
    TapTarget { delay_samples: delay, gain_lo: gain, gain_hi: gain, azimuth: az, elevation: 0.0 }
}

fn impulse_block() -> AudioBuffer {
    let mut b = AudioBuffer::new(1, N as u16);
    b.set(0, 0, 1.0);
    b
}

#[test]
fn a_tap_is_a_delayed_impulse_panned_with_the_vbap_gains_of_its_azimuth() {
    let panner = layout_panner(&SpeakerLayout::Quad);
    let az = -1.0_f32; // 57 deg left
    let mut g = [0.0_f32; 4];
    panner.gains(az, 0.0, &mut g);

    let mut line = EarlyReflectionDelayNode::new(1, SR, 0.2, 16);
    let mut dec = ReflectionDecoder::new(SR, false);
    let mut out = AudioBuffer::new(4, N as u16);
    // Two blocks so the tap is steady (it fades in during the first one).
    for b in 0..2 {
        out.clear();
        line.push_block(&if b == 0 { AudioBuffer::new(1, N as u16) } else { impulse_block() });
        dec.render_add(&line, &[tap(100.0, 0.5, az)], Some(&panner), &mut out, N);
    }
    // Impulse pushed at the start of block 1 appears 100 samples later, gain 0.5 x VBAP.
    for ch in 0..4 {
        let want = 0.5 * g[ch];
        assert!((out.get(ch as u16, 100) - want).abs() < 1e-4, "ch {ch}: {} vs {want}", out.get(ch as u16, 100));
        for i in 0..N {
            if i != 100 {
                assert!(out.get(ch as u16, i as u16).abs() < 1e-4, "ch {ch} sample {i}");
            }
        }
    }
}

#[test]
fn taps_vanish_when_their_target_disappears_and_slots_are_bounded() {
    let panner = layout_panner(&SpeakerLayout::Stereo);
    let mut line = EarlyReflectionDelayNode::new(1, SR, 0.2, 16);
    let mut dec = ReflectionDecoder::new(SR, false);
    let mut out = AudioBuffer::new(2, N as u16);
    let sil = AudioBuffer::new(1, N as u16);

    // More targets than slots (and non-finite junk): never panics, slots stay bounded.
    let mut targets: Vec<TapTarget> = (0..64).map(|i| tap(50.0 + 20.0 * i as f32, 0.1, 0.0)).collect();
    targets.push(TapTarget { delay_samples: f32::NAN, gain_lo: f32::NAN, gain_hi: 1.0, azimuth: f32::NAN, elevation: f32::NAN });
    line.push_block(&sil);
    dec.render_add(&line, &targets, Some(&panner), &mut out, N);
    assert!(dec.active_taps() <= REFLECTION_SLOTS);
    assert!(dec.active_taps() > 0);
    for ch in 0..2 {
        assert!(out.channel(ch).iter().all(|v| v.is_finite()));
    }

    // Targets gone: the taps fade out over one block, then the slots are free.
    out.clear();
    line.push_block(&sil);
    dec.render_add(&line, &[], Some(&panner), &mut out, N);
    assert_eq!(dec.active_taps(), 0);
}

#[test]
fn hrtf_decoder_renders_the_tap_louder_in_the_ipsilateral_ear() {
    let mut line = EarlyReflectionDelayNode::new(1, SR, 0.2, 16);
    let mut dec = ReflectionDecoder::new(SR, true);
    let mut out = AudioBuffer::new(2, N as u16);
    let (mut el, mut er) = (0.0_f32, 0.0_f32);
    for b in 0..3 {
        out.clear();
        line.push_block(&if b == 1 { impulse_block() } else { AudioBuffer::new(1, N as u16) });
        // 70 deg to the right.
        dec.render_add(&line, &[tap(80.0, 0.8, 1.2)], None, &mut out, N);
        el += out.channel(0).iter().map(|v| v * v).sum::<f32>();
        er += out.channel(1).iter().map(|v| v * v).sum::<f32>();
    }

    // The right ear is the ipsilateral one.
    assert!(er > 1.3 * el, "right {er} left {el}");
}
