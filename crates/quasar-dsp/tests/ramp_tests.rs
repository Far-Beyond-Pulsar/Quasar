//! Per-sample ramps of the early-reflection taps and the reverb send (#72).

use quasar_core::bands::Band8;
use quasar_core::param_exchange::{EarlyReflectionCoeffs, SpatialCoefficients};
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::early_reflections::EarlyReflectionDelayNode;
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::node_graph::AudioNode;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn params() -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -120.0,
        version: 0,
    }
}

fn tone_block(freq: f32, start: usize) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        b.set(0, i as u16, (2.0 * std::f64::consts::PI * freq as f64 * (start + i) as f64 / SR as f64).sin() as f32);
    }
    b
}

fn tap(delay: f32, gain: f32) -> EarlyReflectionCoeffs {
    EarlyReflectionCoeffs { azimuth: 0.0, elevation: 0.0, delay_samples: delay, gain: Band8::splat(gain) }
}

fn max_step(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
}

/// A tap whose gain and delay change between blocks does not step at the block boundary.
#[test]
fn reflection_tap_gain_and_delay_ramp_per_sample() {
    let freq = 500.0_f32;
    let mut node = EarlyReflectionDelayNode::new(1, SR, 0.2, 16);
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    let mut all = Vec::new();
    let mut pos = 0;
    for b in 0..24 {
        // Gain 0.2 -> 1.0 and delay 400 -> 440 at block 8 (slope 40/256 = 0.16 < glide limit).
        let (d, g) = if b < 8 { (400.0, 0.2) } else { (440.0, 1.0) };
        node.update_reflections(&[tap(d, g)]);
        node.process(&tone_block(freq, pos), &mut out, &params());
        all.extend_from_slice(&out.channel(0)[..BLOCK]);
        pos += BLOCK;
    }
    let natural = 2.0 * std::f32::consts::PI * freq / SR;
    // Skip the delay-line fill (first 400 samples).
    let step = max_step(&all[500..]);
    assert!(step <= natural * 1.0 * 1.3 + 0.01, "step {step} vs {natural}");
    // The gain really arrived.
    let peak = all[all.len() - 2 * BLOCK..].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!((peak - 1.0).abs() < 0.01, "peak {peak}");
}

/// A reflection that appears fades in, one that disappears fades out (no click).
#[test]
fn appearing_and_vanishing_taps_fade() {
    let freq = 500.0_f32;
    let mut node = EarlyReflectionDelayNode::new(1, SR, 0.2, 16);
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    let mut all = Vec::new();
    let mut pos = 0;
    for b in 0..20 {
        let list: Vec<EarlyReflectionCoeffs> = match b {
            0..=5 => vec![tap(300.0, 0.5)],
            6..=12 => vec![tap(300.0, 0.5), tap(900.0, 0.5)], // second tap appears
            _ => vec![tap(900.0, 0.5)],                       // first tap vanishes
        };
        node.update_reflections(&list);
        node.process(&tone_block(freq, pos), &mut out, &params());
        all.extend_from_slice(&out.channel(0)[..BLOCK]);
        pos += BLOCK;
    }
    let natural = 2.0 * std::f32::consts::PI * freq / SR;
    // Two simultaneous taps at 0.5 each: bound by their summed slope. Skip the line fill.
    let step = max_step(&all[1000..]);
    assert!(step <= natural * 1.1, "step {step} vs {natural}");
}

/// The reverb send level (wet gain) is ramped per sample across a block.
#[test]
fn reverb_wet_level_ramps_per_sample() {
    let mut rev = FdnReverbNode::new(1, SR);

    let mut p = params();
    p.late_gain_db = 0.0;
    let mut dc = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        dc.set(0, i as u16, 0.5);
    }
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    // Let the FDN settle on DC at full send level.
    for _ in 0..200 {
        rev.process(&dc, &mut out, &p);
    }
    let before = out.channel(0)[BLOCK - 1];
    assert!(before.abs() > 1e-3, "reverb must produce output, got {before}");

    // Drop the send by 20 dB: with a per-sample ramp, no step exceeds the ramp slope
    // times the output level (plus the FDN's own tiny sample-to-sample variation).
    p.late_gain_db = -20.0;
    rev.process(&dc, &mut out, &p);
    let step = max_step(&out.channel(0)[..BLOCK]);
    let slope = (1.0 - 0.1) / BLOCK as f32 * before.abs();
    assert!(step <= slope * 2.0 + 1e-4, "step {step} vs ramp slope {slope}");
    // And the level ends 20 dB lower once the lines have re-settled.
    for _ in 0..200 {
        rev.process(&dc, &mut out, &p);
    }
    let after = out.channel(0)[BLOCK - 1];
    assert!((after / before - 0.1).abs() < 0.02, "level ratio {}", after / before);
}
