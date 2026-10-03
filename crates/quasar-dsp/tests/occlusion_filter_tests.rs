//! #47 / #48: per-band filtering and propagation delay of `AirAbsorptionOcclusionNode`.

use quasar_core::bands::{Band8, FREQ_BAND_CENTRES};
use quasar_core::param_exchange::SpatialCoefficients;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::node_graph::AudioNode;
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn params(gains: Band8, delay: f32) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: gains,
        direct_delay_samples: delay,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: 0.0,
        version: 0,
    }
}

/// Run `blocks` blocks of a unit sine through the node with constant params
/// and return the output (all blocks concatenated).
fn run_sine(node: &mut AirAbsorptionOcclusionNode, p: &SpatialCoefficients, freq: f32, blocks: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks * BLOCK);
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    let mut output = AudioBuffer::new(1, BLOCK as u16);
    let mut n = 0usize;
    for _ in 0..blocks {
        for i in 0..BLOCK {
            input.set(0, i as u16, ((2.0 * std::f64::consts::PI * freq as f64 * (n + i) as f64) / SR as f64).sin() as f32);
        }
        n += BLOCK;
        node.process(&input, &mut output, p);
        out.extend_from_slice(&output.channel(0)[..BLOCK]);
    }
    out
}

/// Steady-state gain in dB at `freq`: amplitude of the output sine over the
/// last `window` samples, by projection on sin / cos.
fn gain_db_at(node: &mut AirAbsorptionOcclusionNode, p: &SpatialCoefficients, freq: f32) -> f32 {
    let blocks = 80; // plenty for the 62.5 Hz band to settle
    let out = run_sine(node, p, freq, blocks);
    let start = (blocks - 20) * BLOCK;
    let (mut s, mut c, mut cnt) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (k, &y) in out.iter().enumerate().skip(start) {
        let ph = 2.0 * std::f64::consts::PI * freq as f64 * k as f64 / SR as f64;
        s += y as f64 * ph.sin();
        c += y as f64 * ph.cos();
        cnt += 1.0;
    }
    let amp = 2.0 * (s * s + c * c).sqrt() / cnt;
    20.0 * (amp.max(1e-12)).log10() as f32
}

fn check_response(gains: [f32; 8], tol_db: f32) {
    let p = params(Band8::new(gains), 0.0);
    for (i, &f) in FREQ_BAND_CENTRES.iter().enumerate() {
        let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
        let got = gain_db_at(&mut node, &p, f);
        let want = 20.0 * gains[i].max(1e-9).log10();
        assert!(
            (got - want).abs() <= tol_db,
            "gains {gains:?} band {i} ({f} Hz): got {got:.2} dB, want {want:.2} dB"
        );
    }
}

/// The acceptance vector from the issue.
#[test]
fn response_matches_issue_vector_within_1_db() {
    check_response([1.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.1, 0.05], 1.0);
}

#[test]
fn response_matches_other_gain_vectors() {
    // Flat attenuation, mild tilt, air-absorption-like, occluder-like, and a lone bump.
    check_response([0.3; 8], 1.0);
    check_response([1.0; 8], 1.0);
    check_response([0.9, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3, 0.15], 1.0);
    check_response([0.5, 0.45, 0.4, 0.3, 0.2, 0.12, 0.06, 0.03], 1.0);
    check_response([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5], 1.0);
    check_response([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0);
    check_response([0.2, 0.4, 0.7, 1.0, 0.7, 0.4, 0.2, 0.1], 1.0);
}

#[test]
fn occlusion_darkens_the_sound() {
    // High bands blocked more than low bands: HF loses more than LF.
    let g = [0.9, 0.85, 0.7, 0.5, 0.3, 0.15, 0.07, 0.03];
    let p = params(Band8::new(g), 0.0);
    let mut a = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let mut b = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let lf = gain_db_at(&mut a, &p, 125.0);
    let hf = gain_db_at(&mut b, &p, 8000.0);
    assert!(lf - hf > 20.0, "LF {lf} dB vs HF {hf} dB");
}

/// Everything above the top band keeps following it (shelf, not a bell).
#[test]
fn top_band_gain_extends_above_8k() {
    let p = params(Band8::new([1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.2]), 0.0);
    let mut n = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let at_8k = gain_db_at(&mut n, &p, 8000.0);
    let mut n = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let at_16k = gain_db_at(&mut n, &p, 16_000.0);
    assert!(at_16k <= at_8k + 1.0, "16 kHz {at_16k} dB must not rise above 8 kHz {at_8k} dB");
}

// ── zipper / click behaviour ─────────────────────────────────────────

fn process_block(node: &mut AirAbsorptionOcclusionNode, input: &AudioBuffer, p: &SpatialCoefficients) -> Vec<f32> {
    let mut output = AudioBuffer::new(1, BLOCK as u16);
    node.process(input, &mut output, p);
    output.channel(0)[..BLOCK].to_vec()
}

fn sine_block(freq: f32, start: usize) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        b.set(0, i as u16, ((2.0 * std::f64::consts::PI * freq as f64 * (start + i) as f64) / SR as f64).sin() as f32);
    }
    b
}

/// A gain change between blocks is ramped per sample: the output has no step
/// at the block boundary beyond what the (tiny) gain slope and the tone allow.
#[test]
fn gain_change_has_no_zipper_step() {
    let freq = 440.0_f32;
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let loud = params(Band8::splat(1.0), 0.0);
    let quiet = params(Band8::splat(0.1), 0.0);

    let mut pos = 0usize;
    let mut all = Vec::new();
    for b in 0..8 {
        let p = if b < 4 { &loud } else { &quiet };
        all.extend(process_block(&mut node, &sine_block(freq, pos), p));
        pos += BLOCK;
    }
    // Max natural sample step of a unit sine at 440 Hz.
    let nat = 2.0 * std::f32::consts::PI * freq / SR;
    let max_step = all.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f32, f32::max);
    assert!(max_step <= nat * 1.05, "step {max_step} vs natural {nat}");
    // And the level really did fall.
    let tail = all[all.len() - BLOCK..].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!((tail - 0.1).abs() < 0.01, "settled level {tail}");
}

/// A spectral-shape change (flat -> dark) while a tone plays does not click.
#[test]
fn shape_change_has_no_click() {
    let freq = 3000.0_f32;
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let clear = params(Band8::splat(1.0), 0.0);
    let dark = params(Band8::new([1.0, 0.9, 0.7, 0.4, 0.2, 0.08, 0.03, 0.01]), 0.0);
    let mut pos = 0usize;
    let mut all = Vec::new();
    for b in 0..10 {
        let p = if b < 4 { &clear } else { &dark };
        all.extend(process_block(&mut node, &sine_block(freq, pos), p));
        pos += BLOCK;
    }
    let nat = 2.0 * std::f32::consts::PI * freq / SR;
    let max_step = all.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f32, f32::max);
    assert!(max_step <= nat * 1.15, "step {max_step} vs natural {nat}");
}

#[test]
fn silent_and_garbage_gains_are_safe() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let input = sine_block(1000.0, 0);
    for g in [Band8::splat(0.0), Band8::splat(f32::NAN), Band8::splat(f32::INFINITY), Band8::splat(-1.0), Band8::splat(1e-30)] {
        let out = process_block(&mut node, &input, &params(g, f32::NAN));
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output for {g:?}");
    }
    let _ = process_block(&mut node, &input, &params(Band8::splat(0.0), 0.0));
    let out = process_block(&mut node, &input, &params(Band8::splat(0.0), 0.0));
    assert!(out.iter().all(|&v| v == 0.0), "zero gain must be silent once the ramp is done");
}

// ── propagation delay (#48) ──────────────────────────────────────────

fn impulse_position(delay: f32) -> (usize, f32) {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.5);
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    input.set(0, 0, 1.0);
    let p = params(Band8::splat(1.0), delay);
    let mut best = (0usize, 0.0_f32);
    let mut out_all = process_block(&mut node, &input, &p);
    let silence = AudioBuffer::new(1, BLOCK as u16);
    for _ in 0..60 {
        out_all.extend(process_block(&mut node, &silence, &p));
    }
    for (i, &v) in out_all.iter().enumerate() {
        if v.abs() > best.1.abs() {
            best = (i, v);
        }
    }
    best
}

#[test]
fn impulse_appears_at_the_delay() {
    for d in [0.0_f32, 1.0, 50.0, 100.0, 255.0, 256.0, 700.0, 1700.0, 4321.0, 12_000.0] {
        let (pos, v) = impulse_position(d);
        assert!(pos as f32 >= d.round() - 1.0 && pos as f32 <= d.round() + 1.0, "delay {d}: impulse at {pos}");
        assert!((v - 1.0).abs() < 1e-3, "delay {d}: peak {v}");
    }
    // Fractional delays: the energy centroid sits at the delay within 0.1 sample.
    for d in [10.25_f32, 37.5, 100.8] {
        let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
        let mut input = AudioBuffer::new(1, BLOCK as u16);
        input.set(0, 0, 1.0);
        let out = process_block(&mut node, &input, &params(Band8::splat(1.0), d));
        let sum: f32 = out.iter().sum();
        let centroid: f32 = out.iter().enumerate().map(|(i, v)| i as f32 * v).sum::<f32>() / sum;
        assert!((centroid - d).abs() < 0.1, "delay {d}: centroid {centroid}");
    }
}

#[test]
fn delay_larger_than_capacity_is_clamped() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.01);
    let input = sine_block(500.0, 0);
    let out = process_block(&mut node, &input, &params(Band8::splat(1.0), 1.0e9));
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Estimate the frequency of a sine from rising zero crossings with linear interpolation.
fn estimate_freq(x: &[f32]) -> f32 {
    let mut first = None;
    let mut last = 0.0_f32;
    let mut count = 0;
    for i in 1..x.len() {
        if x[i - 1] < 0.0 && x[i] >= 0.0 {
            let pos = (i - 1) as f32 + (-x[i - 1]) / (x[i] - x[i - 1]);
            if first.is_none() {
                first = Some(pos);
            } else {
                count += 1;
            }
            last = pos;
        }
    }
    count as f32 * SR / (last - first.unwrap())
}

/// A delay that grows by `r` samples per sample shifts the pitch by `1 - r`
/// (receding source), one that shrinks raises it, within 1 %, with no clicks.
#[test]
fn constant_radial_speed_gives_the_expected_doppler_shift() {
    let freq = 2000.0_f32;
    for r in [0.05_f32, -0.05, 0.1, -0.1, 0.2] {
        let mut node = AirAbsorptionOcclusionNode::new(1, SR, 1.0);
        let mut pos = 0usize;
        let mut all = Vec::new();
        // Start well inside the line so a negative slope stays >= 0.
        let start_delay = 4000.0_f32;
        for b in 0..60 {
            let d = start_delay + r * (b * BLOCK) as f32;
            all.extend(process_block(&mut node, &sine_block(freq, pos), &params(Band8::splat(1.0), d)));
            pos += BLOCK;
        }
        // Skip the start (delay-line fill: the first valid output is at ~5000).
        let skip = 6000;
        let est = estimate_freq(&all[skip..]);
        let want = freq * (1.0 - r);
        assert!((est - want).abs() / want < 0.01, "r={r}: pitch {est} vs {want}");
        let nat = 2.0 * std::f32::consts::PI * freq / SR * (1.0 + r.abs());
        let max_step = all[skip..].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f32, f32::max);
        assert!(max_step <= nat * 1.05, "r={r}: click, step {max_step} vs {nat}");
    }
}

/// A teleport (huge delay jump) is crossfaded, not pitch-glided: the output stays
/// bounded and the new delay is in place after one block.
#[test]
fn large_delay_jump_crossfades_without_clicks() {
    let freq = 1000.0_f32;
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 1.0);
    let mut pos = 0usize;
    let mut all = Vec::new();
    for b in 0..80 {
        let d = if b < 15 { 1000.0 } else { 9000.0 };
        all.extend(process_block(&mut node, &sine_block(freq, pos), &params(Band8::splat(1.0), d)));
        pos += BLOCK;
    }
    let skip = 2000;
    let max = all[skip..].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max <= 1.0 + 1e-3, "overshoot {max}");
    let nat = 2.0 * std::f32::consts::PI * freq / SR;
    let max_step = all[skip..].windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f32, f32::max);
    // The crossfade of two phase-shifted copies keeps steps within a small multiple of the tone slope.
    assert!(max_step <= nat * 1.6, "step {max_step} vs natural {nat}");
    // After the jump the output is the input delayed by 9000 samples.
    let n = all.len();
    for i in n - 50..n {
        let expect = ((2.0 * std::f64::consts::PI * freq as f64 * (i as f64 - 9000.0)) / SR as f64).sin() as f32;
        assert!((all[i] - expect).abs() < 1e-3, "sample {i}: {} vs {expect}", all[i]);
    }
}

#[test]
fn reset_clears_state_and_restarts_unramped() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.1);
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    input.set(0, 0, 1.0);
    let _ = process_block(&mut node, &input, &params(Band8::splat(1.0), 100.0));
    node.reset();
    let silence = AudioBuffer::new(1, BLOCK as u16);
    for _ in 0..3 {
        let out = process_block(&mut node, &silence, &params(Band8::splat(1.0), 0.0));
        assert!(out.iter().all(|&v| v == 0.0));
    }
}

/// Two channels are delayed and filtered independently with identical params.
#[test]
fn multichannel_matches_mono() {
    let mut stereo = AirAbsorptionOcclusionNode::new(2, SR, 0.05);
    let mut mono = AirAbsorptionOcclusionNode::new(1, SR, 0.05);
    let p = params(Band8::new([1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]), 33.3);
    let a = sine_block(700.0, 0);
    let mut inp = AudioBuffer::new(2, BLOCK as u16);
    for i in 0..BLOCK {
        inp.set(0, i as u16, a.get(0, i as u16));
        inp.set(1, i as u16, -a.get(0, i as u16));
    }
    let mut out2 = AudioBuffer::new(2, BLOCK as u16);
    stereo.process(&inp, &mut out2, &p);
    let m = process_block(&mut mono, &a, &p);
    for i in 0..BLOCK {
        assert!((out2.get(0, i as u16) - m[i]).abs() < 1e-6);
        assert!((out2.get(1, i as u16) + m[i]).abs() < 1e-6);
    }
}
