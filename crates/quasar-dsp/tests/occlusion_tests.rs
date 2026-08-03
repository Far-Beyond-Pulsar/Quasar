//! Unit tests for the air-absorption / occlusion node (P2 scene pipeline).
//!
//! Note: the node's per-band biquads are RBJ lowpass filters with DC gain 1,
//! so the tests use an impulse + long observation window (impulse response
//! energy) rather than a steady-state level. The delay path is tested with
//! the default (identity) filters to isolate the fractional delay line.

use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::node_graph::AudioNode;
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn params(gain: f32, delay: f32) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(gain),
        direct_delay_samples: delay,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: 0.0,
        version: 0,
    }
}

/// Fire a single unit impulse (block 0) and sum output energy over `blocks`
/// 256-sample blocks. The node's lowpass chain has a long (~500-sample) rise,
/// so a multi-block window is required to capture most of the response.
fn impulse_energy(node: &mut AirAbsorptionOcclusionNode, blocks: usize) -> f64 {
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    let mut output = AudioBuffer::new(1, BLOCK as u16);
    let mut energy = 0.0_f64;
    let mut fired = false;
    for _ in 0..blocks {
        input.clear();
        if !fired {
            input.set(0, 0, 1.0);
            fired = true;
        }
        output.clear();
        node.process(&input, &mut output, &params(1.0, 0.0));
        for i in 0..BLOCK {
            let v = output.get(0, i as u16) as f64;
            energy += v * v;
        }
    }
    energy
}

// ── occlusion_attenuates_direct_path ─────────────────────────────────

#[test]
fn occlusion_attenuates_direct_path() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.1);

    // Clear line of sight (attenuation 1.0 → cutoffs near band centres).
    node.update_occlusion(&Band8::splat(1.0), 0.0);
    let clear_energy = impulse_energy(&mut node, 16);

    // Fully occluded (attenuation 0.02 → cutoffs clamped toward 20 Hz).
    node.reset();
    node.update_occlusion(&Band8::splat(0.02), 0.0);
    let occluded_energy = impulse_energy(&mut node, 16);

    assert!(clear_energy > 0.0, "clear path must pass energy");
    assert!(
        occluded_energy < clear_energy,
        "occluded energy {occluded_energy} must be below clear energy {clear_energy}"
    );
}

// ── direct_delay_shifts_output_later ─────────────────────────────────

#[test]
fn direct_delay_shifts_output_later() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.1);
    // Leave filters at their identity (allpass) defaults so this isolates the
    // fractional delay line: a unit impulse emerges exactly `delay` samples later.
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    input.set(0, 0, 1.0);
    let mut output = AudioBuffer::new(1, BLOCK as u16);
    node.process(&input, &mut output, &params(1.0, 50.0));

    assert!(
        (output.get(0, 50) - 1.0).abs() < 1e-4,
        "peak at sample 50 = {}",
        output.get(0, 50)
    );
    for i in 0..BLOCK {
        if i != 50 {
            assert!(
                output.get(0, i as u16).abs() < 1e-4,
                "unexpected energy at sample {i}"
            );
        }
    }
}

// ── reset_clears_delay_line_tail ─────────────────────────────────────

#[test]
fn reset_clears_delay_line_tail() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.1);
    // Fire an impulse that is still inside the node (100-sample delay).
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    input.set(0, 0, 1.0);
    let mut output = AudioBuffer::new(1, BLOCK as u16);
    node.process(&input, &mut output, &params(1.0, 100.0));
    assert!((output.get(0, 100) - 1.0).abs() < 1e-4);

    node.reset();

    let silence = AudioBuffer::new(1, BLOCK as u16);
    let mut out = AudioBuffer::new(1, BLOCK as u16);
    node.process(&silence, &mut out, &params(1.0, 0.0));
    for i in 0..BLOCK {
        assert_eq!(
            out.get(0, i as u16),
            0.0,
            "reset must clear the delay line (sample {i})"
        );
    }
}
