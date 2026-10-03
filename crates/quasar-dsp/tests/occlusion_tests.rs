//! Unit tests for the air-absorption / occlusion node (P2 scene pipeline).
//!
//! The node renders `SpatialCoefficients::direct_gain` (8 linear band gains)
//! through a per-band EQ cascade and `direct_delay_samples` through a fractional
//! delay line (the response / Doppler / click tests live in
//! `occlusion_filter_tests.rs`). These tests cover the basic contract: occlusion
//! lowers the energy, the delay moves an impulse later by exactly that many
//! samples, and `reset` clears the delay tail.

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
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version: 0,
    }
}

/// Fire a single unit impulse (block 0) and sum output energy over `blocks`
/// 256-sample blocks, rendering with band gain `gain` on every band.
fn impulse_energy(node: &mut AirAbsorptionOcclusionNode, blocks: usize, gain: f32) -> f64 {
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
        node.process(&input, &mut output, &params(gain, 0.0));
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

    // Clear line of sight: unity on every band.
    let clear_energy = impulse_energy(&mut node, 16, 1.0);

    // Fully occluded: the band gains come from `params.direct_gain` (the node is
    // driven by SpatialCoefficients; `update_occlusion` only primes the starting
    // point). This test used to set the attenuation through `update_occlusion`
    // and pass unity gain in params, which only held while `update_occlusion`
    // was the stub's attenuation channel.
    node.reset();
    let occluded_energy = impulse_energy(&mut node, 16, 0.02);

    assert!(clear_energy > 0.0, "clear path must pass energy");
    assert!(
        occluded_energy < clear_energy,
        "occluded energy {occluded_energy} must be below clear energy {clear_energy}"
    );
    // 0.02 amplitude = -34 dB: the energy must fall by about that much, not just a little.
    assert!(
        occluded_energy < clear_energy * 1e-2,
        "occluded energy {occluded_energy} should be ~-34 dB below {clear_energy}"
    );
}

// ── direct_delay_shifts_output_later ─────────────────────────────────

#[test]
fn direct_delay_shifts_output_later() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.1);
    // Flat unity gains make the EQ an identity, so this isolates the fractional
    // delay line: a unit impulse emerges exactly `delay` samples later.
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

// ── update_occlusion primes the starting point ───────────────────────

#[test]
fn update_occlusion_primes_without_ramping() {
    let mut node = AirAbsorptionOcclusionNode::new(1, SR, 0.1);
    node.update_occlusion(&Band8::splat(0.5), 20.0);
    // Params equal to the primed values: the very first block already renders
    // at gain 0.5 and delay 20 (no ramp from silence / zero delay).
    let mut input = AudioBuffer::new(1, BLOCK as u16);
    input.set(0, 0, 1.0);
    let mut output = AudioBuffer::new(1, BLOCK as u16);
    node.process(&input, &mut output, &params(0.5, 20.0));
    assert!((output.get(0, 20) - 0.5).abs() < 1e-3, "got {}", output.get(0, 20));
}
