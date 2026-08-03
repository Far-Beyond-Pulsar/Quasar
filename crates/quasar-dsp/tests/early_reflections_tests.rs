//! Unit tests for the MONO early-reflection delay node (P2 scene pipeline).
//!
//! The node's contribution is mono: each tap's gain is the per-band average
//! and the pan is ignored (spatialized reflections land in P3).

use quasar_core::bands::Band8;
use quasar_core::param_exchange::{EarlyReflectionCoeffs, SpatialCoefficients};
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::early_reflections::EarlyReflectionDelayNode;
use quasar_dsp::node_graph::AudioNode;

const SAMPLES: usize = 256;

fn impulse() -> AudioBuffer {
    let mut buf = AudioBuffer::new(1, SAMPLES as u16);
    buf.set(0, 0, 1.0);
    buf
}

fn default_params() -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: 0.0,
        version: 0,
    }
}

// ── renders_mono_taps_at_expected_delays ─────────────────────────────

#[test]
fn renders_mono_taps_at_expected_delays() {
    let mut node = EarlyReflectionDelayNode::new(1, 48_000.0, 0.2, 16);
    node.update_reflections(&[
        EarlyReflectionCoeffs {
            azimuth: 0.7, // pan is ignored by the mono fold
            elevation: 0.0,
            delay_samples: 10.0,
            gain: Band8::splat(0.5),
        },
        EarlyReflectionCoeffs {
            azimuth: -1.2,
            elevation: 0.0,
            delay_samples: 40.0,
            gain: Band8::splat(0.5),
        },
    ]);

    let input = impulse();
    let mut output = AudioBuffer::new(1, SAMPLES as u16);
    node.process(&input, &mut output, &default_params());

    // A unit impulse produces 0.5 energy at exactly each tap delay.
    assert!(
        (output.get(0, 10) - 0.5).abs() < 1e-4,
        "tap at 10 = {}",
        output.get(0, 10)
    );
    assert!(
        (output.get(0, 40) - 0.5).abs() < 1e-4,
        "tap at 40 = {}",
        output.get(0, 40)
    );

    // Nothing before the first tap, between taps, or after the last tap.
    let mut energy = 0.0_f64;
    for i in 0..SAMPLES {
        let v = output.get(0, i as u16);
        energy += (v as f64) * (v as f64);
        if i != 10 && i != 40 {
            assert!(
                v.abs() < 1e-4,
                "unexpected energy at sample {i}: {v}"
            );
        }
    }
    // Total energy = 2 taps × 0.5².
    assert!((energy - 0.5).abs() < 1e-4, "total energy {energy}");
}
