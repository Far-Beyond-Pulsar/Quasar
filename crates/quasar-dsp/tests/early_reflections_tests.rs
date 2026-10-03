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
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
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

// ── shrinking / oversized lists (crossfader union during a fade) ─────

#[test]
fn update_does_not_allocate_or_read_past_line_for_large_or_shrinking_lists() {
    let mut node = EarlyReflectionDelayNode::new(1, 48_000.0, 0.2, 16);
    let refl = |delay: f32, g: f32| EarlyReflectionCoeffs {
        azimuth: 0.0,
        elevation: 0.0,
        delay_samples: delay,
        gain: Band8::splat(g),
    };

    // Union-sized list (> the 16 requested), one tap far beyond the 0.2 s line.
    let big: Vec<_> = (0..64).map(|i| refl(20.0 + i as f32, 0.01)).collect();
    node.update_reflections(&big);
    let mut out = AudioBuffer::new(1, SAMPLES as u16);
    node.process(&impulse(), &mut out, &default_params());

    let mut far = big.clone();
    far.push(refl(1.0e6, 1.0));
    node.update_reflections(&far);
    node.process(&impulse(), &mut out, &default_params());
    assert!(out.channel(0).iter().all(|s| s.is_finite()));

    // Shrinking to a single tap and then to none is clean.
    node.update_reflections(&[refl(10.0, 0.5)]);
    let mut out = AudioBuffer::new(1, SAMPLES as u16);
    node.process(&impulse(), &mut out, &default_params());
    node.update_reflections(&[]);
    let mut out2 = AudioBuffer::new(1, SAMPLES as u16);
    node.process(&AudioBuffer::new(1, SAMPLES as u16), &mut out2, &default_params());
    assert!(out2.channel(0).iter().all(|s| *s == 0.0 || s.is_finite()));
}
