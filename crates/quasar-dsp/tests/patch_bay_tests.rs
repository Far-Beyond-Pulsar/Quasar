//! Unit tests for the zero-alloc patch-bay mixer (P2 scene pipeline).

use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::patch_bay::{PatchBayNode, PatchEntry};

const SAMPLES: usize = 256;

/// Build a constant-valued (DC) source buffer with `channels` channels.
fn dc(value: f32, channels: usize) -> AudioBuffer {
    let mut buf = AudioBuffer::new(channels as u16, SAMPLES as u16);
    for ch in 0..channels {
        for i in 0..SAMPLES {
            buf.set(ch as u16, i as u16, value);
        }
    }
    buf
}

/// Build a DC source buffer with a distinct value per channel.
fn dc_channels(values: &[f32]) -> AudioBuffer {
    let mut buf = AudioBuffer::new(values.len() as u16, SAMPLES as u16);
    for (ch, &value) in values.iter().enumerate() {
        for i in 0..SAMPLES {
            buf.set(ch as u16, i as u16, value);
        }
    }
    buf
}

/// dB → linear amplitude (mirrors the engine's conversion).
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ── mixes_pulls_into_mono_outputs ────────────────────────────────────

#[test]
fn mixes_pulls_into_mono_outputs() {
    // Source A: 2 channels (ch0 = 1.0, ch1 = 0.5). Source B: 1 channel (ch0 = 2.0).
    let src_a = dc_channels(&[1.0, 0.5]);
    let src_b = dc(2.0, 1);

    let mut bay = PatchBayNode::new(2);
    bay.set_pull(
        0,
        PatchEntry {
            source_idx: 0,
            channel: 0,
            gain_linear: db_to_linear(0.0), // A ch0, 0 dB → 1.0
        },
    );
    bay.set_pull(
        0,
        PatchEntry {
            source_idx: 0,
            channel: 1,
            gain_linear: db_to_linear(-6.0), // A ch1, -6 dB → ~0.501
        },
    );
    bay.set_pull(
        0,
        PatchEntry {
            source_idx: 1,
            channel: 0,
            gain_linear: db_to_linear(6.0), // B ch0, +6 dB → ~1.995
        },
    );
    bay.set_pull(
        1,
        PatchEntry {
            source_idx: 1,
            channel: 0,
            gain_linear: db_to_linear(0.0), // output 1: B ch0, 0 dB
        },
    );

    let sources: [&AudioBuffer; 2] = [&src_a, &src_b];
    let mut outputs = [AudioBuffer::new(1, SAMPLES as u16), AudioBuffer::new(1, SAMPLES as u16)];
    bay.process(&sources, &mut outputs);

    let expected0 = 1.0 * db_to_linear(0.0) + 0.5 * db_to_linear(-6.0) + 2.0 * db_to_linear(6.0);
    let expected1 = 2.0 * db_to_linear(0.0);

    for i in 0..SAMPLES {
        let v0 = outputs[0].get(0, i as u16);
        let v1 = outputs[1].get(0, i as u16);
        assert!(
            (v0 - expected0).abs() < 1e-3,
            "output0[{i}] = {v0}, expected {expected0}"
        );
        assert!(
            (v1 - expected1).abs() < 1e-3,
            "output1[{i}] = {v1}, expected {expected1}"
        );
    }
}

// ── skips_out_of_range_pulls_silently ────────────────────────────────

#[test]
fn skips_out_of_range_pulls_silently() {
    let src = dc(1.0, 1);
    let mut bay = PatchBayNode::new(1);
    // Channel 99 does not exist on a 1-channel source.
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 99, gain_linear: 1.0 });
    // Source index 99 does not exist at all.
    bay.set_pull(0, PatchEntry { source_idx: 99, channel: 0, gain_linear: 1.0 });

    let sources: [&AudioBuffer; 1] = [&src];
    let mut outputs = [AudioBuffer::new(1, SAMPLES as u16)];
    bay.process(&sources, &mut outputs);

    for i in 0..SAMPLES {
        assert_eq!(outputs[0].get(0, i as u16), 0.0, "unexpected output at {i}");
    }
}

// ── set_pull_replaces_identical_tap ──────────────────────────────────

#[test]
fn set_pull_replaces_identical_tap() {
    let src = dc(1.0, 1);
    let mut bay = PatchBayNode::new(1);
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: 1.0 });
    // Same (source, channel) replaces the gain rather than stacking a 2nd tap.
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: 0.5 });
    assert_eq!(bay.pulls(0).len(), 1, "identical tap must be replaced, not appended");

    let sources: [&AudioBuffer; 1] = [&src];
    let mut outputs = [AudioBuffer::new(1, SAMPLES as u16)];
    bay.process(&sources, &mut outputs);
    for i in 0..SAMPLES {
        assert!((outputs[0].get(0, i as u16) - 0.5).abs() < 1e-6);
    }
}

// ── set_pull_gain_updates_existing_tap ───────────────────────────────

#[test]
fn set_pull_gain_updates_existing_tap() {
    let src = dc(1.0, 1);
    let mut bay = PatchBayNode::new(1);
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: 1.0 });
    bay.set_pull_gain(0, 0, 0, 0.25);

    let sources: [&AudioBuffer; 1] = [&src];
    let mut outputs = [AudioBuffer::new(1, SAMPLES as u16)];
    bay.process(&sources, &mut outputs);
    for i in 0..SAMPLES {
        assert!((outputs[0].get(0, i as u16) - 0.25).abs() < 1e-6);
    }
}
