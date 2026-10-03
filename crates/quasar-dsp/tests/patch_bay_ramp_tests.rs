//! #73: patch-bay edits are ramped (click-free) and structural edits keep other buses intact.

use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::patch_bay::{PatchBayBus, PatchBayNode, PatchEntry, MAX_PULLS_PER_OUTPUT};

const N: usize = 256;
const RAMP: u32 = 600;

fn dc(v: f32) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, N as u16);
    for i in 0..N {
        b.set(0, i as u16, v);
    }
    b
}

fn bay_with_pull(gain: f32) -> (PatchBayNode, Vec<AudioBuffer>) {
    let mut bay = PatchBayNode::new(1);
    bay.set_ramp_samples(RAMP);
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: gain });
    (bay, vec![AudioBuffer::new(1, N as u16)])
}

/// Render `blocks` blocks of a DC-1.0 source, returning the concatenated mono output.
fn render(bay: &mut PatchBayNode, out: &mut [AudioBuffer], blocks: usize) -> Vec<f32> {
    let src = dc(1.0);
    let mut all = Vec::new();
    for _ in 0..blocks {
        bay.process(&[&src], out);
        all.extend_from_slice(out[0].channel(0));
    }
    all
}

#[test]
fn before_the_first_block_edits_are_immediate() {
    let (mut bay, mut out) = bay_with_pull(0.5);
    let y = render(&mut bay, &mut out, 1);
    assert!(y.iter().all(|&v| (v - 0.5).abs() < 1e-6), "no ramp before the first render");
}

#[test]
fn set_pull_gain_is_click_free() {
    let (mut bay, mut out) = bay_with_pull(1.0);
    render(&mut bay, &mut out, 2);
    bay.set_pull_gain(0, 0, 0, 0.25);
    let y = render(&mut bay, &mut out, 4);
    let max_step = 0.75 / RAMP as f32;
    let mut prev = 1.0_f32;
    for (i, &v) in y.iter().enumerate() {
        assert!((v - prev).abs() <= max_step * 1.001 + 1e-6, "step {} at sample {i} exceeds ramp slope {max_step}", (v - prev).abs());
        prev = v;
    }
    assert!((y[y.len() - 1] - 0.25).abs() < 1e-6, "settles on the target");
    // And it really ramped (not instant): 100 samples in we are still between the two values.
    assert!(y[100] > 0.3 && y[100] < 0.95, "mid-ramp value {}", y[100]);
}

#[test]
fn retarget_mid_ramp_stays_continuous() {
    let (mut bay, mut out) = bay_with_pull(1.0);
    render(&mut bay, &mut out, 1);
    bay.set_pull_gain(0, 0, 0, 0.0);
    let a = render(&mut bay, &mut out, 1); // 256 samples into a 600 sample ramp
    bay.set_pull_gain(0, 0, 0, 1.0);
    let b = render(&mut bay, &mut out, 1);
    let jump = (b[0] - a[N - 1]).abs();
    assert!(jump <= 1.0 / RAMP as f32 * 1.01 + 1e-6, "retarget jump {jump}");
}

#[test]
fn new_pull_fades_in_and_removed_pull_fades_out_then_drops() {
    let (mut bay, mut out) = bay_with_pull(1.0);
    render(&mut bay, &mut out, 1);
    // Second tap added while running: starts at 0.
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: 1.0 }); // same tap: no-op
    bay.remove_pull(0, 0, 0);
    assert_eq!(bay.pulls(0).len(), 0, "fading tap is not a live pull");
    assert_eq!(bay.fading_pulls(0), 1);
    let y = render(&mut bay, &mut out, 4);
    assert!(y[0] < 1.0 && y[0] > 0.99, "starts at the current gain: {}", y[0]);
    assert!(y.windows(2).all(|w| w[1] <= w[0] + 1e-6), "monotone release");
    assert!(y[y.len() - 1].abs() < 1e-6, "silent after the ramp");
    assert_eq!(bay.fading_pulls(0), 0, "dropped once faded");

    // A pull added to the running bay fades in from 0.
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: 1.0 });
    let y = render(&mut bay, &mut out, 4);
    assert!(y[0] < 0.01, "attack starts at 0: {}", y[0]);
    assert!((y[y.len() - 1] - 1.0).abs() < 1e-6);
}

#[test]
fn reconnect_while_fading_revives_the_tap() {
    let (mut bay, mut out) = bay_with_pull(1.0);
    render(&mut bay, &mut out, 1);
    bay.remove_pull(0, 0, 0);
    let a = render(&mut bay, &mut out, 1);
    bay.set_pull(0, PatchEntry { source_idx: 0, channel: 0, gain_linear: 1.0 });
    let b = render(&mut bay, &mut out, 3);
    assert_eq!(bay.pulls(0).len(), 1);
    assert_eq!(bay.fading_pulls(0), 0);
    assert!((b[0] - a[N - 1]).abs() < 0.01, "continuous through the revive");
    assert!((b[b.len() - 1] - 1.0).abs() < 1e-6);
}

#[test]
fn structural_edits_keep_other_buses_and_remap_sources() {
    let mut bay = PatchBayNode::new(2);
    bay.set_ramp_samples(0);
    bay.set_pull(0, PatchEntry { source_idx: 2, channel: 0, gain_linear: 1.0 });
    bay.set_pull(1, PatchEntry { source_idx: 0, channel: 0, gain_linear: 0.5 });
    bay.set_pull(1, PatchEntry { source_idx: 2, channel: 1, gain_linear: 0.25 });

    // Insert a bus in the middle: existing buses keep their taps.
    assert!(bay.insert_output(1, PatchBayBus::new()).is_ok());
    assert_eq!(bay.num_outputs(), 3);
    assert_eq!(bay.pulls(0).len(), 1);
    assert_eq!(bay.pulls(1).len(), 0);
    assert_eq!(bay.pulls(2).len(), 2);

    // Unload source 0: its taps go, later sources shift down.
    bay.remove_source(0);
    assert_eq!(bay.pulls(0)[0].source_idx, 1);
    assert_eq!(bay.pulls(2).len(), 1);
    assert_eq!(bay.pulls(2)[0], PatchEntry { source_idx: 1, channel: 1, gain_linear: 0.25 });

    // Remove the middle bus.
    assert!(bay.remove_output(1).is_some());
    assert_eq!(bay.num_outputs(), 2);
    assert_eq!(bay.pulls(1).len(), 1);
    assert!(bay.remove_output(9).is_none());
}

#[test]
fn full_bus_evicts_a_fading_tap_instead_of_allocating() {
    let mut bay = PatchBayNode::new(1);
    bay.set_ramp_samples(RAMP);
    let mut out = vec![AudioBuffer::new(1, N as u16)];
    for ch in 0..MAX_PULLS_PER_OUTPUT {
        assert!(bay.set_pull(0, PatchEntry { source_idx: 0, channel: ch, gain_linear: 1.0 }));
    }
    assert!(!bay.set_pull(0, PatchEntry { source_idx: 1, channel: 0, gain_linear: 1.0 }), "full of live taps");
    render(&mut bay, &mut out, 1);
    bay.remove_pull(0, 0, 0);
    assert!(bay.set_pull(0, PatchEntry { source_idx: 1, channel: 0, gain_linear: 1.0 }), "evicts the fading tap");
}
