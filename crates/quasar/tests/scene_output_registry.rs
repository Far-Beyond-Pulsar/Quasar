//! P1 engine registry round-trip tests.
//!
//! These exercise the `SpatialAudioEngine` content-model API added in P1:
//! sources, scene outputs, patch-bay pulls, and listeners. They live here
//! (rather than in `quasar-core`) because `SpatialAudioEngine` is defined in
//! this crate.

use quasar_audio::SpatialAudioEngine;
use quasar_core::scene::Movability;
use quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};

fn source(path: &str, channels: usize) -> SourceConfig {
    SourceConfig {
        path: path.to_string(),
        channels,
    }
}

#[test]
fn registry_round_trip() {
    let mut engine = SpatialAudioEngine::new(8, 48000.0, 15.0);

    // Load 2 sources → sequential IDs.
    let src_a = engine.load_source(source("a.wav", 2)).expect("load source");
    let src_b = engine.load_source(source("b.wav", 2)).expect("load source");
    assert_eq!(src_a.0, 0);
    assert_eq!(src_b.0, 1);

    // Reject zero-channel sources.
    assert!(engine.load_source(source("bad.wav", 0)).is_err());

    // Add a scene output with initial pulls.
    let out = engine.add_scene_output(SceneOutputConfig {
        position: [1.0, 2.0, 3.0],
        orientation: None,
        directivity: 0.0,
        pulls: vec![ChannelPull::new(src_a, 0, -3.0)],
        movability: Movability::Static,
    });
    assert_eq!(out.0, 0);

    // connect_pull appends a new tap and dedupes existing ones.
    engine.connect_pull(out, ChannelPull::new(src_b, 1, -6.0));
    engine.connect_pull(out, ChannelPull::new(src_a, 0, -9.0)); // dedupe → gain replaced
    let pulls = &engine.scene_outputs()[0].pulls;
    assert_eq!(pulls.len(), 2);
    assert!((pulls[0].gain_db - -9.0).abs() < 1e-6, "dedupe replaced gain");
    assert_eq!(pulls[1].source_id, src_b);
    assert_eq!(pulls[1].channel, 1);

    // set_pull_gain updates an existing pull.
    engine.set_pull_gain(out, src_a, 0, 0.0);
    assert!((engine.scene_outputs()[0].pulls[0].gain_db - 0.0).abs() < 1e-6);

    // set_pull_gain on an absent tap is a no-op (no panic).
    engine.set_pull_gain(out, src_a, 5, 12.0);
    assert_eq!(engine.scene_outputs()[0].pulls.len(), 2);

    // Add a listener → sequential IDs.
    let listener = engine.add_listener(ListenerConfig {
        position: [0.0, 0.0, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    assert_eq!(listener.0, 0);

    // Mutations are reflected in stored state.
    engine.set_scene_output_position(out, [9.0, 9.0, 9.0]);
    assert_eq!(engine.scene_outputs()[0].position, [9.0, 9.0, 9.0]);

    engine.update_listener(listener, [1.0, 1.6, -2.0], [0.0, 0.0, -1.0]);
    assert_eq!(engine.listeners()[0].position, [1.0, 1.6, -2.0]);
    assert_eq!(engine.listeners()[0].heading, [0.0, 0.0, -1.0]);

    // disconnect_pull removes every matching tap.
    engine.disconnect_pull(out, src_a, 0);
    assert_eq!(engine.scene_outputs()[0].pulls.len(), 1);
    engine.disconnect_pull(out, src_b, 1);
    assert!(engine.scene_outputs()[0].pulls.is_empty());

    // Remove listener and output.
    engine.remove_listener(listener);
    assert!(engine.listeners().is_empty());
    engine.remove_scene_output(out);
    assert!(engine.scene_outputs().is_empty());

    // unload_source drops the source and any pulls referencing it.
    let out2 = engine.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out2, ChannelPull::new(src_a, 0, 0.0));
    engine.connect_pull(out2, ChannelPull::new(src_b, 0, 0.0));
    engine.unload_source(src_a);
    assert_eq!(engine.sources().len(), 1);
    assert_eq!(engine.scene_outputs()[0].pulls.len(), 1);
    assert_eq!(engine.scene_outputs()[0].pulls[0].source_id, src_b);
}

// ── registry_index_panics ────────────────────────────────────────────

#[test]
#[should_panic(expected = "invalid SceneOutputId")]
fn invalid_scene_output_id_panics() {
    let mut engine = SpatialAudioEngine::new(1, 48000.0, 15.0);
    engine.set_scene_output_position(quasar_core::scene_output::SceneOutputId(99), [1.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "invalid SourceId")]
fn invalid_source_id_panics() {
    let mut engine = SpatialAudioEngine::new(1, 48000.0, 15.0);
    engine.unload_source(quasar_core::scene_output::SourceId(5));
}

#[test]
#[should_panic(expected = "invalid ListenerId")]
fn invalid_listener_id_panics() {
    let mut engine = SpatialAudioEngine::new(1, 48000.0, 15.0);
    engine.update_listener(
        quasar_core::scene_output::ListenerId(5),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    );
}

#[test]
fn add_ids_are_sequential_across_kinds() {
    let mut engine = SpatialAudioEngine::new(1, 48000.0, 15.0);
    let s0 = engine.load_source(source("s.wav", 1)).unwrap();
    let s1 = engine.load_source(source("s.wav", 1)).unwrap();
    let o0 = engine.add_scene_output(SceneOutputConfig::default());
    let o1 = engine.add_scene_output(SceneOutputConfig::default());
    let l0 = engine.add_listener(ListenerConfig::default());
    let l1 = engine.add_listener(ListenerConfig::default());
    assert_eq!((s0.0, s1.0), (0, 1));
    assert_eq!((o0.0, o1.0), (0, 1));
    assert_eq!((l0.0, l1.0), (0, 1));
}
