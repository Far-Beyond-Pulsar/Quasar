//! P1 core-type tests: IDs, configs, defaults, and patch-bay helpers.
//!
//! The engine-level registry round-trip (`SpatialAudioEngine`) lives in
//! `crates/quasar/tests/scene_output_registry.rs` because the engine is
//! defined in the `quasar` crate, which this crate must not depend on.

use std::collections::HashSet;

use quasar_core::scene::Movability;
use quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig, SourceId,
    ListenerId, SceneOutputId,
};

// ── id_types_are_hashable ────────────────────────────────────────────

#[test]
fn id_types_are_hashable() {
    let mut set = HashSet::new();
    set.insert(SourceId(7));
    set.insert(SourceId(8));
    assert!(set.contains(&SourceId(7)));
    assert!(!set.contains(&SourceId(6)));
    assert_eq!(SourceId(7), SourceId(7));
    assert_ne!(SourceId(7), SourceId(8));

    assert_eq!(SceneOutputId(3), SceneOutputId(3));
    assert_ne!(SceneOutputId(3), SceneOutputId(4));
    assert_eq!(ListenerId(1), ListenerId(1));
    assert_ne!(ListenerId(1), ListenerId(2));
}

// ── channel_pull_new ─────────────────────────────────────────────────

#[test]
fn channel_pull_new() {
    let pull = ChannelPull::new(SourceId(2), 3, -6.0);
    assert_eq!(pull.source_id, SourceId(2));
    assert_eq!(pull.channel, 3);
    assert!((pull.gain_db - -6.0).abs() < 1e-6);

    // Copy semantics keep source_id usable after construction.
    let pull2 = pull;
    assert_eq!(pull.source_id, pull2.source_id);
}

// ── source_config_fields ─────────────────────────────────────────────

#[test]
fn source_config_fields() {
    let cfg = SourceConfig {
        path: "assets/8_Channel_ID.wav".to_string(),
        channels: 8,
    };
    assert_eq!(cfg.path, "assets/8_Channel_ID.wav");
    assert_eq!(cfg.channels, 8);
}

// ── scene_output_config_default ──────────────────────────────────────

#[test]
fn scene_output_config_default() {
    let cfg = SceneOutputConfig::default();
    assert_eq!(cfg.position, [0.0, 0.0, 0.0]);
    assert_eq!(cfg.orientation, None);
    assert_eq!(cfg.directivity, 0.0);
    assert!(cfg.pulls.is_empty());
    assert_eq!(cfg.movability, Movability::Static);
}

// ── scene_output_config_new ──────────────────────────────────────────

#[test]
fn scene_output_config_new() {
    let cfg = SceneOutputConfig::new([1.0, 2.0, 3.0], Movability::Dynamic);
    assert_eq!(cfg.position, [1.0, 2.0, 3.0]);
    assert_eq!(cfg.movability, Movability::Dynamic);
    assert_eq!(cfg.directivity, 0.0);
    assert!(cfg.pulls.is_empty());
}

// ── scene_output_config_round_trip_fields ────────────────────────────

#[test]
fn scene_output_config_round_trip_fields() {
    let cfg = SceneOutputConfig {
        position: [4.0, 5.0, 6.0],
        orientation: Some([0.0, 0.0, -1.0]),
        directivity: 0.75,
        pulls: vec![ChannelPull::new(SourceId(1), 0, -3.0)],
        movability: Movability::Streaming,
    };
    assert_eq!(cfg.orientation, Some([0.0, 0.0, -1.0]));
    assert_eq!(cfg.pulls.len(), 1);
    assert_eq!(cfg.pulls[0].source_id, SourceId(1));
    assert_eq!(cfg.movability, Movability::Streaming);
}

// ── listener_config_default ──────────────────────────────────────────

#[test]
fn listener_config_default() {
    let cfg = ListenerConfig::default();
    assert_eq!(cfg.position, [0.0, 0.0, 0.0]);
    assert_eq!(cfg.heading, [0.0, 0.0, -1.0]);
    assert_eq!(cfg.physical_layout, PhysicalOutputLayout::Stereo);
}

// ── listener_config_custom_layout ────────────────────────────────────

#[test]
fn listener_config_custom_layout() {
    let cfg = ListenerConfig {
        position: [1.0, 1.6, 2.0],
        heading: [1.0, 0.0, 0.0],
        physical_layout: PhysicalOutputLayout::Custom {
            positions: vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        },
    };
    assert_eq!(cfg.position, [1.0, 1.6, 2.0]);
    match &cfg.physical_layout {
        PhysicalOutputLayout::Custom { positions } => assert_eq!(positions.len(), 2),
        other => panic!("expected Custom layout, got {other:?}"),
    }
}

// ── movability_variants ──────────────────────────────────────────────

#[test]
fn movability_variants() {
    assert_ne!(Movability::Static, Movability::Dynamic);
    assert_ne!(Movability::Dynamic, Movability::Streaming);
    assert_eq!(Movability::Static, Movability::Static);
}

// ── physical_layout_equality ─────────────────────────────────────────

#[test]
fn physical_layout_equality() {
    assert_eq!(
        PhysicalOutputLayout::Surround51,
        PhysicalOutputLayout::Surround51
    );
    assert_ne!(PhysicalOutputLayout::Quad, PhysicalOutputLayout::Hrtf);
    assert_eq!(
        PhysicalOutputLayout::Custom {
            positions: vec![[0.0, 1.0, 0.0]],
        },
        PhysicalOutputLayout::Custom {
            positions: vec![[0.0, 1.0, 0.0]],
        },
    );
}
