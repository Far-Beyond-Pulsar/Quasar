//! #133: the "scene is not a closed surface" warning is raised at most once per backend
//! and the condition is queryable via `room_is_closed()`.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::IAcousticComputeBackend;
use quasar_core::scene::{AcousticMesh, AcousticScene};

fn open_quad() -> AcousticScene {
    let mut s = AcousticScene::new();
    let v = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]];
    s.add_mesh(AcousticMesh::new(1, v, vec![0, 1, 2, 0, 2, 3], 0));
    s
}

/// Consistently wound tetrahedron (closed, volume 1/6).
fn tetra() -> AcousticScene {
    let mut s = AcousticScene::new();
    let v = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    s.add_mesh(AcousticMesh::new(1, v, vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], 0));
    s
}

#[test]
fn repeated_updates_of_an_open_scene_warn_once() {
    let mut b = CpuSimdComputeBackend::new(open_quad(), CpuSimdConfig::default());
    assert!(!b.room_is_closed());
    assert_eq!(b.room_warning_count(), 1, "construction warns once");
    for _ in 0..20 {
        b.update_scene(&open_quad()).unwrap();
    }
    assert!(!b.room_is_closed());
    assert_eq!(b.room_warning_count(), 1, "dynamic updates must not warn again");
}

#[test]
fn closed_scene_never_warns_and_reports_closed() {
    let mut b = CpuSimdComputeBackend::new(tetra(), CpuSimdConfig::default());
    assert!(b.room_is_closed());
    for _ in 0..5 {
        b.update_scene(&tetra()).unwrap();
    }
    assert!(b.room_is_closed());
    assert_eq!(b.room_warning_count(), 0);
}

#[test]
fn flag_tracks_the_current_scene_and_empty_scene_is_quiet() {
    let mut b = CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default());
    assert_eq!(b.room_warning_count(), 0);
    b.update_scene(&tetra()).unwrap();
    assert!(b.room_is_closed());
    b.update_scene(&open_quad()).unwrap();
    assert!(!b.room_is_closed());
    b.update_scene(&tetra()).unwrap();
    assert!(b.room_is_closed());
    b.update_scene(&open_quad()).unwrap();
    assert_eq!(b.room_warning_count(), 1);
}
