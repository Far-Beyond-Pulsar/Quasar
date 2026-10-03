//! The demo's cathedral (examples/basic): six quads exactly as the demo builds them
//! (the quads are NOT consistently wound) plus its four boxed columns. Reports whether
//! the backend treats it as a closed surface, which decides whether the closed-room
//! shortcuts (outside the shell) can apply.

mod common;

use common::Opaque;
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig, RoomSide};
use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::scene::{AcousticMesh, AcousticScene};

fn demo_scene() -> AcousticScene {
    let q = |id: u64, p: [[f32; 3]; 4]| AcousticMesh::new(id, p.to_vec(), vec![0, 1, 2, 0, 2, 3], 0);
    let mut s = AcousticScene::new();
    s.add_mesh(q(1, [[-11., 0., -28.], [11., 0., -28.], [11., 0., 28.], [-11., 0., 28.]]));
    s.add_mesh(q(2, [[-11., 0., -28.], [-11., 0., 28.], [-11., 21., 28.], [-11., 21., -28.]]));
    s.add_mesh(q(3, [[11., 0., -28.], [11., 0., 28.], [11., 21., 28.], [11., 21., -28.]]));
    s.add_mesh(q(8, [[-11., 21., -28.], [11., 21., -28.], [11., 21., 28.], [-11., 21., 28.]]));
    s.add_mesh(q(9, [[-11., 0., -28.], [11., 0., -28.], [11., 21., -28.], [-11., 21., -28.]]));
    s.add_mesh(q(10, [[-11., 0., 28.], [11., 0., 28.], [11., 21., 28.], [-11., 21., 28.]]));
    for (i, cz) in [-22.0_f32, 18.0].into_iter().enumerate() {
        for (j, cx) in [-5.5_f32, 5.5].into_iter().enumerate() {
            s.add_mesh(AcousticMesh::new(
                4 + (i * 2 + j) as u64,
                vec![
                    [cx - 0.325, 0.0, cz - 0.325], [cx + 0.325, 0.0, cz - 0.325], [cx + 0.325, 0.0, cz + 0.325], [cx - 0.325, 0.0, cz + 0.325],
                    [cx - 0.325, 20.0, cz - 0.325], [cx + 0.325, 20.0, cz - 0.325], [cx + 0.325, 20.0, cz + 0.325], [cx - 0.325, 20.0, cz + 0.325],
                ],
                vec![0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2],
                0,
            ));
        }
    }
    s
}

fn backend(shortcuts: bool) -> CpuSimdComputeBackend {
    CpuSimdComputeBackend::new(demo_scene(), CpuSimdConfig {
        max_reflection_order: 3,
        diffuse_rays_per_query: 128,
        max_reflection_distance: 60.0,
        closed_room_shortcuts: shortcuts,
        ..CpuSimdConfig::default()
    })
}

const SPEAKERS: [[f32; 3]; 8] = [
    [-7.0, 5.5, -12.0], [7.0, 5.5, -12.0], [0.0, 3.0, -12.0], [0.0, 0.3, -7.0],
    [-7.0, 2.0, 12.0], [7.0, 2.0, 12.0], [-7.0, 0.5, -12.0], [7.0, 0.5, -12.0],
];

/// The demo shell is NOT consistently wound (`room_is_closed()` is false: the late-reverb
/// volume falls back to the bounding box) but it is watertight, which is all the
/// parity-based shortcuts need.
#[test]
fn demo_shell_is_watertight_but_not_consistently_wound() {
    let b = backend(true);
    assert!(!b.room_is_closed());
    assert_eq!(b.room_side([0.0, 1.6, 0.0]), RoomSide::Inside);
    assert_eq!(b.room_side([0.0, 1.6, -10.5]), RoomSide::Inside);
    for p in [[-12.0, 1.6, 0.0], [0.0, 22.0, 0.0], [0.0, 1.6, 40.0], [30.0, 1.6, 0.0], [0.0, -2.0, 0.0]] {
        assert_eq!(b.room_side(p), RoomSide::Outside, "{p:?}");
    }
}

#[test]
fn demo_outside_listener_is_cheap_and_identical_to_exhaustive_search() {
    let (fast, slow) = (backend(true), backend(false));
    let queries = |l: [f32; 3]| -> Vec<SpatialQuery> {
        SPEAKERS.iter().enumerate().map(|(i, &s)| SpatialQuery { source_position: s, listener_position: l, source_id: i as u32 }).collect()
    };
    println!("demo, 8 emitters per update: rays (exhaustive -> shortcuts)");
    for (name, l) in [
        ("inside, origin", [0.0, 1.6, 0.0]),
        ("inside, near stage", [0.0, 1.6, -10.5]),
        ("just outside left wall", [-11.5, 1.6, 0.0]),
        ("outside 5 m", [-16.0, 1.6, 0.0]),
        ("outside far", [-80.0, 1.6, 10.0]),
        ("above ceiling", [0.0, 25.0, 0.0]),
    ] {
        let q = queries(l);
        slow.reset_ray_counter();
        let rs = slow.query_spatial(&q, &Opaque);
        let ns = slow.rays_traced();
        fast.reset_ray_counter();
        let rf = fast.query_spatial(&q, &Opaque);
        let nf = fast.rays_traced();
        println!("  {name:<24} {ns:>7} -> {nf:>6}");
        for (a, b) in rs.iter().zip(&rf) {
            assert_eq!(a.direct_path.attenuation.0, b.direct_path.attenuation.0, "{name}");
            assert_eq!(a.direct_path.occlusion.0, b.direct_path.occlusion.0, "{name}");
            assert_eq!(a.early_reflections.len(), b.early_reflections.len(), "{name}");
            for (x, y) in a.early_reflections.iter().zip(&b.early_reflections) {
                assert_eq!((x.delay_samples, x.direction, x.gain.0), (y.delay_samples, y.direction, y.gain.0), "{name}");
            }
        }
        if name.contains("outside") || name.contains("above") {
            assert!(rs.iter().all(|r| r.early_reflections.is_empty()), "{name}");
            assert!(nf <= 8 * 200, "{name}: {nf} rays for 8 emitters");
        }
    }
}
