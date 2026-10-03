//! Ray budget of the CPU solver and the closed-room shortcuts: how many BVH rays one
//! `query_spatial` traces for a listener inside / outside a closed hall with columns, and
//! the proof that the shortcuts do not change any result. Run the report with
//! `cargo test -p quasar-backends --release --test ray_budget_tests -- --nocapture`.

mod common;

use common::{hall, Leaky, Opaque, INSIDE, SRC};
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig, RoomSide};
use quasar_backends::debug_capture::{CaptureDetail, DebugRayKind, RejectReason, MAX_STORED_RAYS, MAX_STORED_REJECTED};
use quasar_core::backend::{IAcousticComputeBackend, MaterialProvider, SpatialQuery, SpatialQueryResult};
use quasar_core::scene::{AcousticMesh, AcousticScene};

fn backend(shortcuts: bool) -> CpuSimdComputeBackend {
    CpuSimdComputeBackend::new(hall(), CpuSimdConfig { closed_room_shortcuts: shortcuts, ..CpuSimdConfig::default() })
}

fn run(b: &CpuSimdComputeBackend, m: &dyn MaterialProvider, src: [f32; 3], lis: [f32; 3]) -> (SpatialQueryResult, u64) {
    b.reset_ray_counter();
    let q = SpatialQuery { source_position: src, listener_position: lis, source_id: 3 };
    let r = b.query_spatial(&[q], m).remove(0);
    (r, b.rays_traced())
}

fn assert_same(a: &SpatialQueryResult, b: &SpatialQueryResult, what: &str) {
    assert_eq!(a.direct_path.attenuation.0, b.direct_path.attenuation.0, "{what}: direct attenuation");
    assert_eq!(a.direct_path.occlusion.0, b.direct_path.occlusion.0, "{what}: occlusion");
    assert_eq!(a.direct_path.occluded, b.direct_path.occluded, "{what}");
    assert_eq!(a.direct_path.delay_samples, b.direct_path.delay_samples, "{what}");
    assert_eq!(a.late_reverb.t60.0, b.late_reverb.t60.0, "{what}: t60");
    assert_eq!(a.late_reverb.late_loudness_db, b.late_reverb.late_loudness_db, "{what}: late level");
    assert_eq!(a.early_reflections.len(), b.early_reflections.len(), "{what}: reflection count");
    for (x, y) in a.early_reflections.iter().zip(&b.early_reflections) {
        assert_eq!((x.order, x.direction, x.delay_samples, x.gain.0), (y.order, y.direction, y.delay_samples, y.gain.0), "{what}");
    }
}

const OUTSIDE: [[f32; 3]; 5] = [
    [-0.3, 1.7, 15.0],
    [-5.0, 1.7, 15.0],
    [-100.0, 1.7, 15.0],
    [10.0, 14.0, 15.0],
    [23.0, 3.0, 36.0],
];

/// Documented bound: parity bookkeeping (2 endpoints x 3..5 rays x a few crossings) plus
/// 13 occlusion rays that each cross the shell and a column or two.
const OUTSIDE_RAY_BOUND: u64 = 200;

// ── ray counter + report ──────────────────────────────────────────────

#[test]
fn counter_counts_every_ray_without_debug_capture() {
    let b = backend(true);
    assert!(b.room_is_closed(), "test hall must be a closed shell");
    assert!(!b.debug_capture().is_enabled());
    let (_, n) = run(&b, &Opaque, SRC, INSIDE);
    assert!(n > 0 && n == b.rays_traced());
    b.reset_ray_counter();
    assert_eq!(b.rays_traced(), 0);
}

#[test]
fn ray_budget_report() {
    let cases: [(&str, [f32; 3]); 7] = [
        ("inside (far)", INSIDE),
        ("inside (behind column)", [6.0, 1.7, 15.0]),
        ("just outside wall (x=-0.3)", [-0.3, 1.7, 15.0]),
        ("outside 5 m (x=-5)", [-5.0, 1.7, 15.0]),
        ("outside far (x=-100)", [-100.0, 1.7, 15.0]),
        ("above ceiling (y=14)", [10.0, 14.0, 15.0]),
        ("outside corner (-3,1.7,-3)", [-3.0, 1.7, -3.0]),
    ];
    println!("rays traced per query (source inside, listener as listed):");
    for (name, l) in cases {
        let (_, off) = run(&backend(false), &Opaque, SRC, l);
        let (_, on) = run(&backend(true), &Opaque, SRC, l);
        println!("  {name:<30} exhaustive {off:>6}   shortcuts {on:>6}");
    }
}

#[test]
fn demo_update_ray_budget_report() {
    // One demo update = 8 emitter queries against the same listener.
    let emitters: Vec<[f32; 3]> = (0..8).map(|i| [3.0 + 2.0 * i as f32, 1.5, 4.0 + 3.0 * i as f32]).collect();
    println!("rays per demo update (8 emitters inside):");
    for (name, l) in [("listener inside", INSIDE), ("listener 5 m outside", [-5.0, 1.7, 15.0]), ("listener above ceiling", [10.0, 14.0, 15.0])] {
        let mut row = String::new();
        for shortcuts in [false, true] {
            let b = backend(shortcuts);
            let qs: Vec<SpatialQuery> = emitters.iter().enumerate().map(|(i, &e)| SpatialQuery { source_position: e, listener_position: l, source_id: i as u32 }).collect();
            b.reset_ray_counter();
            b.query_spatial(&qs, &Opaque);
            row += &format!("  shortcuts={shortcuts}: {:>7}", b.rays_traced());
        }
        println!("  {name:<24}{row}");
    }
}

// ── closed-room shortcuts ─────────────────────────────────────────────

#[test]
fn outside_queries_are_identical_to_exhaustive_search_and_cheap() {
    let (fast, slow) = (backend(true), backend(false));
    for l in OUTSIDE {
        let (a, na) = run(&fast, &Opaque, SRC, l);
        let (b, nb) = run(&slow, &Opaque, SRC, l);
        assert_same(&a, &b, &format!("listener {l:?}"));
        assert!(b.early_reflections.is_empty(), "the exhaustive tracer also finds nothing outside {l:?}");
        assert!(na <= OUTSIDE_RAY_BOUND, "{l:?}: {na} rays > {OUTSIDE_RAY_BOUND}");
        assert!(na * 10 < nb, "{l:?}: {na} rays vs exhaustive {nb}");
        // source outside, listener inside is the same situation
        let (c, nc) = run(&fast, &Opaque, l, INSIDE);
        let (d, _) = run(&slow, &Opaque, l, INSIDE);
        assert_same(&c, &d, &format!("source {l:?}"));
        assert!(nc <= OUTSIDE_RAY_BOUND);
    }
}

#[test]
fn inside_and_two_outside_endpoint_results_are_unchanged() {
    let (fast, slow) = (backend(true), backend(false));
    for (s, l) in [
        (SRC, INSIDE),
        (SRC, [6.0, 1.7, 15.0]),
        ([3.0, 2.0, 12.0], [17.0, 1.2, 28.0]),
        ([6.0, 1.7, 7.0], [6.0, 1.7, 11.0]),    // column between
        ([-5.0, 1.7, 15.0], [25.0, 1.7, 15.0]), // both outside: reflections off the exterior stay
        ([-5.0, 1.7, 15.0], [-8.0, 3.0, 12.0]),
    ] {
        let (a, na) = run(&fast, &Opaque, s, l);
        let (b, nb) = run(&slow, &Opaque, s, l);
        assert_same(&a, &b, &format!("{s:?} -> {l:?}"));
        // The shortcut costs at most the parity bookkeeping rays when it does not apply.
        assert!(na <= nb + 60, "{s:?} -> {l:?}: {na} vs {nb}");
    }
}

#[test]
fn shortcuts_do_not_apply_to_walls_that_transmit() {
    let (fast, slow) = (backend(true), backend(false));
    for l in [[-5.0, 1.7, 15.0], [10.0, 14.0, 15.0]] {
        let (a, _) = run(&fast, &Leaky, SRC, l);
        let (b, _) = run(&slow, &Leaky, SRC, l);
        assert_same(&a, &b, "leaky walls");
        assert!(a.direct_path.attenuation.0.iter().all(|&g| g > 0.0), "transmitted direct sound is audible");
    }
}

#[test]
fn room_side_classifies_points_and_is_unknown_for_open_scenes() {
    let b = backend(true);
    for p in [[10.0, 1.7, 15.0], [6.0, 6.0, 9.7], [0.2, 0.2, 0.2], [19.9, 11.9, 29.9]] {
        assert_eq!(b.room_side(p), RoomSide::Inside, "{p:?}");
    }
    for p in [[-0.3, 1.7, 15.0], [10.0, 12.2, 15.0], [10.0, -0.3, 15.0], [50.0, 50.0, 50.0], [20.3, 6.0, 3.0]] {
        assert_eq!(b.room_side(p), RoomSide::Outside, "{p:?}");
    }
    // Exactly on a wall plane or in line with a column edge must not crash the vote.
    for p in [[0.0, 1.0, 15.0], [5.5, 1.0, 8.5], [10.0, 0.0, 15.0]] {
        let _ = b.room_side(p);
    }
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(1, vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]], vec![0, 1, 2, 0, 2, 3], 0));
    let open = CpuSimdComputeBackend::new(s, CpuSimdConfig::default());
    assert_eq!(open.room_side([1.0, 1.0, 1.0]), RoomSide::Unknown);
    assert_eq!(open.room_side([100.0, 1.0, 1.0]), RoomSide::Unknown);
}

// ── debug capture budget ──────────────────────────────────────────────

#[test]
fn default_capture_stores_no_probe_rays_but_counts_everything() {
    let b = backend(true);
    let cap = b.debug_capture();
    cap.set_enabled(true);
    let qs = [
        SpatialQuery { source_position: SRC, listener_position: INSIDE, source_id: 7 },
        SpatialQuery { source_position: SRC, listener_position: [-5.0, 1.7, 15.0], source_id: 9 },
    ];
    b.reset_ray_counter();
    b.query_spatial(&qs, &Opaque);
    let f = cap.take_frame();
    assert!(f.rays.is_empty(), "probe rays are opt-in");
    assert_eq!(f.directs.len(), 2);
    let mut d = f.directs.clone();
    d.sort_by_key(|d| d.query_index);
    assert_eq!((d[0].source_id, d[1].source_id), (7, 9));
    assert!(!d[0].listener_outside && !d[0].reflections_skipped);
    assert!(d[1].listener_outside && d[1].reflections_skipped && d[1].occluded);
    assert!(d[1].occlusion_factor < 0.01);
    assert!(f.paths.iter().all(|p| p.source_id == 7 && p.query_index == 0), "valid paths only inside");
    assert!(f.paths.iter().any(|p| p.selected));
    assert!(f.rejected.is_empty());
    // ray_count is the total traced by the solver (all of it filtered), not the stored rays.
    assert!(f.ray_count > 1000 && f.rays_filtered == f.ray_count && f.rays_dropped == 0);
    assert!(f.ray_count <= b.rays_traced());
    assert_eq!(d[0].rays_traced as u64 + d[1].rays_traced as u64, f.ray_count);
}

#[test]
fn detailed_capture_tags_kinds_and_respects_the_caps() {
    let b = backend(true);
    let cap = b.debug_capture();
    cap.set_enabled(true);
    cap.set_detail(CaptureDetail { occlusion_probes: true, ..CaptureDetail::NONE });
    b.query_spatial(&[SpatialQuery { source_position: SRC, listener_position: INSIDE, source_id: 1 }], &Opaque);
    let f = cap.take_frame();
    assert!(!f.rays.is_empty() && f.rays.iter().all(|r| r.kind == DebugRayKind::OcclusionProbe));
    assert!(f.rays.len() <= 13 * 8);
    assert!(f.rays_filtered > 0);

    cap.set_detail(CaptureDetail::ALL);
    let qs: Vec<SpatialQuery> = (0..8)
        .map(|i| SpatialQuery { source_position: [3.0 + 2.0 * i as f32, 1.5, 4.0 + 3.0 * i as f32], listener_position: INSIDE, source_id: i })
        .collect();
    b.query_spatial(&qs, &Opaque);
    let f = cap.take_frame();
    assert!(f.rays.len() <= MAX_STORED_RAYS);
    assert_eq!(f.rays.len() as u64 + f.rays_dropped + f.rays_filtered, f.ray_count);
    assert!(f.rays_dropped > 0, "8 emitters with full detail exceed the cap");
    for k in [DebugRayKind::OcclusionProbe, DebugRayKind::ReflectionValidation] {
        assert!(f.rays.iter().any(|r| r.kind == k), "{k:?}");
    }
    assert!(!f.rejected.is_empty() && f.rejected.len() <= MAX_STORED_REJECTED);
    let reasons: Vec<RejectReason> = f.rejected.iter().map(|r| r.reason).collect();
    assert!(reasons.contains(&RejectReason::OutsideSurface), "{reasons:?}");
    assert!(f.rays.iter().all(|r| r.source_id < 8));
}

#[test]
fn diffraction_probes_are_captured_only_on_request() {
    // A free-standing panel between listener and source: the detour search runs.
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(1, vec![[5.0, -3.0, -3.0], [5.0, -3.0, 3.0], [5.0, 3.0, 3.0], [5.0, 3.0, -3.0]], vec![0, 1, 2, 0, 2, 3], 0));
    let b = CpuSimdComputeBackend::new(s, CpuSimdConfig { max_reflection_order: 0, ..CpuSimdConfig::default() });
    let cap = b.debug_capture();
    cap.set_enabled(true);
    let q = SpatialQuery { source_position: [10.0, 0.0, 0.0], listener_position: [0.0, 0.0, 0.0], source_id: 0 };
    b.query_spatial(&[q.clone()], &Opaque);
    let off = cap.take_frame();
    assert!(off.rays.is_empty() && off.ray_count > 13);
    cap.set_detail(CaptureDetail { diffraction_probes: true, ..CaptureDetail::NONE });
    b.query_spatial(&[q], &Opaque);
    let on = cap.take_frame();
    assert!(on.rays.len() > 10 && on.rays.iter().all(|r| r.kind == DebugRayKind::DiffractionProbe));
}

#[test]
fn leaky_walls_skip_only_the_exact_detour_search_unless_a_cutoff_is_set() {
    let l = [-5.0, 1.7, 15.0];
    let slow = backend(false);
    let fast = backend(true);
    let approx = CpuSimdComputeBackend::new(
        hall(),
        CpuSimdConfig { separated_reflection_max_transmission: 0.1, ..CpuSimdConfig::default() },
    );
    let (b, nb) = run(&slow, &Leaky, SRC, l);
    let (a, na) = run(&fast, &Leaky, SRC, l);
    assert_same(&a, &b, "leaky, exact shortcuts");
    assert!(na <= nb);
    assert!(!b.early_reflections.is_empty(), "leaky walls keep (very quiet) reflections in the exact search");
    let (c, nc) = run(&approx, &Leaky, SRC, l);
    assert!(c.early_reflections.is_empty());
    assert_eq!(c.direct_path.attenuation.0, b.direct_path.attenuation.0, "direct leakage unchanged");
    assert_eq!(c.late_reverb.late_loudness_db, b.late_reverb.late_loudness_db);
    assert!(nc <= OUTSIDE_RAY_BOUND, "{nc} rays");
    println!("leaky walls, listener outside: exhaustive {nb} rays, exact shortcuts {na}, with reflection cutoff {nc}");
    // The loudest reflection stays far below the direct leakage it replaces.
    let loudest = b.early_reflections.iter().map(|r| r.gain.0.iter().cloned().fold(0.0, f32::max)).fold(0.0, f32::max);
    println!("loudest dropped reflection gain {loudest:.4}");
    assert!(loudest <= 0.1, "{loudest}");
}
