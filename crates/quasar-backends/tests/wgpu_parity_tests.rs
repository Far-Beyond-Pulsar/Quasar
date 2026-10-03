#![cfg(feature = "wgpu-compute")]
//! GPU vs CPU parity (#78): the WGPU backend must produce the same direct-path
//! gain / occlusion, early reflections and late reverb as `CpuSimdComputeBackend`
//! on a reference scene (closed shoebox + one transmissive column).
//!
//! If the machine has no wgpu adapter every test prints `SKIPPED` and passes (it
//! cannot run); with an adapter they really execute on the device.
//!
//! Tolerances (the GPU runs the geometry in f32 with different operation order / FMA
//! contraction than the CPU, so the results agree to rounding, not bitwise):
//! * direct gain per band, occlusion per band: relative 1e-3 (+1e-6 absolute);
//! * distance, delay samples: relative 1e-6 (the host computes both identically);
//! * late reverb (T60, split, level): relative 1e-5 (same host code, same inputs);
//! * early reflections: same count; every GPU path has a distinct CPU path of equal order,
//!   delay relative 1e-4 (+1e-3 samples), direction
//!   component 1e-3, gain relative 2e-3 (+1e-6 absolute).

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_backends::wgpu_compute::{WgpuComputeBackend, WgpuComputeConfig};
use quasar_core::backend::{IAcousticComputeBackend, MaterialProvider, SpatialQuery, SpatialQueryResult};
use quasar_core::bands::Band8;
use quasar_core::distance::{DistanceCurve, DistanceModel};
use quasar_core::rays::{Ray, RayInteractionContext};
use quasar_core::scene::{AcousticMesh, AcousticScene};

/// Material 0 (walls): angle-dependent absorption, small transmission.
/// Material 1 (column): band-dependent transmission (low bands pass more).
struct Mats;
impl MaterialProvider for Mats {
    fn evaluate_material(&self, h: u32, c: &RayInteractionContext) -> Band8 {
        let base = if h == 0 { 0.12 } else { 0.35 };
        let mut v = [0.0_f32; 8];
        for (b, x) in v.iter_mut().enumerate() {
            *x = (base * (1.0 + 0.15 * b as f32) * (0.6 + 0.4 * c.incident_angle_rad.cos().abs())).min(0.95);
        }
        Band8::new(v)
    }
    fn evaluate_transmission(&self, h: u32, _c: &RayInteractionContext) -> Band8 {
        if h == 0 {
            Band8::splat(0.02)
        } else {
            let mut v = [0.0_f32; 8];
            for (b, x) in v.iter_mut().enumerate() {
                *x = 0.6 - 0.07 * b as f32; // 0.60 .. 0.11
            }
            Band8::new(v)
        }
    }
}

/// Closed box `[min, max]` as 6 quads with normals pointing outward or inward.
fn closed_box(scene: &mut AcousticScene, id0: u64, min: [f32; 3], max: [f32; 3], outward: bool, material: u32) {
    let c = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0, (min[2] + max[2]) / 2.0];
    let faces: [(usize, f32); 6] = [(0, min[0]), (0, max[0]), (1, min[1]), (1, max[1]), (2, min[2]), (2, max[2])];
    for (i, (axis, v)) in faces.iter().enumerate() {
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let mk = |a: f32, b: f32| {
            let mut p = [0.0; 3];
            p[*axis] = *v;
            p[u] = a;
            p[w] = b;
            p
        };
        let quad = [mk(min[u], min[w]), mk(max[u], min[w]), mk(max[u], max[w]), mk(min[u], max[w])];
        let e1 = [quad[1][0] - quad[0][0], quad[1][1] - quad[0][1], quad[1][2] - quad[0][2]];
        let e2 = [quad[2][0] - quad[0][0], quad[2][1] - quad[0][1], quad[2][2] - quad[0][2]];
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let mid = [quad[0][0] - c[0], quad[0][1] - c[1], quad[0][2] - c[2]];
        let facing_out = n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] > 0.0;
        let idx = if facing_out == outward { vec![0, 1, 2, 0, 2, 3] } else { vec![0, 2, 1, 0, 3, 2] };
        scene.add_mesh(AcousticMesh::new(id0 + i as u64, quad.to_vec(), idx, material));
    }
}

/// 8 x 3 x 5 m room (inward walls, material 0) with a floating 1 x 2 x 2 m column
/// (outward faces, material 1) at x 3.5..4.5, y 0.5..2.5, z 1.5..3.5.
fn reference_scene() -> AcousticScene {
    let mut s = AcousticScene::new();
    closed_box(&mut s, 1, [0.0, 0.0, 0.0], [8.0, 3.0, 5.0], false, 0);
    closed_box(&mut s, 100, [3.5, 0.5, 1.5], [4.5, 2.5, 3.5], true, 1);
    s
}

fn queries() -> Vec<SpatialQuery> {
    let q = |id: u32, s: [f32; 3], l: [f32; 3]| SpatialQuery { source_id: id, source_position: s, listener_position: l };
    vec![
        // Column straight between the two: both walls of the column are crossed.
        q(1, [7.0, 1.5, 2.5], [1.0, 1.5, 2.5]),
        // Clear line of sight (passes beside the column).
        q(2, [7.0, 2.0, 1.0], [1.0, 1.0, 1.0]),
        // Penumbra: the probe disc straddles the column's edge (soft fraction + diffraction).
        q(3, [7.0, 1.5, 4.5], [1.0, 1.5, 2.5]),
        // Source close to the listener, clear.
        q(4, [2.0, 1.2, 1.0], [1.0, 1.5, 1.0]),
    ]
}

/// Tiny local headless-device probe: `None` (and a printed SKIPPED) if no adapter.
fn gpu(scene: AcousticScene, config: WgpuComputeConfig) -> Option<WgpuComputeBackend> {
    match WgpuComputeBackend::request_headless_device() {
        Err(e) => {
            println!("SKIPPED: no usable wgpu adapter in this environment ({e}); the GPU parity test did not run");
            None
        }
        Ok((device, queue)) => {
            println!("wgpu adapter available: running on the device");
            Some(WgpuComputeBackend::new(device, queue, scene, config).expect("adapter exists but the backend failed to build"))
        }
    }
}

fn cpu(scene: AcousticScene) -> CpuSimdComputeBackend {
    CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())
}

fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= abs + rel * a.abs().max(b.abs())
}

fn assert_band(what: &str, a: &Band8, b: &Band8, rel: f32, abs: f32) {
    for i in 0..8 {
        assert!(close(a.0[i], b.0[i], rel, abs), "{what} band {i}: gpu {} vs cpu {}", a.0[i], b.0[i]);
    }
}

fn assert_parity(label: &str, g: &SpatialQueryResult, c: &SpatialQueryResult) {
    assert_eq!(g.source_id, c.source_id, "{label}: source id");
    let (gd, cd) = (&g.direct_path, &c.direct_path);
    assert!(close(gd.distance, cd.distance, 1e-6, 0.0), "{label}: distance");
    assert!(close(gd.delay_samples, cd.delay_samples, 1e-6, 0.0), "{label}: delay");
    assert_eq!(gd.occluded, cd.occluded, "{label}: occluded flag");
    assert!(close(gd.occlusion_factor, cd.occlusion_factor, 1e-3, 1e-6), "{label}: occlusion factor");
    assert_band(&format!("{label}: occlusion"), &gd.occlusion, &cd.occlusion, 1e-3, 1e-6);
    assert_band(&format!("{label}: direct gain"), &gd.attenuation, &cd.attenuation, 1e-3, 1e-6);

    assert_band(&format!("{label}: t60"), &g.late_reverb.t60, &c.late_reverb.t60, 1e-5, 0.0);
    assert!(close(g.late_reverb.early_late_split_secs, c.late_reverb.early_late_split_secs, 1e-5, 0.0), "{label}: split");
    assert!(close(g.late_reverb.late_loudness_db, c.late_reverb.late_loudness_db, 1e-5, 1e-5), "{label}: late level");

    assert_eq!(
        g.early_reflections.len(),
        c.early_reflections.len(),
        "{label}: early reflection count (gpu {} vs cpu {})",
        g.early_reflections.len(),
        c.early_reflections.len()
    );
    // Paths of (mathematically) equal energy, e.g. mirror-symmetric ones, are ranked by
    // f32 rounding noise, so the GPU and CPU lists may order them differently: match
    // every GPU path to a distinct CPU path of the same order, delay and direction.
    let mut used = vec![false; c.early_reflections.len()];
    for (i, ge) in g.early_reflections.iter().enumerate() {
        let found = c.early_reflections.iter().enumerate().find(|(j, ce)| {
            !used[*j]
                && ge.order == ce.order
                && close(ge.delay_samples, ce.delay_samples, 1e-4, 1e-3)
                && (0..3).all(|k| (ge.direction[k] - ce.direction[k]).abs() < 1e-3)
        });
        let (j, ce) = found.unwrap_or_else(|| panic!("{label}: gpu reflection {i} (order {}, delay {}, dir {:?}) has no CPU counterpart", ge.order, ge.delay_samples, ge.direction));
        used[j] = true;
        assert_band(&format!("{label}: reflection {i} gain"), &ge.gain, &ce.gain, 2e-3, 1e-6);
    }
}

#[test]
fn reference_scene_matches_cpu_backend() {
    let Some(g) = gpu(reference_scene(), WgpuComputeConfig::default()) else { return };
    let c = cpu(reference_scene());
    let qs = queries();
    let gr = g.try_query_spatial(&qs, &Mats).expect("gpu query");
    let cr = c.query_spatial(&qs, &Mats);
    assert_eq!(gr.len(), qs.len(), "one result per query");
    for (i, (a, b)) in gr.iter().zip(&cr).enumerate() {
        assert_parity(&format!("query {i}"), a, b);
    }

    // The scenario really exercises what it claims.
    let d = |i: usize| &gr[i].direct_path;
    assert!(d(0).occluded && d(0).occlusion_factor < 0.9, "column blocks the first pair");
    assert!(!d(1).occluded && d(1).occlusion_factor == 1.0, "second pair has a clear line");
    assert!(d(2).occluded && d(2).occlusion_factor > d(0).occlusion_factor, "third pair is a penumbra");
    assert!(!gr[0].early_reflections.is_empty() && gr[0].late_reverb.t60.0[0] > 0.0);
    println!(
        "parity ok: occlusion factors {:.4} {:.4} {:.4} {:.4}; reflections {:?}",
        d(0).occlusion_factor,
        d(1).occlusion_factor,
        d(2).occlusion_factor,
        d(3).occlusion_factor,
        gr.iter().map(|r| r.early_reflections.len()).collect::<Vec<_>>()
    );
}

#[test]
fn one_dispatch_covers_every_query_and_splits_large_batches() {
    // 3 queries per dispatch force several dispatches for 10 queries; results must
    // be complete and in query order (the stub used `queries[0]` only).
    let cfg = WgpuComputeConfig { max_sources_per_dispatch: 3, ..WgpuComputeConfig::default() };
    let Some(g) = gpu(reference_scene(), cfg) else { return };
    let c = cpu(reference_scene());
    let base = queries();
    let qs: Vec<SpatialQuery> = (0..10u32)
        .map(|i| {
            let mut q = base[(i as usize) % base.len()].clone();
            q.source_id = 1000 + i;
            q
        })
        .collect();
    let gr = g.try_query_spatial(&qs, &Mats).expect("gpu query");
    let cr = c.query_spatial(&qs, &Mats);
    assert_eq!(gr.len(), 10);
    for (i, (a, b)) in gr.iter().zip(&cr).enumerate() {
        assert_eq!(a.source_id, 1000 + i as u32);
        assert_parity(&format!("batch query {i}"), a, b);
    }
    assert!(g.try_query_spatial(&[], &Mats).expect("empty").is_empty());
}

#[test]
fn distance_model_sample_rate_and_scene_update_are_honoured() {
    let Some(mut g) = gpu(AcousticScene::new(), WgpuComputeConfig::default()) else { return };
    let mut c = cpu(AcousticScene::new());
    let model = DistanceModel { curve: DistanceCurve::Exponential, reference_distance: 2.0, rolloff_factor: 0.7, min_distance: 0.5, max_distance: 500.0 };
    g.set_distance_model(model);
    c.set_distance_model(model);
    g.set_sample_rate(44_100.0);
    c.set_sample_rate(44_100.0);
    let qs = queries();

    // Empty scene: free field, no reflections, anechoic late estimate.
    let (gr, cr) = (g.try_query_spatial(&qs, &Mats).unwrap(), c.query_spatial(&qs, &Mats));
    for (i, (a, b)) in gr.iter().zip(&cr).enumerate() {
        assert_parity(&format!("empty scene {i}"), a, b);
        assert!(a.early_reflections.is_empty() && !a.direct_path.occluded);
    }
    let expected_delay = gr[0].direct_path.distance * 44_100.0 / 343.0;
    assert!(close(gr[0].direct_path.delay_samples, expected_delay, 1e-6, 0.0), "sample rate hook");

    // Dynamic geometry: the room appears.
    g.update_scene(&reference_scene()).expect("update_scene");
    c.update_scene(&reference_scene()).expect("update_scene");
    let (gr, cr) = (g.try_query_spatial(&qs, &Mats).unwrap(), c.query_spatial(&qs, &Mats));
    for (i, (a, b)) in gr.iter().zip(&cr).enumerate() {
        assert_parity(&format!("after update_scene {i}"), a, b);
    }
    assert!(gr[0].direct_path.occluded && !gr[0].early_reflections.is_empty());
}

#[test]
fn trace_ray_hits_the_nearest_surface() {
    let Some(g) = gpu(reference_scene(), WgpuComputeConfig::default()) else { return };
    let ray = Ray { origin: [1.0, 1.5, 2.5], direction: [1.0, 0.0, 0.0], min_distance: 0.0, max_distance: 100.0 };
    let hits = g.trace_ray(&ray);
    assert_eq!(hits.len(), 1);
    assert!((hits[0].distance - 2.5).abs() < 1e-4, "column face at x = 3.5, got {}", hits[0].distance);
}

#[test]
fn uploaded_bvh_keeps_cpu_gpu_direct_path_parity_across_separated_leaf_clusters() {
    // Triangles surround (but do not cover) the tested direct ray. The ray enters
    // the overall BVH bounds and visits internal nodes before pruning to leaves.
    let mut scene = AcousticScene::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (cy, cz) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
        let base = vertices.len() as u32;
        vertices.extend([
            [0.0, cy - 0.4, cz - 0.4], [0.0, cy + 0.4, cz - 0.4],
            [0.0, cy + 0.4, cz + 0.4], [0.0, cy - 0.4, cz + 0.4],
        ]);
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    scene.add_mesh(AcousticMesh::new(900, vertices, indices, 0));
    let config = WgpuComputeConfig { max_reflection_order: 0, ..WgpuComputeConfig::default() };
    let Some(gpu) = gpu(scene.clone(), config) else { return };
    let cpu = CpuSimdComputeBackend::new(scene, CpuSimdConfig { max_reflection_order: 0, ..CpuSimdConfig::default() });
    let q = SpatialQuery { source_id: 91, source_position: [-1.0, 0.0, 0.0], listener_position: [1.0, 0.0, 0.0] };
    let gr = gpu.try_query_spatial(std::slice::from_ref(&q), &Mats).expect("GPU query");
    let cr = cpu.query_spatial(std::slice::from_ref(&q), &Mats);
    assert_eq!(gr[0].direct_path.occlusion, cr[0].direct_path.occlusion);
    for band in 0..8 {
        assert!((gr[0].direct_path.attenuation.0[band] - cr[0].direct_path.attenuation.0[band]).abs() < 1e-5);
    }
}

#[test]
fn full_reflection_lists_match_without_truncation() {
    // Rank cut-offs at 16 paths could hide disagreements among weak paths: compare the
    // complete lists (up to 64 paths, order 3) as well.
    let gcfg = WgpuComputeConfig { max_reflections: 64, max_candidates_per_query: 512, ..WgpuComputeConfig::default() };
    let Some(g) = gpu(reference_scene(), gcfg) else { return };
    let c = CpuSimdComputeBackend::new(reference_scene(), CpuSimdConfig { max_reflections: 64, ..CpuSimdConfig::default() });
    let qs = queries();
    let gr = g.try_query_spatial(&qs, &Mats).expect("gpu query");
    let cr = c.query_spatial(&qs, &Mats);
    for (i, (a, b)) in gr.iter().zip(&cr).enumerate() {
        assert_parity(&format!("full list query {i}"), a, b);
        assert!(b.early_reflections.len() > 16, "scene should yield more than 16 paths, got {}", b.early_reflections.len());
    }
    println!("full reflection lists: {:?}", cr.iter().map(|r| r.early_reflections.len()).collect::<Vec<_>>());
}
