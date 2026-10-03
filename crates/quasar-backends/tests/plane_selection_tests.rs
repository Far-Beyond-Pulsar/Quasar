//! #152: importance-based mirror-plane selection. A closed hall with a faceted barrel vault
//! (64 facets), exact duplicate faces and a buried back face: the vault contributes plane
//! groups, duplicates / buried faces take no slots, and the first-order ceiling reflection
//! arrives at the analytic delay (flat ceiling) or within the documented tolerance of the
//! best-fit plane (vault).

mod common;

use common::{box_mesh, Opaque};
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::scene::{AcousticMesh, AcousticScene};

const W: f32 = 20.0; // x
const L: f32 = 30.0; // z
const WALL_H: f32 = 8.0; // springing height of the vault
const FACETS: usize = 64;

/// Mesh builder that orients every triangle toward the room interior.
struct Soup {
    pos: Vec<[f32; 3]>,
    idx: Vec<u32>,
    inside: [f32; 3],
}

impl Soup {
    fn tri(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
        let n = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        let m = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
        let to = [self.inside[0] - m[0], self.inside[1] - m[1], self.inside[2] - m[2]];
        let (b, c) = if n[0] * to[0] + n[1] * to[1] + n[2] * to[2] >= 0.0 { (b, c) } else { (c, b) };
        let base = self.pos.len() as u32;
        self.pos.extend([a, b, c]);
        self.idx.extend([base, base + 1, base + 2]);
    }
    fn quad(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) {
        self.tri(a, b, c);
        self.tri(a, c, d);
    }
}

/// Closed hall `W x WALL_H(+vault) x L`. `flat`: flat ceiling at `WALL_H`; else a semicircular
/// barrel vault of radius `W / 2` made of `FACETS` planar facets. `extras` adds exact duplicates
/// of the end-wall triangles (two copies, one reversed).
fn hall_scene(flat: bool, extras: bool) -> AcousticScene {
    let mut s = Soup { pos: Vec::new(), idx: Vec::new(), inside: [W / 2.0, 3.0, L / 2.0] };
    // floor and side walls up to the springing line
    s.quad([0.0, 0.0, 0.0], [W, 0.0, 0.0], [W, 0.0, L], [0.0, 0.0, L]);
    s.quad([0.0, 0.0, 0.0], [0.0, WALL_H, 0.0], [0.0, WALL_H, L], [0.0, 0.0, L]);
    s.quad([W, 0.0, 0.0], [W, WALL_H, 0.0], [W, WALL_H, L], [W, 0.0, L]);
    let end_start = s.idx.len();
    for z in [0.0, L] {
        s.quad([0.0, 0.0, z], [W, 0.0, z], [W, WALL_H, z], [0.0, WALL_H, z]);
    }
    if flat {
        s.quad([0.0, WALL_H, 0.0], [W, WALL_H, 0.0], [W, WALL_H, L], [0.0, WALL_H, L]);
    } else {
        let r = W / 2.0;
        let arc = |i: usize| {
            let a = std::f32::consts::PI * i as f32 / FACETS as f32;
            [r - r * a.cos(), WALL_H + r * a.sin()]
        };
        for i in 0..FACETS {
            let (p, q) = (arc(i), arc(i + 1));
            s.quad([p[0], p[1], 0.0], [q[0], q[1], 0.0], [q[0], q[1], L], [p[0], p[1], L]);
            // end caps (fan to the springing centre)
            for z in [0.0, L] {
                s.tri([r, WALL_H, z], [p[0], p[1], z], [q[0], q[1], z]);
            }
        }
    }
    if extras {
        let ends: Vec<u32> = s.idx[end_start..end_start + 12].to_vec();
        for t in ends.chunks_exact(3) {
            let (a, b, c) = (s.pos[t[0] as usize], s.pos[t[1] as usize], s.pos[t[2] as usize]);
            let base = s.pos.len() as u32;
            s.pos.extend([a, b, c, a, b, c]);
            s.idx.extend([base, base + 1, base + 2, base + 3, base + 5, base + 4]); // equal and reversed
        }
    }
    let mut scene = AcousticScene::new();
    scene.add_mesh(AcousticMesh::new(1, s.pos, s.idx, 0));
    scene
}

const SRC: [f32; 3] = [10.0, 1.7, 5.0];
const LIS: [f32; 3] = [10.0, 1.7, 25.0];

fn first_order_delays(b: &CpuSimdComputeBackend) -> Vec<f32> {
    let q = SpatialQuery { source_position: SRC, listener_position: LIS, source_id: 0 };
    b.query_spatial(&[q], &Opaque)[0].early_reflections.iter().filter(|r| r.order == 1).map(|r| r.delay_samples).collect()
}

fn samples(d: f32) -> f32 {
    d * 48_000.0 / 343.0
}

fn is_vault(p: &quasar_backends::cpu_simd::ReflectionPlaneInfo) -> bool {
    p.centroid[1] > WALL_H + 0.5 && p.normal[1].abs() > 0.2
}

#[test]
fn flat_ceiling_first_order_arrives_at_the_analytic_delay() {
    let b = CpuSimdComputeBackend::new(hall_scene(true, false), CpuSimdConfig::default());
    // Ceiling at y = 8: image of the source at y = 2 * 8 - 1.7.
    let image = [SRC[0], 2.0 * WALL_H - SRC[1], SRC[2]];
    let d = ((image[0] - LIS[0]).powi(2) + (image[1] - LIS[1]).powi(2) + (image[2] - LIS[2]).powi(2)).sqrt();
    let delays = first_order_delays(&b);
    assert!(
        delays.iter().any(|&x| (x - samples(d)).abs() < 0.5),
        "ceiling reflection at {:.2} samples expected, first-order delays {delays:?}",
        samples(d)
    );
}

#[test]
fn vault_gets_plane_groups_and_merging_is_configurable() {
    let b = CpuSimdComputeBackend::new(hall_scene(false, false), CpuSimdConfig::default());
    let planes = b.reflection_planes();
    let vault: Vec<_> = planes.iter().filter(|p| is_vault(p)).collect();
    println!("default merge (5 deg / 0.25 m): {} planes, {} vault groups", planes.len(), vault.len());
    assert!(!vault.is_empty(), "the vault must contribute at least one plane group");
    assert!(vault.len() <= FACETS / 2, "a faceted vault must merge ({} groups for {FACETS} facets)", vault.len());
    assert!(vault.iter().any(|p| p.merged && p.triangles > 2));

    // Wider cone: a handful of groups for the whole barrel.
    let wide = CpuSimdConfig { plane_merge_angle_deg: 25.0, plane_merge_offset: 1.0, ..CpuSimdConfig::default() };
    let b = CpuSimdComputeBackend::new(hall_scene(false, false), wide);
    let wide_vault = b.reflection_planes().into_iter().filter(|p| is_vault(p)).count();
    println!("wide merge (25 deg / 1 m): {wide_vault} vault groups");
    assert!(wide_vault >= 1 && wide_vault <= 6, "{wide_vault}");

    // No merging: every facet pair is its own plane (the old behaviour).
    let none = CpuSimdConfig { plane_merge_angle_deg: 0.0, max_reflection_planes: 64, ..CpuSimdConfig::default() };
    let b = CpuSimdComputeBackend::new(hall_scene(false, false), none);
    assert!(b.reflection_planes().iter().filter(|p| is_vault(p)).count() >= FACETS / 2);
}

#[test]
fn vault_reflection_matches_the_best_fit_plane_within_the_documented_tolerance() {
    let b = CpuSimdComputeBackend::new(hall_scene(false, false), CpuSimdConfig::default());
    let delays = first_order_delays(&b);
    let mut matched = false;
    let mut report = String::new();
    for p in b.reflection_planes().iter().filter(|p| is_vault(p)) {
        // Image of the source across the group's best-fit plane.
        let sd = p.normal[0] * SRC[0] + p.normal[1] * SRC[1] + p.normal[2] * SRC[2] - p.offset;
        let image = [SRC[0] - 2.0 * sd * p.normal[0], SRC[1] - 2.0 * sd * p.normal[1], SRC[2] - 2.0 * sd * p.normal[2]];
        let d = ((image[0] - LIS[0]).powi(2) + (image[1] - LIS[1]).powi(2) + (image[2] - LIS[2]).powi(2)).sqrt();
        // Documented tolerance: the bounce point is snapped onto the member facet, at most
        // `spread` from the fit plane, which changes the path length by at most 2 * spread.
        let tol = samples(2.0 * p.spread) + 0.5;
        report += &format!("[n=({:.2},{:.2}) spread {:.3} m, expected {:.1} +- {:.1}] ", p.normal[0], p.normal[1], p.spread, samples(d), tol);
        if delays.iter().any(|&x| (x - samples(d)).abs() <= tol) {
            matched = true;
        }
    }
    assert!(matched, "no first-order arrival within tolerance of a vault plane: delays {delays:?} vs {report}");
}

#[test]
fn duplicate_and_buried_faces_take_no_slots() {
    let clean = CpuSimdComputeBackend::new(hall_scene(false, false), CpuSimdConfig::default());
    let dirty = CpuSimdComputeBackend::new(hall_scene(false, true), CpuSimdConfig::default());
    let (a, b): (f32, f32) = (clean.reflection_planes().iter().map(|p| p.area).sum(), dirty.reflection_planes().iter().map(|p| p.area).sum());
    assert!((a - b).abs() < 1e-2 * a, "duplicates must not add area: clean {a}, with duplicates {b}");
    assert_eq!(clean.reflection_planes().len(), dirty.reflection_planes().len());

    // A 3 cm panel 1 cm in front of the x = 0 wall: its back face (x = 0.01) has the wall
    // within 1 cm on one side and its own front face within 3 cm on the other: buried.
    let mut scene = hall_scene(false, false);
    scene.add_mesh(box_mesh(7, [0.01, 1.0, 5.0], [0.04, 5.0, 15.0], false));
    let with_panel = CpuSimdComputeBackend::new(scene, CpuSimdConfig::default());
    let has = |b: &CpuSimdComputeBackend, offset: f32| {
        b.reflection_planes().iter().any(|p| p.normal[0].abs() > 0.999 && (p.offset - offset).abs() < 5e-3)
    };
    assert!(has(&with_panel, 0.04), "the panel's room-facing face is a real reflector");
    assert!(!has(&with_panel, 0.01), "the buried back face must take no slot");

    let mut scene = hall_scene(false, false);
    scene.add_mesh(box_mesh(7, [0.01, 1.0, 5.0], [0.04, 5.0, 15.0], false));
    let no_probe = CpuSimdConfig { plane_buried_distance: 0.0, ..CpuSimdConfig::default() };
    assert!(has(&CpuSimdComputeBackend::new(scene, no_probe), 0.01), "without the probe the back face competes");
}

#[test]
fn selection_is_deterministic_and_independent_of_the_listener() {
    let a = CpuSimdComputeBackend::new(hall_scene(false, true), CpuSimdConfig::default());
    let b = CpuSimdComputeBackend::new(hall_scene(false, true), CpuSimdConfig::default());
    let (pa, pb) = (a.reflection_planes(), b.reflection_planes());
    assert_eq!(pa.len(), pb.len());
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!((x.normal, x.offset.to_bits(), x.triangles), (y.normal, y.offset.to_bits(), y.triangles));
    }
    // Queries never change the plane set.
    for l in [[10.0, 1.7, 25.0], [3.0, 5.0, 8.0], [-5.0, 1.0, 10.0]] {
        let q = SpatialQuery { source_position: SRC, listener_position: l, source_id: 0 };
        let _ = a.query_spatial(&[q], &Opaque);
        assert_eq!(a.reflection_planes().len(), pa.len());
    }
}

/// The merged vault must reflect into the hall from every bounce position: ceiling arrivals
/// exist for listeners across the width.
#[test]
fn vault_reflections_are_found_across_the_hall() {
    let b = CpuSimdComputeBackend::new(hall_scene(false, false), CpuSimdConfig::default());
    for x in [3.0_f32, 6.0, 10.0, 14.0, 17.0] {
        let q = SpatialQuery { source_position: [x, 1.7, 5.0], listener_position: [20.0 - x, 1.7, 25.0], source_id: 0 };
        let n = b.query_spatial(&[q], &Opaque)[0].early_reflections.iter().filter(|r| r.order == 1 && r.direction[1] > 0.3).count();
        assert!(n >= 1, "no first-order vault reflection for source x = {x}");
    }
}

// ── cost ──────────────────────────────────────────────────────────────

/// `cargo test -p quasar-backends --release --test plane_selection_tests -- --ignored --nocapture`
/// Scene build (BVH + plane selection) on ~400 k triangles: a hall with a faceted vault and a
/// lattice of small ribs / columns.
#[test]
#[ignore = "timing report; run in release"]
fn plane_build_cost_on_a_400k_triangle_scene() {
    use std::time::Instant;
    let build = |cfg: CpuSimdConfig| {
        let mut scene = hall_scene(false, false);
        let boxes = 400_000 / 12;
        for n in 0..boxes {
            let (ix, iz, iy) = (n % 60, (n / 60) % 100, n / 6000);
            let (x, z, y) = (0.5 + ix as f32 * 0.3, 0.5 + iz as f32 * 0.29, 0.2 + iy as f32 * 0.12);
            scene.add_mesh(box_mesh(100 + n as u64, [x, y, z], [x + 0.2, y + 0.1, z + 0.2], false));
        }
        let t = Instant::now();
        let b = CpuSimdComputeBackend::new(scene, cfg);
        (t.elapsed().as_secs_f64() * 1e3, b.reflection_plane_count())
    };
    let (legacy_ms, legacy_n) = build(CpuSimdConfig { plane_merge_angle_deg: 0.0, plane_buried_distance: 0.0, ..CpuSimdConfig::default() });
    let (new_ms, new_n) = build(CpuSimdConfig::default());
    println!("~400k triangles: BVH + legacy plane selection {legacy_ms:.0} ms ({legacy_n} planes); BVH + importance selection {new_ms:.0} ms ({new_n} planes)");
}
