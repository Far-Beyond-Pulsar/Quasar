//! Statistical late-field estimate (#65): volume, random-incidence absorption,
//! Eyring / Sabine T60, diffuse level, source/listener dependence.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::air::air_absorption_db_per_m;
use quasar_core::backend::{IAcousticComputeBackend, LateReverbEstimate, MaterialProvider, SpatialQuery};
use quasar_core::bands::{Band8, FREQ_BAND_CENTRES};
use quasar_core::rays::RayInteractionContext;
use quasar_core::scene::{AcousticMesh, AcousticScene};

struct Abs(f32);
impl MaterialProvider for Abs {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(self.0)
    }
}

/// alpha(theta) = a0 cos(theta): normal incidence a0, grazing 0.
struct CosAbs(f32);
impl MaterialProvider for CosAbs {
    fn evaluate_material(&self, _h: u32, c: &RayInteractionContext) -> Band8 {
        Band8::splat(self.0 * c.incident_angle_rad.cos())
    }
}

/// Closed box `[min, max]` as 6 quads (4 own vertices each), wound so the normals point
/// outward (`outward`) or inward.
fn closed_box(scene: &mut AcousticScene, id0: u64, min: [f32; 3], max: [f32; 3], outward: bool) {
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
        scene.add_mesh(AcousticMesh::new(id0 + i as u64, quad.to_vec(), idx, 0));
    }
}

/// Box room built from open quads (arbitrary winding): not a closed oriented surface.
fn quad_box(l: [f32; 3]) -> AcousticScene {
    let mut s = AcousticScene::new();
    for (i, (axis, v)) in [(0usize, 0.0), (0, l[0]), (1, 0.0), (1, l[1]), (2, 0.0), (2, l[2])].iter().enumerate() {
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let mk = |a: f32, b: f32| {
            let mut p = [0.0; 3];
            p[*axis] = *v;
            p[u] = a;
            p[w] = b;
            p
        };
        s.add_mesh(AcousticMesh::new(i as u64 + 1, vec![mk(0.0, 0.0), mk(l[u], 0.0), mk(l[u], l[w]), mk(0.0, l[w])], vec![0, 1, 2, 0, 2, 3], 0));
    }
    s
}

fn estimate(scene: AcousticScene, m: &dyn MaterialProvider, src: [f32; 3], lis: [f32; 3]) -> LateReverbEstimate {
    let b = CpuSimdComputeBackend::new(scene, CpuSimdConfig::default());
    b.query_spatial(&[SpatialQuery { source_position: src, listener_position: lis, source_id: 0 }], m)[0].late_reverb.clone()
}

/// Analytic Eyring T60 with air absorption for band `b`.
fn eyring(v: f32, s: f32, a: f32, b: usize) -> f32 {
    let m = air_absorption_db_per_m(FREQ_BAND_CENTRES[b], 20.0, 50.0) / 4.343;
    0.161 * v / (-s * (1.0 - a).ln() + 4.0 * m * v)
}

#[test]
fn t60_matches_eyring_and_sabine_for_a_box_room() {
    let l = [8.0, 3.0, 6.0];
    let (v, s) = (l[0] * l[1] * l[2], 2.0 * (l[0] * l[1] + l[0] * l[2] + l[1] * l[2]));
    for a in [0.05_f32, 0.2, 0.5] {
        let e = estimate(quad_box(l), &Abs(a), [2.0, 1.2, 1.5], [5.5, 1.6, 4.0]);
        for b in 0..8 {
            let want = eyring(v, s, a, b);
            assert!((e.t60.0[b] / want - 1.0).abs() < 0.05, "a {a} band {b}: {} vs Eyring {want}", e.t60.0[b]);
        }
        // Low absorption, low bands (air negligible): Sabine 0.161 V / (S a) agrees with Eyring.
        if a <= 0.05 {
            let sabine = 0.161 * v / (s * a);
            assert!((e.t60.0[1] / sabine - 1.0).abs() < 0.05, "Sabine {sabine} vs {}", e.t60.0[1]);
        }
    }
    // Eyring differs from Sabine at high absorption (Sabine over-predicts T60 there).
    let a = 0.5;
    let e = estimate(quad_box(l), &Abs(a), [2.0, 1.2, 1.5], [5.5, 1.6, 4.0]);
    let sabine = 0.161 * v / (s * a);
    assert!(e.t60.0[1] < 0.75 * sabine, "must follow Eyring, not Sabine: {} vs {sabine}", e.t60.0[1]);
}

#[test]
fn volume_comes_from_the_mesh_when_closed_and_the_box_volume_otherwise() {
    // Closed inward-wound shell 10 x 10 x 10 with a closed outward-wound 2 x 2 x 2 solid
    // inside: V = 1000 - 8. T60 follows that volume and the TOTAL area (shell + solid).
    let mut s = AcousticScene::new();
    closed_box(&mut s, 1, [0.0; 3], [10.0; 3], false);
    closed_box(&mut s, 10, [4.0; 3], [6.0; 3], true);
    let e = estimate(s, &Abs(0.2), [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    let (v, area) = (1000.0 - 8.0, 600.0 + 24.0);
    let want = eyring(v, area, 0.2, 2);
    assert!((e.t60.0[2] / want - 1.0).abs() < 0.02, "{} vs {want}", e.t60.0[2]);
    // Open (arbitrarily wound) quads: bounding-box volume (exact for a box).
    let e = estimate(quad_box([10.0; 3]), &Abs(0.2), [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    let want = eyring(1000.0, 600.0, 0.2, 2);
    assert!((e.t60.0[2] / want - 1.0).abs() < 0.02);
}

#[test]
fn random_incidence_absorption_integrates_the_angle_dependent_model() {
    // alpha = a0 cos(theta): 2 int a0 cos^2 sin dtheta = 2 a0 / 3.
    let l = [8.0, 3.0, 6.0];
    let (v, s) = (l[0] * l[1] * l[2], 2.0 * (l[0] * l[1] + l[0] * l[2] + l[1] * l[2]));
    let a0 = 0.6;
    let e = estimate(quad_box(l), &CosAbs(a0), [2.0, 1.2, 1.5], [5.5, 1.6, 4.0]);
    let want = eyring(v, s, 2.0 * a0 / 3.0, 2);
    assert!((e.t60.0[2] / want - 1.0).abs() < 0.03, "{} vs {want}", e.t60.0[2]);
    // Constant alpha integrates to itself.
    let c = estimate(quad_box(l), &Abs(0.3), [2.0, 1.2, 1.5], [5.5, 1.6, 4.0]);
    assert!((c.t60.0[2] / eyring(v, s, 0.3, 2) - 1.0).abs() < 0.01);
}

#[test]
fn level_is_the_diffuse_field_ratio_and_depends_on_source_and_listener_only_weakly() {
    let l = [8.0, 3.0, 6.0];
    let s = 2.0 * (l[0] * l[1] + l[0] * l[2] + l[1] * l[2]);
    let a = 0.2_f32;
    let r_const = s * a / (1.0 - a);
    let want_db = 10.0 * (16.0 * std::f32::consts::PI / r_const).log10();
    // Coincident-ish source and listener: the Barron factor is ~1.
    let near = estimate(quad_box(l), &Abs(a), [4.0, 1.5, 3.0], [4.1, 1.5, 3.0]);
    assert!((near.late_loudness_db - want_db).abs() < 0.3, "{} vs {want_db}", near.late_loudness_db);
    // Farther apart: the late energy after the direct sound is lower by 60 r / (c T) dB.
    let far = estimate(quad_box(l), &Abs(a), [1.0, 1.5, 1.0], [7.0, 1.5, 5.0]);
    let r = ((6.0_f32).powi(2) + (4.0_f32).powi(2)).sqrt();
    let t = far.t60.mean();
    let want_far = want_db - 60.0 * r / (343.0 * t);
    assert!((far.late_loudness_db - want_far).abs() < 0.3, "{} vs {want_far}", far.late_loudness_db);
    assert!(far.late_loudness_db < near.late_loudness_db);
    // Listener outside the room: much quieter.
    let out = estimate(quad_box(l), &Abs(a), [4.0, 1.5, 3.0], [30.0, 1.5, 3.0]);
    assert!(out.late_loudness_db < near.late_loudness_db - 15.0);
    // More absorption = quieter late field and shorter T60.
    let dead = estimate(quad_box(l), &Abs(0.6), [4.0, 1.5, 3.0], [4.1, 1.5, 3.0]);
    assert!(dead.late_loudness_db < near.late_loudness_db - 3.0);
    assert!(dead.t60.0[2] < near.t60.0[2]);
    // Mixing-time split scales with the room.
    assert!(near.early_late_split_secs >= 0.02 && near.early_late_split_secs <= 0.15);
}
