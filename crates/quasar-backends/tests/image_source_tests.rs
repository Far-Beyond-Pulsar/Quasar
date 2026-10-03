//! Image-source early-reflection tracer (#55, #56): analytic shoebox room,
//! blocked paths, single mirror, arrival direction, continuity, cost.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::air::air_absorption_gain;
use quasar_core::backend::{EarlyReflection, IAcousticComputeBackend, MaterialProvider, SpatialQuery, SPEED_OF_SOUND};
use quasar_core::bands::Band8;
use quasar_core::distance::DistanceModel;
use quasar_core::rays::RayInteractionContext;
use quasar_core::scene::{AcousticMesh, AcousticScene};

const FS: f32 = 48_000.0;

#[test]
fn debug_capture_records_actual_paths_and_preserves_solver_results() {
    let backend = CpuSimdComputeBackend::new(shoebox(8.0, 3.0, 6.0), CpuSimdConfig { max_reflections: 3, ..cfg(2) });
    let capture = backend.debug_capture();
    let baseline = reflections(&backend, &Abs(0.2), SRC, LIS);
    assert!(capture.take_frame().rays.is_empty());

    capture.set_enabled(true);
    let actual = reflections(&backend, &Abs(0.2), SRC, LIS);
    let frame = capture.take_frame();
    assert!(!frame.rays.is_empty());
    assert_eq!(baseline.len(), actual.len());
    for (a, b) in baseline.iter().zip(&actual) {
        assert_eq!(a.direction, b.direction);
        assert_eq!(a.delay_samples, b.delay_samples);
        assert_eq!(a.gain.0, b.gain.0);
    }
    let selected: Vec<_> = frame.paths.iter().filter(|p| p.selected).collect();
    assert_eq!(selected.len(), actual.len());
    assert_eq!(selected.len(), 3);
    assert!(frame.paths.iter().any(|p| !p.selected));
    for path in selected {
        assert_eq!(path.bounces.len(), path.reflection.order as usize);
        assert_eq!(path.normals.len(), path.bounces.len());
        assert_eq!(path.material_handles.len(), path.bounces.len());
        let mut total = 0.0;
        let mut from = SRC;
        for &bounce in &path.bounces {
            assert!((0..3).any(|i| bounce[i].abs() < 1e-4 || (bounce[i] - ROOM[i]).abs() < 1e-4));
            total += dist3(from, bounce);
            from = bounce;
        }
        total += dist3(from, LIS);
        assert!((total * FS / SPEED_OF_SOUND - path.reflection.delay_samples).abs() < 0.01);
        assert!(angle_deg(unit_from(LIS, from), path.reflection.direction) < 0.05);
    }
    for sample in &frame.rays {
        if let Some(hit) = &sample.hit {
            assert!(dist3(sample.ray.point_at(hit.distance), hit.point) < 1e-5);
        }
    }

    let moved = [4.0, 1.0, 3.0];
    capture.begin_update();
    reflections(&backend, &Abs(0.2), SRC, moved);
    assert!(capture.take_frame().paths.iter().all(|p| p.listener == moved));
    capture.set_enabled(false);
    reflections(&backend, &Abs(0.2), SRC, LIS);
    let disabled = capture.take_frame();
    assert!(disabled.rays.is_empty() && disabled.paths.is_empty());
}

#[test]
fn debug_capture_keeps_queries_from_parallel_pairs() {
    let backend = CpuSimdComputeBackend::new(shoebox(8.0, 3.0, 6.0), cfg(1));
    let capture = backend.debug_capture();
    capture.set_enabled(true);
    let other = [3.0, 1.0, 2.0];
    backend.query_spatial(&[query(SRC, LIS), query(other, LIS)], &Abs(0.2));
    let frame = capture.take_frame();
    assert!(frame.paths.iter().any(|p| p.source == SRC));
    assert!(frame.paths.iter().any(|p| p.source == other));
}

/// Constant absorption on every surface and band.
struct Abs(f32);
impl MaterialProvider for Abs {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(self.0)
    }
}

struct AbsTrans { absorption: f32, transmission: f32 }
impl MaterialProvider for AbsTrans {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(self.absorption)
    }
    fn evaluate_transmission(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(self.transmission)
    }
}

/// Absorption = incidence angle / (pi/2): 0 at normal incidence, 1 at grazing.
struct AngleAbs;
impl MaterialProvider for AngleAbs {
    fn evaluate_material(&self, _h: u32, c: &RayInteractionContext) -> Band8 {
        Band8::splat((c.incident_angle_rad / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0))
    }
}

/// Axis-aligned quad as two triangles.
fn quad(scene: &mut AcousticScene, id: u64, p: [[f32; 3]; 4]) {
    scene.add_mesh(AcousticMesh::new(id, p.to_vec(), vec![0, 1, 2, 0, 2, 3], 0));
}

/// Closed shoebox `[0,lx] x [0,ly] x [0,lz]`, six quads.
fn shoebox(lx: f32, ly: f32, lz: f32) -> AcousticScene {
    let mut s = AcousticScene::new();
    quad(&mut s, 1, [[0.0, 0.0, 0.0], [0.0, ly, 0.0], [0.0, ly, lz], [0.0, 0.0, lz]]); // x = 0
    quad(&mut s, 2, [[lx, 0.0, 0.0], [lx, ly, 0.0], [lx, ly, lz], [lx, 0.0, lz]]); // x = lx
    quad(&mut s, 3, [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, 0.0, lz], [0.0, 0.0, lz]]); // y = 0
    quad(&mut s, 4, [[0.0, ly, 0.0], [lx, ly, 0.0], [lx, ly, lz], [0.0, ly, lz]]); // y = ly
    quad(&mut s, 5, [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, ly, 0.0], [0.0, ly, 0.0]]); // z = 0
    quad(&mut s, 6, [[0.0, 0.0, lz], [lx, 0.0, lz], [lx, ly, lz], [0.0, ly, lz]]); // z = lz
    s
}

const ROOM: [f32; 3] = [8.0, 3.0, 6.0];
const SRC: [f32; 3] = [2.0, 1.2, 1.5];
const LIS: [f32; 3] = [5.5, 1.6, 4.0];

fn cfg(order: u32) -> CpuSimdConfig {
    CpuSimdConfig { max_reflection_order: order, max_reflections: 64, ..CpuSimdConfig::default() }
}

fn query(src: [f32; 3], lis: [f32; 3]) -> SpatialQuery {
    SpatialQuery { source_position: src, listener_position: lis, source_id: 0 }
}

fn reflections(b: &CpuSimdComputeBackend, m: &dyn MaterialProvider, src: [f32; 3], lis: [f32; 3]) -> Vec<EarlyReflection> {
    b.query_spatial(&[query(src, lis)], m)[0].early_reflections.clone()
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn unit_from(lis: [f32; 3], to: [f32; 3]) -> [f32; 3] {
    let d = [to[0] - lis[0], to[1] - lis[1], to[2] - lis[2]];
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / l, d[1] / l, d[2] / l]
}

fn angle_deg(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Coordinate of the `i`-th lattice image of `s` along an axis of length `l`.
fn lattice(i: i32, s: f32, l: f32) -> f32 {
    if i.rem_euclid(2) == 0 {
        i as f32 * l + s
    } else {
        (i + 1) as f32 * l - s
    }
}

/// Analytic image sources of the shoebox with `|i|+|j|+|k| == order` (1..=max).
fn shoebox_images(max_order: i32) -> Vec<([f32; 3], u32)> {
    let mut v = Vec::new();
    for i in -max_order..=max_order {
        for j in -max_order..=max_order {
            for k in -max_order..=max_order {
                let o = i.abs() + j.abs() + k.abs();
                if o >= 1 && o <= max_order {
                    v.push((
                        [lattice(i, SRC[0], ROOM[0]), lattice(j, SRC[1], ROOM[1]), lattice(k, SRC[2], ROOM[2])],
                        o as u32,
                    ));
                }
            }
        }
    }
    v
}

#[test]
fn shoebox_first_order_matches_analytic_image_sources() {
    let b = CpuSimdComputeBackend::new(shoebox(ROOM[0], ROOM[1], ROOM[2]), cfg(1));
    assert_eq!(b.reflection_plane_count(), 6, "12 triangles merge into 6 planes");
    let r = reflections(&b, &Abs(0.3), SRC, LIS);
    let mut images = shoebox_images(1);
    assert_eq!(r.len(), 6, "six walls, six first-order paths: {r:?}");
    assert_eq!(images.len(), 6);

    let dm = DistanceModel::default();
    for refl in &r {
        assert_eq!(refl.order, 1);
        // Match the analytic image by delay.
        let (idx, img) = images
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let da = (dist3(LIS, a.1 .0) * FS / SPEED_OF_SOUND - refl.delay_samples).abs();
                let db = (dist3(LIS, b.1 .0) * FS / SPEED_OF_SOUND - refl.delay_samples).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, v)| (i, v.0))
            .unwrap();
        let len = dist3(LIS, img);
        let want_delay = len * FS / SPEED_OF_SOUND;
        assert!((refl.delay_samples - want_delay).abs() <= 1.0, "delay {} vs {}", refl.delay_samples, want_delay);
        // #56: direction = unit vector from the listener toward the reflection point,
        // which lies on the line listener -> image source.
        let want_dir = unit_from(LIS, img);
        let ang = angle_deg(refl.direction, want_dir);
        assert!(ang <= 1.0, "direction off by {ang} deg: {:?} vs {:?}", refl.direction, want_dir);
        // Gain = sqrt(1 - alpha) x distance law x air, at the full path length.
        let air = air_absorption_gain(len, 20.0, 50.0);
        for band in 0..8 {
            let want = 0.7_f32.sqrt() * dm.gain(len) * air.0[band];
            assert!((refl.gain.0[band] - want).abs() <= 2e-3 * want, "band {band}: {} vs {want}", refl.gain.0[band]);
        }
        images.remove(idx);
    }
}

#[test]
fn shoebox_order_two_and_three_counts_and_delays() {
    // Lattice counts of a shoebox: 6, 18, 38 images at orders 1, 2, 3.
    for (max, total) in [(2, 6 + 18), (3, 6 + 18 + 38)] {
        let b = CpuSimdComputeBackend::new(shoebox(ROOM[0], ROOM[1], ROOM[2]), cfg(max));
        let r = reflections(&b, &Abs(0.2), SRC, LIS);
        let want = shoebox_images(max as i32);
        assert_eq!(want.len(), total);
        assert_eq!(r.len(), total, "order {max}: {} paths", r.len());

        let mut got: Vec<(f32, u32, [f32; 3])> = r.iter().map(|x| (x.delay_samples, x.order, x.direction)).collect();
        let mut exp: Vec<(f32, u32, [f32; 3])> = want
            .iter()
            .map(|(p, o)| (dist3(LIS, *p) * FS / SPEED_OF_SOUND, *o, unit_from(LIS, *p)))
            .collect();
        // Greedy one-to-one matching on (order, delay within 1 sample, direction within
        // 1 degree); distinct images can share a delay by symmetry, hence the direction.
        for g in &got {
            let pos = exp
                .iter()
                .position(|e| e.1 == g.1 && (g.0 - e.0).abs() <= 1.0 && angle_deg(g.2, e.2) <= 1.0);
            match pos {
                Some(p) => {
                    exp.remove(p);
                }
                None => panic!("order {} path at delay {} dir {:?} has no analytic image", g.1, g.0, g.2),
            }
        }
        assert!(exp.is_empty());
    }
}

#[test]
fn results_are_sorted_by_energy_capped_and_deterministic() {
    let scene = shoebox(ROOM[0], ROOM[1], ROOM[2]);
    let b = CpuSimdComputeBackend::new(scene.clone(), CpuSimdConfig { max_reflections: 5, ..cfg(3) });
    let r = reflections(&b, &Abs(0.2), SRC, LIS);
    assert_eq!(r.len(), 5);
    let energy = |x: &EarlyReflection| x.gain.0.iter().map(|g| g * g).sum::<f32>();
    for w in r.windows(2) {
        assert!(energy(&w[0]) >= energy(&w[1]), "must be strongest first");
    }
    // The cap keeps exactly the strongest of the uncapped set.
    let all = reflections(&CpuSimdComputeBackend::new(scene.clone(), cfg(3)), &Abs(0.2), SRC, LIS);
    for (a, c) in all.iter().take(5).zip(r.iter()) {
        assert_eq!(a.delay_samples, c.delay_samples);
    }
    // Deterministic.
    let again = reflections(&b, &Abs(0.2), SRC, LIS);
    assert_eq!(r.len(), again.len());
    for (a, c) in r.iter().zip(again.iter()) {
        assert_eq!(a.delay_samples, c.delay_samples);
        assert_eq!(a.direction, c.direction);
        assert_eq!(a.gain, c.gain);
    }
}

#[test]
fn order_zero_and_degenerate_inputs_give_nothing() {
    let b = CpuSimdComputeBackend::new(shoebox(ROOM[0], ROOM[1], ROOM[2]), cfg(0));
    assert!(reflections(&b, &Abs(0.2), SRC, LIS).is_empty());
    let b = CpuSimdComputeBackend::new(shoebox(ROOM[0], ROOM[1], ROOM[2]), cfg(2));
    assert!(reflections(&b, &Abs(0.2), [f32::NAN, 0.0, 0.0], LIS).is_empty());
    // Empty scene: no planes.
    let b = CpuSimdComputeBackend::new(AcousticScene::new(), cfg(2));
    assert!(reflections(&b, &Abs(0.2), SRC, LIS).is_empty());
    // A fully absorbing room reflects nothing audible.
    let b = CpuSimdComputeBackend::new(shoebox(ROOM[0], ROOM[1], ROOM[2]), cfg(2));
    assert!(reflections(&b, &Abs(1.0), SRC, LIS).is_empty());
}

#[test]
fn angle_dependent_absorption_is_evaluated_at_the_bounce_angle() {
    // Single wall at x = 0 (size 20 x 20), source and listener on the +x side.
    let mut s = AcousticScene::new();
    quad(&mut s, 1, [[0.0, -10.0, -10.0], [0.0, 10.0, -10.0], [0.0, 10.0, 10.0], [0.0, -10.0, 10.0]]);
    let b = CpuSimdComputeBackend::new(s, cfg(1));
    let (src, lis) = ([4.0, 0.0, -3.0], [2.0, 0.0, 3.0]);
    let r = reflections(&b, &AngleAbs, src, lis);
    assert_eq!(r.len(), 1);
    // Image source at (-4, 0, -3); the incidence angle at the wall is the angle
    // between the path and the wall normal (x axis).
    let img = [-4.0, 0.0, -3.0];
    let len = dist3(lis, img);
    let cos_i = (img[0] - lis[0]).abs() / len;
    let alpha = cos_i.acos() / std::f32::consts::FRAC_PI_2;
    let want = (1.0 - alpha).sqrt() * DistanceModel::default().gain(len) * air_absorption_gain(len, 20.0, 50.0).0[3];
    assert!((r[0].gain.0[3] - want).abs() <= 2e-3 * want, "{} vs {want}", r[0].gain.0[3]);
}

// ── single mirror (#56) ───────────────────────────────────────────────

fn mirror_wall() -> AcousticScene {
    // Wall at z = -5, 10 m wide (x in -5..5) and 6 m high (y in -3..3); normal +z.
    let mut s = AcousticScene::new();
    quad(&mut s, 1, [[-5.0, -3.0, -5.0], [5.0, -3.0, -5.0], [5.0, 3.0, -5.0], [-5.0, 3.0, -5.0]]);
    s
}

#[test]
fn single_mirror_direction_points_from_listener_to_the_mirror_point() {
    let b = CpuSimdComputeBackend::new(mirror_wall(), cfg(1));
    let (src, lis) = ([-2.0, 0.5, 0.0], [3.0, 0.5, 0.0]);
    let r = reflections(&b, &Abs(0.0), src, lis);
    assert_eq!(r.len(), 1);
    // Mirror point on z = -5 where the path src -> wall -> lis bounces:
    // by symmetry its x = (-2 + 3) / 2 = 0.5 (equal distances to the wall).
    let mp = [0.5, 0.5, -5.0];
    // Hand-derived: |src-mp| = |mp-lis| = sqrt(2.5^2 + 5^2).
    assert!((dist3(src, mp) - dist3(mp, lis)).abs() < 1e-5);
    let want = unit_from(lis, mp);
    let ang = angle_deg(r[0].direction, want);
    assert!(ang <= 1.0, "direction {:?} vs {:?} ({ang} deg)", r[0].direction, want);
    // NOT the old (source -> hit point) vector.
    let old = unit_from(src, mp);
    assert!(angle_deg(r[0].direction, old) > 5.0, "must be listener-relative");
    // Unit length, world space.
    let l = r[0].direction.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((l - 1.0).abs() < 1e-5);
    // Delay = (|src-mp| + |mp-lis|) fs / c.
    let want_delay = (dist3(src, mp) + dist3(mp, lis)) * FS / SPEED_OF_SOUND;
    assert!((r[0].delay_samples - want_delay).abs() <= 1.0);
}

#[test]
fn no_reflection_when_endpoints_are_on_opposite_sides_or_outside_the_surface() {
    let b = CpuSimdComputeBackend::new(mirror_wall(), cfg(1));
    // Source in front, listener behind the wall: no specular path.
    assert!(reflections(&b, &Abs(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, -8.0]).is_empty());
    // Bounce point would be at x = 40: beyond the wall's 10 m extent.
    assert!(reflections(&b, &Abs(0.0), [38.0, 0.5, 0.0], [42.0, 0.5, 0.0]).is_empty());
}

// ── blocked paths ─────────────────────────────────────────────────────

#[test]
fn blocked_paths_are_rejected_and_unblocked_ones_survive() {
    let room = shoebox(ROOM[0], ROOM[1], ROOM[2]);
    let free = reflections(&CpuSimdComputeBackend::new(room.clone(), cfg(1)), &Abs(0.2), SRC, LIS);
    assert_eq!(free.len(), 6);

    // The first-order path off the x = 0 wall: image (-2, 1.2, 1.5), bounce point where
    // the listener -> image line meets x = 0. Put a panel across the listener leg.
    let img = [-SRC[0], SRC[1], SRC[2]];
    let t = LIS[0] / (LIS[0] - img[0]);
    let bounce = [0.0, LIS[1] + (img[1] - LIS[1]) * t, LIS[2] + (img[2] - LIS[2]) * t];
    let mid = [LIS[0] + (bounce[0] - LIS[0]) * 0.05, LIS[1] + (bounce[1] - LIS[1]) * 0.05, LIS[2] + (bounce[2] - LIS[2]) * 0.05];
    let mut blocked = room.clone();
    // 12 cm x 12 cm panel perpendicular to x, 5 % along that leg from the listener (so it
    // shadows only a narrow cone: the other paths leave the listener at other angles).
    quad(
        &mut blocked,
        9,
        [[mid[0], mid[1] - 0.06, mid[2] - 0.06], [mid[0], mid[1] + 0.06, mid[2] - 0.06], [mid[0], mid[1] + 0.06, mid[2] + 0.06], [mid[0], mid[1] - 0.06, mid[2] + 0.06]],
    );
    let r = reflections(&CpuSimdComputeBackend::new(blocked, cfg(1)), &Abs(0.2), SRC, LIS);
    let want_delay = dist3(LIS, img) * FS / SPEED_OF_SOUND;
    assert!(
        !r.iter().any(|x| (x.delay_samples - want_delay).abs() < 2.0),
        "the path behind the panel must be rejected"
    );
    // The panel (a seventh plane) is itself a mirror, so the count is not simply 5,
    // but every other original wall path must still be there.
    for f in free.iter().filter(|f| (f.delay_samples - want_delay).abs() >= 2.0) {
        assert!(
            r.iter().any(|x| (x.delay_samples - f.delay_samples).abs() < 0.5),
            "unblocked path at delay {} dir {:?} lost (blocked path delay {want_delay})",
            f.delay_samples, f.direction
        );
    }
}

// ── coplanar surfaces ─────────────────────────────────────────────────

#[test]
fn coplanar_triangles_make_one_plane_and_one_reflection() {
    // A wall tessellated into 32 triangles (4 x 4 cells), plus a flipped-winding copy
    // of one cell: still a single mirror plane.
    let mut pos = Vec::new();
    let mut idx = Vec::new();
    for iy in 0..=4 {
        for ix in 0..=4 {
            pos.push([-5.0 + 2.5 * ix as f32, -5.0 + 2.5 * iy as f32, -5.0]);
        }
    }
    for iy in 0..4u32 {
        for ix in 0..4u32 {
            let a = iy * 5 + ix;
            idx.extend_from_slice(&[a, a + 1, a + 6, a, a + 6, a + 5]);
        }
    }
    idx.extend_from_slice(&[0, 6, 1]); // reversed winding
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(1, pos, idx, 0));
    let b = CpuSimdComputeBackend::new(s, cfg(1));
    assert_eq!(b.reflection_plane_count(), 1);
    let r = reflections(&b, &Abs(0.1), [-1.0, 0.0, 0.0], [1.5, 0.3, 0.0]);
    assert_eq!(r.len(), 1, "the shared interior edges are not borders: {r:?}");
    // Bounce at the tessellation's centre vertex region keeps full gain (edge window 1).
    let len = r[0].delay_samples * SPEED_OF_SOUND / FS;
    let want = 0.9_f32.sqrt() * DistanceModel::default().gain(len) * air_absorption_gain(len, 20.0, 50.0).0[4];
    assert!((r[0].gain.0[4] - want).abs() <= 2e-3 * want);
}

// ── continuity under listener motion ──────────────────────────────────

#[test]
fn path_fades_to_zero_at_the_surface_border_without_steps() {
    // Wall z = -5 spans x in -5..5. Source fixed; the listener walks in +x so the
    // bounce point (x = (sx + lx) / 2) crosses the border at x = 5.
    let b = CpuSimdComputeBackend::new(mirror_wall(), cfg(1));
    let src = [2.0, 0.0, 0.0];
    let gain_at = |lx: f32| -> f32 {
        let r = reflections(&b, &Abs(0.0), src, [lx, 0.0, 0.0]);
        r.first().map_or(0.0, |x| x.gain.0[3])
    };
    // Bounce x = (2 + lx)/2 -> 5 at lx = 8. Step the listener by 1 cm.
    let mut prev = gain_at(5.0);
    let mut max_step = 0.0_f32;
    let mut lx = 5.0;
    while lx < 8.2 {
        lx += 0.01;
        let g = gain_at(lx);
        max_step = max_step.max((g - prev).abs());
        prev = g;
    }
    // Beyond the border the path is gone; right before it the gain is ~0, and no
    // 1 cm step moves the gain by more than a few percent of the interior value.
    let interior = gain_at(5.0);
    assert!(interior > 0.05, "interior gain {interior}");
    assert!(gain_at(8.3) == 0.0, "no path past the border");
    assert!(gain_at(7.99) < 0.2 * interior, "faded near the border: {}", gain_at(7.99));
    assert!(max_step < 0.1 * interior, "largest 1 cm step {max_step} vs interior {interior}");
    // Monotone fade over the last 10 cm of bounce travel (20 cm of listener travel).
    let mut last = f32::INFINITY;
    let mut x = 7.8;
    while x <= 8.0 {
        let g = gain_at(x);
        assert!(g <= last + 1e-6, "gain must decrease toward the border: {g} > {last} at {x}");
        last = g;
        x += 0.01;
    }
}

#[test]
fn reflected_path_visibility_fades_across_a_blocker_edge() {
    let mut s = mirror_wall();
    // Panel at z = -2.5, x from 0.9 .. 3 (blocks the leg from the bounce at x=0.5.. to the listener).
    quad(&mut s, 2, [[0.9, -3.0, -2.5], [3.0, -3.0, -2.5], [3.0, 3.0, -2.5], [0.9, 3.0, -2.5]]);
    let b = CpuSimdComputeBackend::new(s, cfg(1));
    let src = [-2.0, 0.0, 0.0];
    let find_wall_path = |lis: [f32; 3]| {
        reflections(&b, &AbsTrans { absorption: 0.0, transmission: 0.0 }, src, lis)
            .into_iter()
            .filter(|r| r.direction[2] < -0.3 && r.order == 1)
            .map(|r| r.gain.0[3])
            .fold(0.0_f32, f32::max)
    };
    // Listener leg to bounce point (0.5 .. , -5): clear at lx = 0.6 (bounce x = -0.7), blocked at 3.
    let clear = find_wall_path([0.6, 0.0, 0.0]);
    assert!(clear > 0.05);
    let mut prev = clear;
    let mut max_step = 0.0_f32;
    let mut lx = 0.6;
    while lx < 3.0 {
        lx += 0.01;
        let gain = find_wall_path([lx, 0.0, 0.0]);
        max_step = max_step.max((gain - prev).abs());
        prev = gain;
    }
    // Nine-ray bundle bounds an isolated sample transition to 1/9 of the
    // unobstructed gain; allow three simultaneous transitions from geometry.
    assert!(max_step < clear * (3.0 / 9.0), "largest 1 cm step {max_step}, clear {clear}");
    assert!(prev < clear * 0.4, "well inside the blocker, gain {prev} vs clear {clear}");
}

#[test]
fn fully_transmissive_blocker_preserves_reflected_path() {
    let mut s = mirror_wall();
    quad(&mut s, 2, [[0.9, -3.0, -2.5], [3.0, -3.0, -2.5], [3.0, 3.0, -2.5], [0.9, 3.0, -2.5]]);
    let b = CpuSimdComputeBackend::new(s, cfg(1));
    let src = [-2.0, 0.0, 0.0];
    let lis = [3.0, 0.0, 0.0];
    let r = reflections(&b, &AbsTrans { absorption: 0.0, transmission: 1.0 }, src, lis);
    assert!(r.iter().any(|path| path.order == 1 && path.direction[2] < -0.3 && path.gain.0[3] > 0.05),
        "the blocker must not remove a fully transmissive path: {r:?}");
}

// ── cost ──────────────────────────────────────────────────────────────

#[test]
fn cost_report() {
    // Printed timing (run with `cargo test --release -p quasar-backends cost_report -- --nocapture`).
    let planes_scene = |n: usize| -> AcousticScene {
        // A shoebox plus `n` free-standing panels (each its own plane).
        let mut s = shoebox(ROOM[0], ROOM[1], ROOM[2]);
        for i in 0..n {
            let x = 0.5 + 7.0 * (i as f32 + 0.5) / n.max(1) as f32;
            let tilt = 0.15 * (i % 5) as f32;
            quad(
                &mut s,
                100 + i as u64,
                [[x, 0.2, 0.3], [x + tilt, 2.8, 0.3], [x + tilt, 2.8, 0.9], [x, 0.2, 0.9]],
            );
        }
        s
    };
    for (extra, order) in [(0usize, 3u32), (26, 2), (26, 3)] {
        let b = CpuSimdComputeBackend::new(planes_scene(extra), cfg(order));
        let q = [query(SRC, LIS)];
        let _ = b.query_spatial(&q, &Abs(0.2)); // warm-up
        let reps = 20;
        let t0 = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..reps {
            n = b.query_spatial(&q, &Abs(0.2))[0].early_reflections.len();
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "image-source cost: {} planes, order {order}: {:.3} ms/query, {n} paths kept",
            b.reflection_plane_count(),
            per * 1e3
        );
        if !cfg!(debug_assertions) {
            assert!(per < 0.05, "{per} s per query is too slow for a 15-30 Hz compute tick");
        }
    }
}
