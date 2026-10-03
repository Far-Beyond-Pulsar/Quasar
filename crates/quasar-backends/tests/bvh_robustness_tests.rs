//! Ray / AABB / triangle robustness of the CPU tracer (#54).
//!
//! Property tests with a deterministic pseudo-random generator (no RNG crate, no
//! seeds from the clock): rays never escape a closed mesh, rays through shared
//! edges / vertices of a tessellated plane always hit, zero direction components
//! are safe, and the front-to-back BVH agrees with a brute-force reference.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::IAcousticComputeBackend;
use quasar_core::rays::Ray;
use quasar_core::scene::{AcousticMesh, AcousticScene};

/// xorshift64* in [0, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn unit_vec(&mut self) -> [f32; 3] {
        loop {
            let v = [self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0)];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0.1 && l <= 1.0 {
                return [v[0] / l, v[1] / l, v[2] / l];
            }
        }
    }
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn backend(scene: AcousticScene) -> CpuSimdComputeBackend {
    CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())
}

/// Closed cube `[-h, h]^3`, 12 triangles, shared edges along every face diagonal.
fn cube(h: f32) -> AcousticScene {
    let p = vec![
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ];
    let idx = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    ];
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(1, p, idx, 0));
    s
}

/// `n x n` cell plane at `y = 0` over `[0, n]^2`, two triangles per cell.
fn tessellated_plane(n: usize) -> AcousticScene {
    let mut p = Vec::new();
    for z in 0..=n {
        for x in 0..=n {
            p.push([x as f32, 0.0, z as f32]);
        }
    }
    let mut idx = Vec::new();
    let w = (n + 1) as u32;
    for z in 0..n as u32 {
        for x in 0..n as u32 {
            let a = z * w + x;
            idx.extend_from_slice(&[a, a + w, a + 1, a + 1, a + w, a + w + 1]);
        }
    }
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(2, p, idx, 0));
    s
}

fn hits(b: &CpuSimdComputeBackend, origin: [f32; 3], dir: [f32; 3]) -> bool {
    !b.trace_ray(&Ray::new(origin, dir)).is_empty()
}

#[test]
fn rays_from_inside_a_closed_box_never_escape() {
    let b = backend(cube(1.0));
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for i in 0..60_000 {
        let o = [rng.range(-0.95, 0.95), rng.range(-0.95, 0.95), rng.range(-0.95, 0.95)];
        let d = rng.unit_vec();
        assert!(hits(&b, o, d), "ray {i} escaped: origin {o:?} dir {d:?}");
    }
}

#[test]
fn rays_through_box_edges_vertices_and_diagonals_hit() {
    let b = backend(cube(1.0));
    let mut dirs: Vec<[f32; 3]> = Vec::new();
    // 8 vertices, 12 edge midpoints, 6 face centres (these cross the shared face diagonals).
    for sx in [-1.0_f32, 0.0, 1.0] {
        for sy in [-1.0_f32, 0.0, 1.0] {
            for sz in [-1.0_f32, 0.0, 1.0] {
                if sx != 0.0 || sy != 0.0 || sz != 0.0 {
                    dirs.push(norm([sx, sy, sz]));
                }
            }
        }
    }
    // Points on the face diagonals themselves (the shared edge of the 2 triangles).
    for t in [0.1_f32, 0.25, 0.5, 0.75, 0.9] {
        for s in [-1.0_f32, 1.0] {
            dirs.push(norm([s, s * (2.0 * t - 1.0), 1.0]));
            dirs.push(norm([1.0, s, s * (2.0 * t - 1.0)]));
            dirs.push(norm([s * (2.0 * t - 1.0), 1.0, s]));
        }
    }
    for d in dirs {
        assert!(hits(&b, [0.0, 0.0, 0.0], d), "centre ray {d:?} escaped");
        // Also from off-centre origins, towards the same targets.
        for o in [[0.3, -0.2, 0.1], [-0.6, 0.5, -0.4]] {
            assert!(hits(&b, o, d), "ray from {o:?} dir {d:?} escaped");
        }
    }
}

#[test]
fn rays_through_every_vertex_and_edge_of_a_tessellated_plane_hit() {
    let n = 12;
    let b = backend(tessellated_plane(n));
    let mut rng = Rng(12345);
    let mut checked = 0;
    // Straight down through every grid vertex, edge midpoint (horizontal, vertical, diagonal)
    // and cell centre, from several heights.
    for z in 0..=n {
        for x in 0..=n {
            let (xf, zf) = (x as f32, z as f32);
            let mut pts = vec![(xf, zf)];
            if x < n { pts.push((xf + 0.5, zf)); }
            if z < n { pts.push((xf, zf + 0.5)); }
            if x < n && z < n { pts.push((xf + 0.5, zf + 0.5)); pts.push((xf + 0.25, zf + 0.75)); }
            for (px, pz) in pts {
                for h in [0.5_f32, 3.0, 40.0] {
                    assert!(hits(&b, [px, h, pz], [0.0, -1.0, 0.0]), "plane miss at ({px}, {pz}) from height {h}");
                    checked += 1;
                }
                // Slanted rays aimed at the same point (and from below).
                for _ in 0..4 {
                    let tilt = [rng.range(-0.8, 0.8), 1.0, rng.range(-0.8, 0.8)];
                    let hh = rng.range(0.5, 5.0);
                    let origin = [px + tilt[0] * hh, hh, pz + tilt[2] * hh];
                    let d = norm([-tilt[0], -1.0, -tilt[2]]);
                    // Interior points only: slanted rays can legitimately leave through the rim.
                    if px > 0.5 && px < n as f32 - 0.5 && pz > 0.5 && pz < n as f32 - 0.5 {
                        assert!(hits(&b, origin, d), "slanted ray at ({px}, {pz}) dir {d:?} missed");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000);
}

#[test]
fn zero_direction_components_and_origins_on_slab_planes_are_safe() {
    let b = backend(cube(1.0));
    // Axis-aligned rays; the zero components make the old `1/dir` slab test produce NaN
    // when the origin lies exactly on a slab plane.
    let origins_on_planes = [
        [-5.0, 1.0, 0.0], [-5.0, -1.0, 0.0], [-5.0, 0.0, 1.0], [-5.0, 0.0, -1.0],
        [-5.0, 1.0, 1.0], [-5.0, -1.0, -1.0], [-5.0, 0.0, 0.0], [-5.0, 0.5, 0.5],
    ];
    for o in origins_on_planes {
        assert!(hits(&b, o, [1.0, 0.0, 0.0]), "axis ray from {o:?} must hit the cube");
    }
    for o in [[0.0, -5.0, 0.3], [0.2, -5.0, 1.0], [1.0, -5.0, 1.0]] {
        assert!(hits(&b, o, [0.0, 1.0, 0.0]), "ray from {o:?} must hit");
    }
    // Clean misses stay misses (outside every slab) and never panic or return NaN hits.
    assert!(!hits(&b, [-5.0, 3.0, 0.0], [1.0, 0.0, 0.0]));
    assert!(!hits(&b, [-5.0, 0.0, -3.0], [1.0, 0.0, 0.0]));
    let r = b.trace_ray(&Ray::new([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
    assert_eq!(r.len(), 1);
    assert!(r[0].distance.is_finite() && (r[0].distance - 4.0).abs() < 1e-3);
}

#[test]
fn glancing_hits_on_a_large_triangle_are_found() {
    // One huge triangle; a ray that meets it at a very shallow angle (the old
    // absolute `|det| < 1e-12` test discarded these).
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(
        3,
        vec![[-1000.0, 0.0, -1000.0], [1000.0, 0.0, -1000.0], [0.0, 0.0, 1000.0]],
        vec![0, 2, 1],
        0,
    ));
    let b = backend(s);
    let d = norm([1.0, -1.0e-4, 0.0]);
    assert!(hits(&b, [-500.0, 0.05, 10.0], d), "glancing hit lost");
}

/// Reference brute-force Moller-Trumbore (strict, no epsilon).
fn brute(tris: &[[[f32; 3]; 3]], o: [f32; 3], d: [f32; 3]) -> Option<f32> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut best: Option<f32> = None;
    for t in tris {
        let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
        let h = cross(d, e2);
        let det = dot(e1, h);
        if det.abs() < 1e-9 {
            continue;
        }
        let inv = 1.0 / det;
        let s = sub(o, t[0]);
        let u = dot(s, h) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = cross(s, e1);
        let v = dot(d, q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let tt = dot(e2, q) * inv;
        if tt > 0.0 && best.map_or(true, |b| tt < b) {
            best = Some(tt);
        }
    }
    best
}

#[test]
fn front_to_back_bvh_agrees_with_brute_force() {
    let mut rng = Rng(777);
    let mut tris: Vec<[[f32; 3]; 3]> = Vec::new();
    for _ in 0..400 {
        let c = [rng.range(-20.0, 20.0), rng.range(-20.0, 20.0), rng.range(-20.0, 20.0)];
        let mut t = [[0.0; 3]; 3];
        for v in t.iter_mut() {
            *v = [c[0] + rng.range(-2.0, 2.0), c[1] + rng.range(-2.0, 2.0), c[2] + rng.range(-2.0, 2.0)];
        }
        tris.push(t);
    }
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for (i, t) in tris.iter().enumerate() {
        positions.extend_from_slice(t);
        let k = (i * 3) as u32;
        indices.extend_from_slice(&[k, k + 1, k + 2]);
    }
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(4, positions, indices, 0));
    let b = backend(s);

    let mut agree = 0;
    let mut hit_count = 0;
    for _ in 0..4000 {
        let o = [rng.range(-25.0, 25.0), rng.range(-25.0, 25.0), rng.range(-25.0, 25.0)];
        let d = rng.unit_vec();
        let reference = brute(&tris, o, d);
        let got = b.trace_ray(&Ray::new(o, d)).first().map(|h| h.distance);
        match (reference, got) {
            (None, None) => agree += 1,
            (Some(a), Some(g)) => {
                assert!((a - g).abs() < 1e-3 * (1.0 + a), "closest hit {g} differs from reference {a}");
                agree += 1;
                hit_count += 1;
            }
            // Edge-tolerance can add a hit the strict reference misses (and vice versa for
            // near-parallel rays); such ties are rare, tolerate a handful.
            _ => {}
        }
    }
    assert!(agree >= 3980, "only {agree}/4000 rays agree with the reference");
    assert!(hit_count > 200, "test scene produced too few hits ({hit_count})");
}

#[test]
fn nearest_wall_wins_from_both_directions() {
    let mut s = AcousticScene::new();
    for (id, x) in [(10u64, 2.0_f32), (11, 5.0), (12, 9.0)] {
        s.add_mesh(AcousticMesh::new(
            id,
            vec![[x, -10.0, -10.0], [x, -10.0, 10.0], [x, 10.0, 10.0], [x, 10.0, -10.0]],
            vec![0, 1, 2, 0, 2, 3],
            0,
        ));
    }
    let b = backend(s);
    let h = b.trace_ray(&Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
    assert!((h[0].distance - 2.0).abs() < 1e-4);
    let h = b.trace_ray(&Ray::new([20.0, 0.0, 0.0], [-1.0, 0.0, 0.0]));
    assert!((h[0].distance - 11.0).abs() < 1e-4);
    // max_distance limits the search.
    let mut r = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    r.max_distance = 1.5;
    assert!(b.trace_ray(&r).is_empty());
}

#[test]
fn degenerate_triangles_are_ignored() {
    let mut s = AcousticScene::new();
    // A zero-area sliver (collinear points) plus a real triangle behind it.
    s.add_mesh(AcousticMesh::new(
        20,
        vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [-5.0, -5.0, 3.0], [5.0, -5.0, 3.0], [0.0, 5.0, 3.0]],
        vec![0, 1, 2, 3, 4, 5],
        0,
    ));
    let b = backend(s);
    let h = b.trace_ray(&Ray::new([0.5, 0.0, 0.0], [0.0, 0.0, 1.0]));
    assert_eq!(h.len(), 1);
    assert!((h[0].distance - 3.0).abs() < 1e-4, "hit the degenerate sliver at {}", h[0].distance);
    assert!((h[0].normal[1]).abs() < 1e-3, "normal must come from the real triangle: {:?}", h[0].normal);
}
