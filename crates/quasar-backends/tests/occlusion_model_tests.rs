//! Direct-path occlusion model (#49, #50): per-band transmission through every
//! surface crossed, soft multi-ray visibility and edge diffraction.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::bands::Band8;
use quasar_core::scene::{AcousticMesh, AcousticScene};
use quasar_materials::instance::AcousticMaterialInstance;
use quasar_materials::registry::AcousticMaterialRegistry;
use quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};

fn registry_with(transmission: Band8) -> (AcousticMaterialRegistry, u32) {
    let registry = AcousticMaterialRegistry::new();
    registry.register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    // Absorption deliberately high and unrelated: it must NOT decide how much passes.
    let h = registry.add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.9), Band8::zeros(), transmission),
    ));
    (registry, h)
}

/// Large wall (normal along x) at `x`, much larger than any detour the model searches.
fn wall(scene: &mut AcousticScene, id: u64, x: f32, mat: u32) {
    scene.add_mesh(AcousticMesh::new(
        id,
        vec![[x, -60.0, -60.0], [x, -60.0, 60.0], [x, 60.0, 60.0], [x, 60.0, -60.0]],
        vec![0, 1, 2, 0, 2, 3],
        mat,
    ));
}

/// Box column `w` wide (x, z) and 20 m tall, centred on the origin.
fn column(scene: &mut AcousticScene, w: f32, mat: u32) {
    let h = w * 0.5;
    let p = vec![
        [-h, -10.0, -h], [h, -10.0, -h], [h, 10.0, -h], [-h, 10.0, -h],
        [-h, -10.0, h], [h, -10.0, h], [h, 10.0, h], [-h, 10.0, h],
    ];
    let idx = vec![
        0, 2, 1, 0, 3, 2, // z = -h
        4, 5, 6, 4, 6, 7, // z = +h
        0, 4, 7, 0, 7, 3, // x = -h
        1, 2, 6, 1, 6, 5, // x = +h
        0, 1, 5, 0, 5, 4, // y = -10
        3, 7, 6, 3, 6, 2, // y = +10
    ];
    scene.add_mesh(AcousticMesh::new(7, p, idx, mat));
}

fn query(src: [f32; 3], lis: [f32; 3]) -> SpatialQuery {
    SpatialQuery { source_position: src, listener_position: lis, source_id: 0 }
}

fn occlusion(scene: AcousticScene, reg: &AcousticMaterialRegistry, q: &SpatialQuery) -> quasar_core::backend::DirectPathResult {
    let b = CpuSimdComputeBackend::new(scene, CpuSimdConfig::default());
    b.query_spatial(&[q.clone()], reg)[0].direct_path.clone()
}

#[test]
fn clear_path_is_exactly_unoccluded() {
    let (reg, mat) = registry_with(Band8::zeros());
    let mut s = AcousticScene::new();
    wall(&mut s, 1, 3.0, mat); // behind the listener's side, not between
    let d = occlusion(s, &reg, &query([-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
    assert!(!d.occluded);
    assert_eq!(d.occlusion_factor, 1.0);
    assert!(d.occlusion.0.iter().all(|&o| o == 1.0));
}

#[test]
fn two_walls_attenuate_more_than_one() {
    let (reg, mat) = registry_with(Band8::splat(0.5));
    let q = query([10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

    let mut one = AcousticScene::new();
    wall(&mut one, 1, 4.0, mat);
    let mut two = AcousticScene::new();
    wall(&mut two, 1, 4.0, mat);
    wall(&mut two, 2, 6.0, mat);

    let d1 = occlusion(one, &reg, &q);
    let d2 = occlusion(two, &reg, &q);
    for b in 0..8 {
        assert!((d1.occlusion.0[b] - 0.5).abs() < 1e-3, "one wall band {b}: {}", d1.occlusion.0[b]);
        assert!((d2.occlusion.0[b] - 0.25).abs() < 1e-3, "two walls band {b}: {}", d2.occlusion.0[b]);
        assert!(d2.attenuation.0[b] < d1.attenuation.0[b]);
    }
    assert!(d1.occluded && d2.occluded);
}

#[test]
fn zero_transmission_blocks_every_band() {
    let (reg, mat) = registry_with(Band8::zeros());
    let mut s = AcousticScene::new();
    wall(&mut s, 1, 4.0, mat);
    let d = occlusion(s, &reg, &query([10.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
    for b in 0..8 {
        assert!(d.occlusion.0[b] <= 1.0e-3, "band {b} leaks: {}", d.occlusion.0[b]);
        assert!(d.occlusion.0[b].is_finite() && d.occlusion.0[b] > 0.0);
    }
}

#[test]
fn transmission_is_per_band_not_absorption() {
    let t = Band8::new([0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]);
    let (reg, mat) = registry_with(t);
    let mut s = AcousticScene::new();
    wall(&mut s, 1, 4.0, mat);
    let d = occlusion(s, &reg, &query([10.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
    for b in 0..8 {
        assert!((d.occlusion.0[b] - t.0[b]).abs() < 1e-3, "band {b}: {} vs {}", d.occlusion.0[b], t.0[b]);
    }
    // The old model read absorption (0.9 -> 0.1 through): a stone wall (low
    // absorption, no transmission) must not leak.
    let (reg2, mat2) = registry_with(Band8::zeros());
    let mut s2 = AcousticScene::new();
    wall(&mut s2, 1, 4.0, mat2);
    let d2 = occlusion(s2, &reg2, &query([10.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
    assert!(d2.occlusion_factor < 1.0e-3);
}

/// Source swept across the edge of an opaque column (listener and source 10 m
/// either side, column 0.65 m wide).
fn sweep(step: f32) -> Vec<(f32, Band8)> {
    let (reg, mat) = registry_with(Band8::zeros());
    let mut s = AcousticScene::new();
    column(&mut s, 0.65, mat);
    let backend = CpuSimdComputeBackend::new(s, CpuSimdConfig::default());
    let mut out = Vec::new();
    let mut x = -2.0_f32;
    while x <= 2.0 {
        let q = query([x, 0.0, 10.0], [0.0, 0.0, -10.0]);
        out.push((x, backend.query_spatial(&[q], &reg)[0].direct_path.occlusion));
        x += step;
    }
    out
}

fn db(v: f32) -> f32 {
    20.0 * v.max(1e-9).log10()
}

#[test]
fn sweeping_a_source_across_a_column_edge_is_continuous() {
    let curve = sweep(0.01);
    // Far outside the shadow: exactly clear; centred behind the column: deep shadow.
    assert!(curve.first().unwrap().1 .0.iter().all(|&o| o == 1.0));
    assert!(curve.last().unwrap().1 .0.iter().all(|&o| o == 1.0));
    let centre = curve.iter().min_by(|a, b| (a.0.abs()).partial_cmp(&b.0.abs()).unwrap()).unwrap();
    assert!(centre.1 .0[7] < 0.5 && centre.1 .0[0] < 0.9, "centre shadow {:?}", centre.1);

    // Documented bound: with 13 probe rays on a 0.35 m radius disc, a 1 cm tick
    // flips at most ~one ray, and one ray moves a band by at most |shadow dB| / 13
    // (<= ~2.3 dB for the 30 dB diffraction cap). Measured on this column sweep: 0.61 dB
    // per 1 cm tick, so the test bound is 1.0 dB.
    let mut worst = 0.0_f32;
    for w in curve.windows(2) {
        for b in 0..8 {
            worst = worst.max((db(w[1].1 .0[b]) - db(w[0].1 .0[b])).abs());
        }
    }
    assert!(worst <= 1.0, "max per-tick step {worst} dB");
    // A hard 1/0 model would jump by the whole shadow depth at once.
    let depth = curve.iter().map(|c| -db(c.1 .0[7])).fold(0.0_f32, f32::max);
    assert!(depth > 8.0 && worst < depth * 0.3, "shadow depth {depth} dB, step {worst} dB");
}

#[test]
fn high_bands_are_shadowed_more_than_low_bands() {
    let curve = sweep(0.05);
    let mut checked = 0;
    for (x, o) in &curve {
        // Fully inside the shadow (beyond the penumbra on the shadow side).
        if x.abs() < 0.1 {
            for b in 1..8 {
                assert!(o.0[b] <= o.0[b - 1] + 1e-4, "x={x}: band {b} ({}) louder than band {} ({})", o.0[b], b - 1, o.0[b - 1]);
            }
            assert!(o.0[7] < o.0[0], "x={x}: HF must be attenuated more than LF");
            checked += 1;
        }
    }
    assert!(checked >= 3);
}

#[test]
fn occlusion_is_deterministic() {
    let a = sweep(0.07);
    let b = sweep(0.07);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.1, y.1);
    }
}

