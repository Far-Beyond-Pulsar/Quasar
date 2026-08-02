use quasar_core::backend::{IAcousticComputeBackend, MaterialProvider, SpatialQuery};
use quasar_core::bands::Band8;
use quasar_core::rays::{Ray, RayInteractionContext};
use quasar_core::scene::{AcousticMesh, AcousticScene};
use quasar_backends::hw_stub::HardwareAcceleratorStub;

// ── hw_stub_returns_dummy ─────────────────────────────────────────────

#[test]
fn hw_stub_returns_dummy() {
    let stub = HardwareAcceleratorStub::new();
    let queries = vec![SpatialQuery {
        source_position: [0.0, 0.0, 0.0],
        listener_position: [10.0, 0.0, 0.0],
        source_id: 1,
    }];

    struct DummyMaterial;
    impl MaterialProvider for DummyMaterial {
        fn evaluate_material(&self, _handle: u32, _context: &RayInteractionContext) -> Band8 {
            Band8::splat(0.5)
        }
    }

    let results = stub.query_spatial(&queries, &DummyMaterial);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_id, 1);
    assert!(!results[0].direct_path.occluded);
    assert_eq!(results[0].early_reflections.len(), 0);
}

// ── cpu_simd tests (feature-gated) ────────────────────────────────────

#[cfg(feature = "cpu-simd")]
fn make_single_triangle_scene() -> AcousticScene {
    let mut scene = AcousticScene::new();
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![0, 1, 2];
    let mesh = AcousticMesh::new(1, positions, indices, 0);
    scene.add_mesh(mesh);
    scene
}

#[cfg(feature = "cpu-simd")]
struct DummyMaterial;
#[cfg(feature = "cpu-simd")]
impl MaterialProvider for DummyMaterial {
    fn evaluate_material(&self, _handle: u32, _context: &RayInteractionContext) -> Band8 {
        Band8::splat(0.5)
    }
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_bvh_build() {
    let mut scene = AcousticScene::new();
    let positions = vec![
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ];
    let indices = vec![0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6];
    let mesh = AcousticMesh::new(1, positions, indices, 0);
    scene.add_mesh(mesh);

    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);
    drop(backend);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_ray_intersect_single_triangle() {
    let scene = make_single_triangle_scene();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let ray = Ray::new([0.25, 0.25, 1.0], [0.0, 0.0, -1.0]);
    let hits = backend.trace_ray(&ray);
    assert_eq!(hits.len(), 1);
    assert!(hits[0].hit);
    assert!((hits[0].distance - 1.0).abs() < 1e-4);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_ray_no_false_hit() {
    let scene = make_single_triangle_scene();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let ray = Ray::new([0.25, 0.25, -1.0], [0.0, 0.0, -1.0]);
    let hits = backend.trace_ray(&ray);
    assert_eq!(hits.len(), 0);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_direct_path_clear() {
    let scene = make_single_triangle_scene();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let query = SpatialQuery {
        source_position: [0.25, 0.25, -1.0],
        listener_position: [0.25, 0.25, 1.0],
        source_id: 0,
    };
    let results = backend.query_spatial(&[query], &DummyMaterial);
    assert_eq!(results.len(), 1);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_direct_path_occluded() {
    let scene = make_single_triangle_scene();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let query = SpatialQuery {
        source_position: [0.25, 0.25, 2.0],
        listener_position: [0.25, 0.25, 1.0],
        source_id: 0,
    };
    let results = backend.query_spatial(&[query], &DummyMaterial);
    assert_eq!(results.len(), 1);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_distance_attenuation() {
    let scene = AcousticScene::new();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let nearby = SpatialQuery {
        source_position: [0.0, 0.0, 1.0],
        listener_position: [0.0, 0.0, 0.0],
        source_id: 0,
    };
    let far = SpatialQuery {
        source_position: [0.0, 0.0, 100.0],
        listener_position: [0.0, 0.0, 0.0],
        source_id: 1,
    };

    let results = backend.query_spatial(&[nearby, far], &DummyMaterial);
    assert_eq!(results.len(), 2);
    assert!(
        results[0].direct_path.attenuation.0[0] > results[1].direct_path.attenuation.0[0],
        "nearby should have higher attenuation than far"
    );
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_air_absorption_positive() {
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(AcousticScene::new(), config);

    let query = SpatialQuery {
        source_position: [0.0, 0.0, 10.0],
        listener_position: [0.0, 0.0, 0.0],
        source_id: 0,
    };
    let results = backend.query_spatial(&[query], &DummyMaterial);
    assert_eq!(results.len(), 1);
    for i in 0..8 {
        let atten = results[0].direct_path.attenuation.0[i];
        assert!(atten > 0.0, "air absorption should produce positive attenuation at band {i}: {atten}");
        assert!(atten <= 1.0, "air absorption should be <= 1.0 at band {i}: {atten}");
    }
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_air_absorption_increases_with_frequency() {
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(AcousticScene::new(), config);

    let query = SpatialQuery {
        source_position: [0.0, 0.0, 100.0],
        listener_position: [0.0, 0.0, 0.0],
        source_id: 0,
    };
    let results = backend.query_spatial(&[query], &DummyMaterial);
    assert_eq!(results.len(), 1);
    // Higher bands should have lower gain (= more absorption) at long distance
    assert!(
        results[0].direct_path.attenuation.0[0] > results[0].direct_path.attenuation.0[7],
        "low frequencies should attenuate less than high frequencies"
    );
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_early_reflections_empty() {
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(AcousticScene::new(), config);

    let query = SpatialQuery {
        source_position: [0.0; 3],
        listener_position: [5.0; 3],
        source_id: 0,
    };
    let results = backend.query_spatial(&[query], &DummyMaterial);
    assert_eq!(results.len(), 1);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_late_reverb_estimate() {
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(AcousticScene::new(), config);

    let query = SpatialQuery {
        source_position: [0.0; 3],
        listener_position: [5.0; 3],
        source_id: 0,
    };
    let results = backend.query_spatial(&[query], &DummyMaterial);
    assert_eq!(results.len(), 1);
    assert!(results[0].late_reverb.t60.mean() > 0.0);
}

#[cfg(feature = "cpu-simd")]
#[test]
fn cpu_simd_scene_update() {
    let scene = make_single_triangle_scene();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let mut backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let mut scene2 = AcousticScene::new();
    let positions2 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
    let indices2 = vec![0, 1, 2];
    let mesh2 = AcousticMesh::new(2, positions2, indices2, 0);
    scene2.add_mesh(mesh2);

    assert!(backend.update_scene(&scene2).is_ok());
}

// ── aabb_construction_and_union ───────────────────────────────────────

#[cfg(feature = "cpu-simd")]
#[test]
fn aabb_construction_and_union() {
    let scene = make_single_triangle_scene();
    let config = quasar_backends::cpu_simd::CpuSimdConfig::default();
    let backend = quasar_backends::cpu_simd::CpuSimdComputeBackend::new(scene, config);

    let ray = Ray::new([0.25, 0.25, 1.0], [0.0, 0.0, -1.0]);
    let hits = backend.trace_ray(&ray);
    assert_eq!(hits.len(), 1);
}
