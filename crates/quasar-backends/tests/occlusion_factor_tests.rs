//! M1 regression: the traced occlusion factor must reach the signal.
//!
//! `compute_direct_path` computes `occlusion_factor = 1 - material absorption
//! mean` (clear path = 1.0) and stored it on `DirectPathResult`, but never
//! applied it to the attenuation. This test verifies that an occluded path
//! (ray through a high-absorption wall) attenuates strictly more than the same
//! distance clear path.

use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::bands::Band8;
use quasar_core::scene::{AcousticMesh, AcousticScene};
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_materials::instance::AcousticMaterialInstance;
use quasar_materials::registry::AcousticMaterialRegistry;
use quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};

#[cfg(feature = "cpu-simd")]
#[test]
fn occluded_path_attenuates_more_than_clear_path() {
    // High-absorption wall material: mean absorption 0.9 → occlusion factor 0.1.
    let registry = AcousticMaterialRegistry::new();
    registry.register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let wall_handle = registry.add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.9), Band8::zeros(), Band8::zeros()),
    ));

    let config = CpuSimdConfig::default();
    let query = SpatialQuery {
        source_position: [0.0, 0.0, 0.0],
        listener_position: [10.0, 0.0, 0.0],
        source_id: 0,
    };

    // Clear path: empty scene, no geometry between source and listener.
    let clear_backend = CpuSimdComputeBackend::new(AcousticScene::new(), config.clone());
    let clear = clear_backend.query_spatial(&[query.clone()], &registry);
    assert_eq!(clear.len(), 1);
    assert!(!clear[0].direct_path.occluded);
    assert!(
        (clear[0].direct_path.occlusion_factor - 1.0).abs() < 1e-6,
        "clear path must keep occlusion_factor == 1.0"
    );

    // Occluded path: thin wall quad at x=5 spanning y,z in [-1, 1].
    let mut wall_scene = AcousticScene::new();
    let positions = vec![
        [5.0, -1.0, -1.0],
        [5.0, -1.0, 1.0],
        [5.0, 1.0, 1.0],
        [5.0, 1.0, -1.0],
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    wall_scene.add_mesh(AcousticMesh::new(1, positions, indices, wall_handle));
    let wall_backend = CpuSimdComputeBackend::new(wall_scene, config);
    let occluded = wall_backend.query_spatial(&[query], &registry);
    assert_eq!(occluded.len(), 1);
    assert!(occluded[0].direct_path.occluded);
    assert!(
        occluded[0].direct_path.occlusion_factor < 1.0,
        "wall between source and listener must lower the occlusion factor"
    );

    let clear_mean = clear[0].direct_path.attenuation.mean();
    let occluded_mean = occluded[0].direct_path.attenuation.mean();
    assert!(
        occluded_mean < clear_mean,
        "occluded path mean attenuation ({occluded_mean}) must be strictly less than \
         clear path mean attenuation ({clear_mean})"
    );
}
