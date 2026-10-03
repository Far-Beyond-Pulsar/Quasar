use quasar_audio::quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_audio::quasar_core::backend::{IAcousticComputeBackend, SpatialQuery, SPEED_OF_SOUND};
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene};
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;
#[test] fn dbg() {
    let (lx, ly, lz) = (10.0_f32, 4.0_f32, 8.0_f32);
    let mut e = SpatialAudioEngine::new(0, 48000.0, 15.0);
    e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let wall = e.materials().add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, Tabular8BandEvaluator::create_params(Band8::splat(0.2), Band8::zeros(), Band8::zeros())));
    let quads: [[[f32; 3]; 4]; 6] = [
        [[0.0, 0.0, 0.0], [0.0, ly, 0.0], [0.0, ly, lz], [0.0, 0.0, lz]],
        [[lx, 0.0, 0.0], [lx, ly, 0.0], [lx, ly, lz], [lx, 0.0, lz]],
        [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, 0.0, lz], [0.0, 0.0, lz]],
        [[0.0, ly, 0.0], [lx, ly, 0.0], [lx, ly, lz], [0.0, ly, lz]],
        [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, ly, 0.0], [0.0, ly, 0.0]],
        [[0.0, 0.0, lz], [lx, 0.0, lz], [lx, ly, lz], [0.0, ly, lz]],
    ];
    let mut scene = AcousticScene::new();
    for (i, q) in quads.iter().enumerate() { scene.add_mesh(AcousticMesh::new(i as u64 + 1, q.to_vec(), vec![0, 1, 2, 0, 2, 3], wall)); }
    let b = CpuSimdComputeBackend::new(scene, CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() });
    let r = b.query_spatial(&[SpatialQuery { source_position: [2.0, 1.2, 2.0], listener_position: [6.0, 1.6, 5.0], source_id: 0 }], e.materials());
    for er in &r[0].early_reflections {
        let d = er.delay_samples * SPEED_OF_SOUND / 48000.0;
        let ideal = 0.8f32.sqrt() / d;
        let en: f32 = er.gain.0.iter().map(|g| g*g).sum::<f32>()/8.0;
        eprintln!("refl path {d:.2} m b0 ratio {:.3} b7 ratio {:.3} dir {:?} mean-energy ratio {:.2} dB", er.gain.0[0]/ideal, er.gain.0[7]/ideal, er.direction, 10.0*(en/(ideal*ideal)).log10());
    }
}
