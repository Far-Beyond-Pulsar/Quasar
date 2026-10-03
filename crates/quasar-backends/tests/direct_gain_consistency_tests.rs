//! #51 / #53: the same clear-path query yields the same direct gain through the
//! CPU backend, BakedOnly and HybridBlend, and follows the shared distance model.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::{MaterialProvider, SpatialQuery};
use quasar_core::bands::Band8;
use quasar_core::distance::{DistanceCurve, DistanceModel};
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::rays::RayInteractionContext;
use quasar_core::scene::AcousticScene;

struct Mat;
impl MaterialProvider for Mat {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(0.1)
    }
}

fn grid() -> AcousticProbeGrid {
    let mut probes = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                probes.push(AcousticProbe {
                    position: [x as f32 * 2.0 - 1.0, y as f32 * 2.0 - 1.0, z as f32 * 2.0 - 1.0],
                    rir_samples: Vec::new(),
                    sample_rate: 48_000,
                    t60: Band8::splat(1.0),
                    broadband_t60: 1.0,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }
    AcousticProbeGrid::new(probes, [-1.0; 3], [2.0; 3], [2, 2, 2]).unwrap()
}

fn sampler(strategy: HybridSamplingStrategy, model: Option<DistanceModel>) -> HybridProbeSampler {
    let mut h = HybridProbeSampler::new(strategy);
    h.set_probe_grid(grid());
    if let Some(m) = model {
        h.set_distance_model(m);
    }
    h.set_realtime_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default())));
    h
}

fn q(d: f32) -> SpatialQuery {
    SpatialQuery { source_position: [0.0, 0.0, d], listener_position: [0.0, 0.0, 0.0], source_id: 0 }
}

#[test]
fn all_strategies_give_the_same_direct_gain() {
    for model in [None, Some(DistanceModel { curve: DistanceCurve::Exponential, rolloff_factor: 1.5, ..DistanceModel::default() }),
                  Some(DistanceModel { reference_distance: 2.0, min_distance: 2.0, ..DistanceModel::default() })] {
        for d in [0.2_f32, 1.0, 3.0, 17.0, 120.0] {
            let baked = sampler(HybridSamplingStrategy::BakedOnly, model).resolve(&q(d), &Mat).unwrap();
            let rt = sampler(HybridSamplingStrategy::RealTimeOnly, model).resolve(&q(d), &Mat).unwrap();
            let hy = sampler(HybridSamplingStrategy::HybridBlend, model).resolve(&q(d), &Mat).unwrap();
            for b in 0..8 {
                let (x, y, z) = (baked.direct_path.attenuation.0[b], rt.direct_path.attenuation.0[b], hy.direct_path.attenuation.0[b]);
                assert!((x - y).abs() < 1e-4 && (y - z).abs() < 1e-4, "d={d} band {b}: baked {x} rt {y} hybrid {z}");
            }
        }
    }
}

#[test]
fn default_gain_drops_6_02_db_per_doubling_at_low_band() {
    // 62.5 Hz has negligible air absorption, so the band is the pure distance law.
    let s = sampler(HybridSamplingStrategy::RealTimeOnly, None);
    let g = |d: f32| s.resolve(&q(d), &Mat).unwrap().direct_path.attenuation.0[0];
    for d in [1.0_f32, 2.0, 5.0, 20.0] {
        let db = 20.0 * (g(2.0 * d) / g(d)).log10();
        assert!((db + 6.02).abs() < 0.02, "doubling at {d}: {db} dB");
    }
}

#[test]
fn near_field_is_clamped_to_unity() {
    let s = sampler(HybridSamplingStrategy::RealTimeOnly, None);
    let g = s.resolve(&q(0.05), &Mat).unwrap().direct_path.attenuation.0[0];
    assert!(g <= 1.0 + 1e-6 && g > 0.99, "{g}");
}
