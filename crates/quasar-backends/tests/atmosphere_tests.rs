//! #121: `set_atmosphere` reaches every strategy, so temperature / humidity change the
//! direct-path band gains consistently on baked, real-time and hybrid sampling.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::{IAcousticComputeBackend, MaterialProvider, SpatialQuery};
use quasar_core::bands::Band8;
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

fn sampler(strategy: HybridSamplingStrategy) -> HybridProbeSampler {
    let mut h = HybridProbeSampler::new(strategy);
    h.set_probe_grid(grid());
    h.set_realtime_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default())));
    h
}

fn atten(h: &HybridProbeSampler, d: f32) -> Band8 {
    let q = SpatialQuery { source_position: [0.0, 0.0, d], listener_position: [0.0, 0.0, 0.0], source_id: 0 };
    h.resolve(&q, &Mat).unwrap().direct_path.attenuation
}

#[test]
fn atmosphere_changes_gains_consistently_across_strategies() {
    let d = 30.0;
    let strategies = [HybridSamplingStrategy::BakedOnly, HybridSamplingStrategy::RealTimeOnly, HybridSamplingStrategy::HybridBlend];
    let mut high_band = Vec::new();
    for &(t, rh) in &[(20.0_f32, 50.0_f32), (-5.0, 20.0), (35.0, 90.0)] {
        let mut per = Vec::new();
        for &s in &strategies {
            let mut h = sampler(s);
            h.set_atmosphere(t, rh);
            per.push(atten(&h, d));
        }
        for b in 0..8 {
            let (x, y, z) = (per[0].0[b], per[1].0[b], per[2].0[b]);
            assert!((x - y).abs() < 1e-4 && (y - z).abs() < 1e-4, "t={t} rh={rh} band {b}: {x} {y} {z}");
        }
        high_band.push(per[0].0[7]);
    }
    // The atmosphere really changes the top band (and not just by rounding).
    assert!((high_band[0] - high_band[1]).abs() / high_band[0] > 0.1 && (high_band[0] - high_band[2]).abs() / high_band[0] > 0.1, "{high_band:?}");
}

#[test]
fn setting_atmosphere_before_or_after_installing_the_backend_is_equivalent() {
    let mut a = HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly);
    a.set_atmosphere(-5.0, 20.0);
    a.set_realtime_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default())));
    let mut b = sampler(HybridSamplingStrategy::RealTimeOnly);
    b.set_atmosphere(-5.0, 20.0);
    let (ga, gb) = (atten(&a, 80.0), atten(&b, 80.0));
    for i in 0..8 {
        assert!((ga.0[i] - gb.0[i]).abs() < 1e-6);
    }
}

#[test]
fn backend_keeps_its_own_atmosphere_until_one_is_set() {
    // A sampler that never called set_atmosphere must not clobber the backend's config.
    let cfg = CpuSimdConfig { temperature_celsius: -5.0, humidity_percent: 20.0, ..CpuSimdConfig::default() };
    let mut h = HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly);
    h.set_realtime_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), cfg)));
    let mut direct = CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default());
    direct.set_atmosphere(-5.0, 20.0);
    let q = SpatialQuery { source_position: [0.0, 0.0, 80.0], listener_position: [0.0; 3], source_id: 0 };
    let a = h.resolve(&q, &Mat).unwrap().direct_path.attenuation;
    let b = direct.query_spatial(&[q], &Mat)[0].direct_path.attenuation;
    for i in 0..8 {
        assert!((a.0[i] - b.0[i]).abs() < 1e-6);
    }
}
