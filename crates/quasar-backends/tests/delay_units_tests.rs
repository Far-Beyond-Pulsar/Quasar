//! #52: `delay_samples` is `path_length * fs / c` on every strategy / backend,
//! for the device sample rate, not a hardcoded 48 kHz.

use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_core::backend::{
    IAcousticComputeBackend, MaterialProvider, SpatialQuery, SPEED_OF_SOUND,
};
use quasar_core::bands::Band8;
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::rays::RayInteractionContext;
use quasar_core::scene::{AcousticMesh, AcousticScene};

const RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

struct Mat;
impl MaterialProvider for Mat {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(0.1)
    }
}

fn query() -> SpatialQuery {
    // 3-4-12 triple: distance exactly 13 m.
    SpatialQuery { source_position: [3.0, 4.0, 12.0], listener_position: [0.0, 0.0, 0.0], source_id: 0 }
}

fn expected(fs: f32) -> f32 {
    13.0 * fs / SPEED_OF_SOUND
}

fn floor_scene() -> AcousticScene {
    // Wall at x = 10 (the current tracer shoots its first ray from the listener
    // away from the source, so +x from a listener at x = 4 hits it).
    let mut s = AcousticScene::new();
    let p = vec![[10.0, -50.0, -50.0], [10.0, 50.0, -50.0], [10.0, 50.0, 50.0], [10.0, -50.0, 50.0]];
    s.add_mesh(AcousticMesh::new(1, p, vec![0, 1, 2, 0, 2, 3], 0));
    s
}

#[test]
fn cpu_simd_delay_scales_with_config_sample_rate() {
    for fs in RATES {
        let cfg = CpuSimdConfig { sample_rate: fs, ..CpuSimdConfig::default() };
        let b = CpuSimdComputeBackend::new(AcousticScene::new(), cfg);
        let r = b.query_spatial(&[query()], &Mat);
        let d = r[0].direct_path.delay_samples;
        assert!((d - expected(fs)).abs() < 1e-2, "fs {fs}: {d} vs {}", expected(fs));
    }
}

#[test]
fn cpu_simd_delay_follows_set_sample_rate() {
    let mut b = CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default());
    for fs in RATES {
        b.set_sample_rate(fs);
        let d = b.query_spatial(&[query()], &Mat)[0].direct_path.delay_samples;
        assert!((d - expected(fs)).abs() < 1e-2);
    }
    // Garbage rates are ignored.
    b.set_sample_rate(f32::NAN);
    b.set_sample_rate(-1.0);
    let d = b.query_spatial(&[query()], &Mat)[0].direct_path.delay_samples;
    assert!((d - expected(96_000.0)).abs() < 1e-2);
}

#[test]
fn cpu_simd_reflection_delays_scale_with_sample_rate() {
    let q = SpatialQuery { source_position: [0.0, 0.0, 0.0], listener_position: [4.0, 0.0, 0.0], source_id: 0 };
    let base = {
        let b = CpuSimdComputeBackend::new(floor_scene(), CpuSimdConfig { sample_rate: 48_000.0, ..CpuSimdConfig::default() });
        b.query_spatial(&[q.clone()], &Mat)[0].early_reflections.clone()
    };
    assert!(!base.is_empty(), "floor scene must produce a reflection");
    for fs in RATES {
        let b = CpuSimdComputeBackend::new(floor_scene(), CpuSimdConfig { sample_rate: fs, ..CpuSimdConfig::default() });
        let refl = b.query_spatial(&[q.clone()], &Mat)[0].early_reflections.clone();
        assert_eq!(refl.len(), base.len());
        for (a, r) in base.iter().zip(refl.iter()) {
            // Same path, so delay in seconds is unchanged.
            let secs_a = a.delay_samples / 48_000.0;
            let secs_r = r.delay_samples / fs;
            assert!((secs_a - secs_r).abs() < 1e-6, "fs {fs}: {secs_a} vs {secs_r}");
        }
    }
}

#[test]
fn hw_stub_delay_scales_with_sample_rate() {
    let mut s = HardwareAcceleratorStub::new();
    for fs in RATES {
        s.set_sample_rate(fs);
        let d = s.query_spatial(&[query()], &Mat)[0].direct_path.delay_samples;
        assert!((d - expected(fs)).abs() < 1e-2);
    }
}

fn grid() -> AcousticProbeGrid {
    let probe = |x: f32, y: f32, z: f32| AcousticProbe {
        position: [x, y, z],
        rir_samples: Vec::new(),
        sample_rate: 48_000,
        t60: Band8::splat(1.0),
        broadband_t60: 1.0,
        early_late_split_secs: 0.05,
    };
    let mut probes = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                probes.push(probe(x as f32 * 2.0 - 1.0, y as f32 * 2.0 - 1.0, z as f32 * 2.0 - 1.0));
            }
        }
    }
    AcousticProbeGrid::new(probes, [-1.0; 3], [2.0; 3], [2, 2, 2]).unwrap()
}

#[test]
fn baked_only_delay_is_distance_times_fs_over_c() {
    for fs in RATES {
        let mut h = HybridProbeSampler::new(HybridSamplingStrategy::BakedOnly);
        h.set_probe_grid(grid());
        h.set_sample_rate(fs);
        let r = h.resolve(&query(), &Mat).unwrap();
        assert!((r.direct_path.delay_samples - expected(fs)).abs() < 1e-2, "fs {fs}");
    }
}

#[test]
fn hybrid_blend_and_realtime_forward_the_rate_to_the_backend() {
    for strategy in [HybridSamplingStrategy::RealTimeOnly, HybridSamplingStrategy::HybridBlend] {
        for fs in RATES {
            let mut h = HybridProbeSampler::new(strategy);
            h.set_probe_grid(grid());
            // Rate set before and after installing the backend must both reach it.
            h.set_sample_rate(fs);
            h.set_realtime_backend(Box::new(CpuSimdComputeBackend::new(AcousticScene::new(), CpuSimdConfig::default())));
            let r = h.resolve(&query(), &Mat).unwrap();
            assert!((r.direct_path.delay_samples - expected(fs)).abs() < 1e-2, "{strategy:?} fs {fs}");
            h.set_sample_rate(48_000.0);
            let r = h.resolve(&query(), &Mat).unwrap();
            assert!((r.direct_path.delay_samples - expected(48_000.0)).abs() < 1e-2);
        }
    }
}
