use quasar_core::bands::Band8;
use quasar_core::backend::{
    DirectPathResult, EarlyReflection, LateReverbEstimate, SpatialQueryResult,
};
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::param_exchange::{ParameterTripleBuffer, SpatialCoefficients};
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::rays::{Ray, RayHit};
use quasar_core::scene::{AcousticMesh, AcousticScene};

// ── band8_math_operations ─────────────────────────────────────────────

#[test]
fn band8_math_operations() {
    let a = Band8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Band8::new([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);

    let sum = a.add(&b);
    assert_eq!(sum.0[0], 1.5);
    assert_eq!(sum.0[7], 12.0);

    let diff = a.sub(&b);
    assert_eq!(diff.0[0], 0.5);
    assert_eq!(diff.0[7], 4.0);

    let prod = a.mul(&b);
    assert_eq!(prod.0[0], 0.5);
    assert_eq!(prod.0[7], 32.0);

    let scaled = a.scale(2.0);
    assert_eq!(scaled.0[0], 2.0);
    assert_eq!(scaled.0[7], 16.0);

    let lerped = a.lerp(&b, 0.5);
    assert!((lerped.0[0] - 0.75).abs() < 1e-6);
    assert!((lerped.0[7] - 6.0).abs() < 1e-6);

    let gain_db = Band8::splat(-6.0);
    let gained = Band8::splat(1.0).apply_gain_db(&gain_db);
    assert!((gained.0[0] - 0.501187).abs() < 1e-3);

    let db = Band8::splat(1.0).to_db();
    assert!((db.0[0] - 0.0).abs() < 1e-5);

    let db2 = Band8::zeros().to_db();
    assert_eq!(db2.0[0], -160.0);

    assert!((Band8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).sum() - 36.0).abs() < 1e-6);
    assert!((Band8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).mean() - 4.5).abs() < 1e-6);
}

// ── band8_lerp_accuracy ───────────────────────────────────────────────

#[test]
fn band8_lerp_accuracy() {
    let a = Band8::new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let b = Band8::new([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]);

    let at0 = a.lerp(&b, 0.0);
    for i in 0..8 {
        assert!((at0.0[i] - a.0[i]).abs() < 1e-6);
    }

    let at1 = a.lerp(&b, 1.0);
    for i in 0..8 {
        assert!((at1.0[i] - b.0[i]).abs() < 1e-6);
    }

    let at05 = a.lerp(&b, 0.5);
    for i in 0..8 {
        let expected = (a.0[i] + b.0[i]) * 0.5;
        assert!((at05.0[i] - expected).abs() < 1e-6);
    }
}

// ── ray_construction ───────────────────────────────────────────────────

#[test]
fn ray_construction() {
    let r = Ray::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0]);
    assert_eq!(r.origin, [1.0, 2.0, 3.0]);
    assert_eq!(r.direction, [0.0, 1.0, 0.0]);
    assert_eq!(r.min_distance, 0.0);
    assert_eq!(r.max_distance, f32::MAX);

    let pt = r.point_at(5.0);
    assert_eq!(pt, [1.0, 7.0, 3.0]);
}

// ── ray_hit_miss ──────────────────────────────────────────────────────

#[test]
fn ray_hit_miss() {
    let miss = RayHit::miss();
    assert!(!miss.hit);
    assert_eq!(miss.distance, f32::MAX);
    assert_eq!(miss.material_handle, 0);
}

// ── triple_buffer_write_read ──────────────────────────────────────────

#[test]
fn triple_buffer_write_read() {
    let initial = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    let tb = ParameterTripleBuffer::new(1, initial);

    let updated = SpatialCoefficients {
        source_id: 1,
        direct_gain: Band8::splat(0.8),
        direct_delay_samples: 12.5,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(1.2),
        late_gain_db: -6.0,
        version: 0,
    };

    unsafe {
        *tb.begin_write(0) = updated.clone();
    }
    tb.end_write(0);

    tb.update();

    let read = unsafe { tb.read(0) };
    assert_eq!(read.source_id, updated.source_id);
    assert!((read.direct_gain.0[0] - 0.8).abs() < 1e-6);
    assert!((read.direct_delay_samples - 12.5).abs() < 1e-6);
    assert!((read.late_t60.0[3] - 1.2).abs() < 1e-6);
    assert!((read.late_gain_db + 6.0).abs() < 1e-6);
}

// ── triple_buffer_no_data_loss ────────────────────────────────────────

#[test]
fn triple_buffer_no_data_loss() {
    let initial = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    let tb = ParameterTripleBuffer::new(1, initial);

    for i in 1..=5 {
        let v = SpatialCoefficients {
            source_id: i,
            direct_gain: Band8::splat(i as f32 * 0.1),
            direct_delay_samples: i as f32,
            early_reflections: Vec::new(),
            late_t60: Band8::splat(0.5),
            late_gain_db: -10.0,
            version: 0,
        };
        unsafe { *tb.begin_write(0) = v; }
        tb.end_write(0);
    }

    tb.update();
    let read = unsafe { tb.read(0) };
    assert_eq!(read.source_id, 5);
    assert!((read.direct_gain.0[0] - 0.5).abs() < 1e-6);
}

// ── triple_buffer_thread_safety_loom ──────────────────────────────────

#[test]
fn triple_buffer_thread_safety_loom() {
    let initial = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    let tb = std::sync::Arc::new(ParameterTripleBuffer::new(1, initial));

    let tb_writer = tb.clone();
    let writer = std::thread::spawn(move || {
        for i in 0..100 {
            let v = SpatialCoefficients {
                source_id: i,
                direct_gain: Band8::splat(i as f32 * 0.01),
                direct_delay_samples: i as f32,
                early_reflections: Vec::new(),
                late_t60: Band8::splat(0.5),
                late_gain_db: -10.0,
                version: 0,
            };
            unsafe { *tb_writer.begin_write(0) = v; }
            tb_writer.end_write(0);
            std::thread::yield_now();
        }
    });

    let tb_reader = tb.clone();
    let reader = std::thread::spawn(move || {
        let mut last_version = 0u32;
        for _ in 0..500 {
            tb_reader.update();
            let r = unsafe { tb_reader.read(0) };
            assert!(r.source_id >= last_version);
            last_version = r.source_id;
            std::thread::yield_now();
        }
    });

    writer.join().unwrap();
    reader.join().unwrap();
}

// ── acoustic_scene_create ─────────────────────────────────────────────

#[test]
fn acoustic_scene_create() {
    let mut scene = AcousticScene::new();
    assert!(scene.is_empty());
    assert_eq!(scene.total_triangle_count(), 0);

    let mesh = AcousticMesh::new(
        1,
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 2],
        42,
    );
    let idx = scene.add_mesh(mesh);
    assert_eq!(idx, 0);
    assert!(!scene.is_empty());
    assert_eq!(scene.total_triangle_count(), 1);
}

// ── probe_grid_trilinear ──────────────────────────────────────────────

#[test]
fn probe_grid_trilinear() {
    let mut probes = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let t60 = Band8::splat((x + y + z) as f32 * 0.5);
                probes.push(AcousticProbe {
                    position: [x as f32, y as f32, z as f32],
                    rir_samples: Vec::new(),
                    sample_rate: 48000,
                    t60,
                    broadband_t60: (x + y + z) as f32 * 0.5,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }

    let grid = AcousticProbeGrid::new(probes, [0.0; 3], [1.0; 3], [2, 2, 2]).unwrap();
    assert_eq!(grid.len(), 8);

    let sample = grid.sample(&[0.5, 0.5, 0.5]).expect("should be inside grid");
    assert!((sample.t60.0[0] - 1.5).abs() < 1e-4);
    assert!((sample.interpolation_quality - 1.0).abs() < 1e-4);
}

// ── probe_grid_out_of_bounds ──────────────────────────────────────────

#[test]
fn probe_grid_out_of_bounds() {
    let probes = (0..8)
        .map(|_| AcousticProbe {
            position: [0.0; 3],
            rir_samples: Vec::new(),
            sample_rate: 48000,
            t60: Band8::zeros(),
            broadband_t60: 0.0,
            early_late_split_secs: 0.05,
        })
        .collect();

    let grid = AcousticProbeGrid::new(probes, [0.0; 3], [1.0; 3], [2, 2, 2]).unwrap();
    assert!(grid.sample(&[-1.0, 0.5, 0.5]).is_none());
    assert!(grid.sample(&[0.5, 0.5, 3.0]).is_none());
}

// ── hybrid_sampler_strategies ─────────────────────────────────────────

#[test]
fn hybrid_sampler_strategies() {
    let sampler = HybridProbeSampler::new(HybridSamplingStrategy::BakedOnly);
    assert_eq!(sampler.strategy(), HybridSamplingStrategy::BakedOnly);

    let sampler2 = HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly);
    assert_eq!(sampler2.strategy(), HybridSamplingStrategy::RealTimeOnly);

    let sampler3 = HybridProbeSampler::new(HybridSamplingStrategy::HybridBlend);
    assert_eq!(sampler3.strategy(), HybridSamplingStrategy::HybridBlend);
}

// ── spatial_query_result_construction ─────────────────────────────────

#[test]
fn spatial_query_result_construction() {
    let result = SpatialQueryResult {
        source_id: 7,
        direct_path: DirectPathResult {
            attenuation: Band8::splat(0.5),
            delay_samples: 23.0,
            distance: 5.0,
            occluded: false,
            occlusion_factor: 1.0,
        },
        early_reflections: vec![EarlyReflection {
            direction: [1.0, 0.0, 0.0],
            delay_samples: 45.0,
            gain: Band8::splat(0.3),
            order: 1,
        }],
        late_reverb: LateReverbEstimate {
            t60: Band8::splat(1.5),
            early_late_split_secs: 0.05,
            late_loudness_db: -8.0,
        },
    };

    assert_eq!(result.source_id, 7);
    assert!((result.direct_path.attenuation.0[0] - 0.5).abs() < 1e-6);
    assert_eq!(result.early_reflections.len(), 1);
    assert_eq!(result.early_reflections[0].order, 1);
    assert!((result.late_reverb.t60.0[3] - 1.5).abs() < 1e-6);
    assert!((result.late_reverb.late_loudness_db + 8.0).abs() < 1e-6);
}
