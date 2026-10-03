//! #151: `HybridProbeSampler::resolve_batch` is bitwise identical to resolving every pair on its
//! own, for every strategy and for listeners inside / outside the probe grid. The ignored timing
//! test compares batched vs serial wall time on a large procedural scene:
//! `cargo test -p quasar-backends --release --test batch_resolve_tests -- --ignored --nocapture`.

mod common;

use common::{box_mesh, hall, Opaque, INSIDE, SRC};
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::{SpatialQuery, SpatialQueryResult};
use quasar_core::bands::Band8;
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::scene::AcousticScene;

/// 2 x 2 x 2 probes spanning most of the hall interior (0..20, 0..12, 0..30).
fn grid() -> AcousticProbeGrid {
    let mut probes = Vec::new();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                probes.push(AcousticProbe {
                    position: [1.0 + x as f32 * 18.0, 1.0 + y as f32 * 10.0, 1.0 + z as f32 * 28.0],
                    rir_samples: Vec::new(),
                    sample_rate: 48_000,
                    t60: Band8::splat(1.0 + 0.5 * (x + y + z) as f32),
                    broadband_t60: 1.5,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }
    AcousticProbeGrid::new(probes, [1.0, 1.0, 1.0], [18.0, 10.0, 28.0], [2, 2, 2]).unwrap()
}

fn sampler(strategy: HybridSamplingStrategy, scene: AcousticScene) -> HybridProbeSampler {
    let mut h = HybridProbeSampler::new(strategy);
    h.set_probe_grid(grid());
    h.set_realtime_backend(Box::new(CpuSimdComputeBackend::new(scene, CpuSimdConfig::default())));
    h
}

fn assert_same(a: &SpatialQueryResult, b: &SpatialQueryResult, what: &str) {
    assert_eq!(a.source_id, b.source_id, "{what}: id");
    let (x, y) = (&a.direct_path, &b.direct_path);
    assert_eq!(x.attenuation.0, y.attenuation.0, "{what}: attenuation");
    assert_eq!(x.delay_samples.to_bits(), y.delay_samples.to_bits(), "{what}: delay");
    assert_eq!(x.distance.to_bits(), y.distance.to_bits(), "{what}: distance");
    assert_eq!(x.occluded, y.occluded, "{what}: occluded");
    assert_eq!(x.occlusion_factor.to_bits(), y.occlusion_factor.to_bits(), "{what}: occlusion factor");
    assert_eq!(x.occlusion.0, y.occlusion.0, "{what}: occlusion");
    assert_eq!(a.late_reverb.t60.0, b.late_reverb.t60.0, "{what}: t60");
    assert_eq!(a.late_reverb.early_late_split_secs.to_bits(), b.late_reverb.early_late_split_secs.to_bits(), "{what}: split");
    assert_eq!(a.late_reverb.late_loudness_db.to_bits(), b.late_reverb.late_loudness_db.to_bits(), "{what}: late level");
    assert_eq!(a.early_reflections.len(), b.early_reflections.len(), "{what}: reflection count");
    for (p, q) in a.early_reflections.iter().zip(&b.early_reflections) {
        assert_eq!(
            (p.order, p.direction, p.delay_samples.to_bits(), p.gain.0),
            (q.order, q.direction, q.delay_samples.to_bits(), q.gain.0),
            "{what}: reflection"
        );
    }
}

/// Two emitters x listeners inside the grid, inside the hall but outside the grid, outside the hall.
fn queries() -> Vec<SpatialQuery> {
    let listeners = [INSIDE, [10.0, 6.0, 15.0], [0.4, 1.7, 15.0], [-6.0, 1.7, 15.0]];
    let mut v = Vec::new();
    for (i, l) in listeners.iter().enumerate() {
        for (j, s) in [SRC, [4.0, 3.0, 9.0]].iter().enumerate() {
            v.push(SpatialQuery { source_position: *s, listener_position: *l, source_id: (i * 2 + j) as u32 });
        }
    }
    v
}

#[test]
fn batch_is_bitwise_identical_to_per_pair_for_every_strategy() {
    let qs = queries();
    for strategy in [HybridSamplingStrategy::RealTimeOnly, HybridSamplingStrategy::HybridBlend, HybridSamplingStrategy::BakedOnly] {
        let h = sampler(strategy, hall());
        let batch = h.resolve_batch(&qs, &Opaque);
        assert_eq!(batch.len(), qs.len());
        let mut ok = 0;
        for (q, b) in qs.iter().zip(&batch) {
            let single = h.resolve(q, &Opaque);
            match (b, &single) {
                (Ok(b), Ok(s)) => {
                    ok += 1;
                    assert_same(b, s, &format!("{strategy:?} id {}", q.source_id));
                }
                (Err(_), Err(_)) => assert_eq!(strategy, HybridSamplingStrategy::BakedOnly, "only BakedOnly may fail (listener outside grid)"),
                _ => panic!("{strategy:?} id {}: batch and single disagree on success", q.source_id),
            }
        }
        assert!(ok > 0, "{strategy:?}: nothing resolved");
    }
}

#[test]
fn hybrid_blend_overlays_baked_estimate_only_inside_the_grid() {
    let h = sampler(HybridSamplingStrategy::HybridBlend, hall());
    let rt = sampler(HybridSamplingStrategy::RealTimeOnly, hall());
    let qs = queries();
    let blend = h.resolve_batch(&qs, &Opaque);
    let plain = rt.resolve_batch(&qs, &Opaque);
    let g = grid();
    let (mut inside, mut outside) = (0, 0);
    for (q, (b, p)) in qs.iter().zip(blend.iter().zip(&plain)) {
        let (b, p) = (b.as_ref().unwrap(), p.as_ref().unwrap());
        if g.sample(&q.listener_position).is_some() {
            inside += 1;
            assert!((b.late_reverb.early_late_split_secs - 0.05).abs() < 1e-5, "inside the grid the baked estimate is used");
        } else {
            outside += 1;
            assert_eq!(b.late_reverb.t60.0, p.late_reverb.t60.0, "outside the grid the statistical estimate stays");
        }
        assert_eq!(b.direct_path.delay_samples.to_bits(), p.direct_path.delay_samples.to_bits());
    }
    assert!(inside > 0 && outside > 0, "test must cover both cases");
}

#[test]
fn empty_batch_and_missing_configuration() {
    let h = sampler(HybridSamplingStrategy::RealTimeOnly, hall());
    assert!(h.resolve_batch(&[], &Opaque).is_empty());
    let none = HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly);
    let r = none.resolve_batch(&queries(), &Opaque);
    assert_eq!(r.len(), queries().len());
    assert!(r.iter().all(|x| x.is_err()));
}

// ── timing ──────────────────────────────────────────────────────────────

/// Hall 40 x 15 x 60 with roughly `target_tris` triangles of small boxes (ribs / columns).
fn big_scene(target_tris: usize) -> AcousticScene {
    let mut s = AcousticScene::new();
    s.add_mesh(box_mesh(1, [0.0, 0.0, 0.0], [40.0, 15.0, 60.0], true));
    let boxes = target_tris / 12;
    let (nx, nz) = (60usize, 100usize);
    let mut n = 0usize;
    let mut iy = 0usize;
    while n < boxes {
        for ix in 0..nx {
            for iz in 0..nz {
                if n >= boxes {
                    break;
                }
                let x = 0.5 + ix as f32 * (39.0 / nx as f32);
                let z = 0.5 + iz as f32 * (59.0 / nz as f32);
                let y = 9.0 + iy as f32 * 0.5;
                s.add_mesh(box_mesh(10 + n as u64, [x, y, z], [x + 0.25, y + 0.25, z + 0.25], false));
                n += 1;
            }
        }
        iy += 1;
    }
    s
}

#[test]
#[ignore = "timing report; run in release"]
fn batched_vs_serial_timing_on_a_large_scene() {
    use std::time::Instant;
    let t = Instant::now();
    let backend = CpuSimdComputeBackend::new(big_scene(400_000), CpuSimdConfig::default());
    println!(
        "scene build (BVH + planes): {:.2} s, cores {}",
        t.elapsed().as_secs_f32(),
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
    let mut h = HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly);
    h.set_realtime_backend(Box::new(backend));
    let listener = [20.0, 1.7, 40.0];
    let qs: Vec<SpatialQuery> = (0..8)
        .map(|i| SpatialQuery {
            source_position: [5.0 + 4.0 * i as f32, 1.7, 10.0 + 2.0 * i as f32],
            listener_position: listener,
            source_id: i,
        })
        .collect();
    let best = |f: &mut dyn FnMut()| {
        f(); // warm-up
        (0..5)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f32() * 1e3
            })
            .fold(f32::MAX, f32::min)
    };
    let backend = h.realtime_backend().unwrap();
    let raw = best(&mut || {
        let _ = std::hint::black_box(backend.query_spatial(&qs, &Opaque));
    });
    let batched = best(&mut || {
        let _ = std::hint::black_box(h.resolve_batch(&qs, &Opaque));
    });
    let serial = best(&mut || {
        for q in &qs {
            let _ = std::hint::black_box(h.resolve(q, &Opaque));
        }
    });
    println!(
        "8 emitters: raw query_spatial {raw:.1} ms, resolve_batch {batched:.1} ms, per-pair loop {serial:.1} ms (speedup {:.2}x, batch/raw {:.2})",
        serial / batched,
        batched / raw
    );
    let a = h.resolve_batch(&qs, &Opaque);
    for (q, r) in qs.iter().zip(&a) {
        assert_same(r.as_ref().unwrap(), &h.resolve(q, &Opaque).unwrap(), "large scene");
    }
    assert!(batched <= raw * 1.15 + 1.0, "batched must be within 15% of one query_spatial");
}
