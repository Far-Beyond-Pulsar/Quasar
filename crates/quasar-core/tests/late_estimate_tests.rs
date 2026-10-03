//! Probe-derived late estimate (#64): split and level come from the probes (RIR when
//! baked, else the documented T60 / volume model); no constants in the sampler.

use quasar_core::backend::{MaterialProvider, SpatialQuery};
use quasar_core::bands::Band8;
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::rays::RayInteractionContext;
use quasar_core::reverb_model::{late_loudness_from_t60_volume, LATE_DB_MIN};

struct Mat;
impl MaterialProvider for Mat {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(0.1)
    }
}

/// RIR (energy per band): unit direct spike, then an exponential tail of total energy `late`.
fn rir(sr: u32, late_total: f32, split: f32) -> Vec<Band8> {
    let n = (sr as f32 * 1.0) as usize;
    let mut v = vec![Band8::zeros(); n];
    v[0] = Band8::splat(1.0);
    let s0 = (split * sr as f32) as usize;
    let count = (n - s0) as f32;
    for i in s0..n {
        v[i] = Band8::splat(late_total / count);
    }
    v
}

fn probe(pos: [f32; 3], split: f32, rir: Vec<Band8>, t60: f32) -> AcousticProbe {
    AcousticProbe { position: pos, rir_samples: rir, sample_rate: 48_000, t60: Band8::splat(t60), broadband_t60: t60, early_late_split_secs: split }
}

fn grid(make: impl Fn(usize) -> AcousticProbe) -> AcousticProbeGrid {
    let probes: Vec<AcousticProbe> = (0..8).map(|i| make(i)).collect();
    AcousticProbeGrid::new(probes, [0.0; 3], [10.0; 3], [2, 2, 2]).unwrap()
}

#[test]
fn late_level_from_t60_and_volume_follows_the_diffuse_field_model() {
    // 312.2 T / V: 2 s in 1000 m^3 -> -2.0 dB; doubling the volume costs 3 dB.
    let a = late_loudness_from_t60_volume(2.0, 1000.0);
    assert!((a - 10.0 * (312.2_f32 * 2.0 / 1000.0).log10()).abs() < 0.05, "{a}");
    assert!((late_loudness_from_t60_volume(2.0, 2000.0) - a + 3.01).abs() < 0.05);
    assert_eq!(late_loudness_from_t60_volume(f32::NAN, 100.0), LATE_DB_MIN);
    assert_eq!(late_loudness_from_t60_volume(1.0, 0.0), LATE_DB_MIN);
}

#[test]
fn grid_interpolates_split_and_rir_level() {
    // x = 0 probes: split 0.04 s, late energy 0.1 (-10 dB); x = 10 probes: 0.08 s, 0.01 (-20 dB).
    let g = grid(|i| {
        let right = i % 2 == 1;
        let (split, late) = if right { (0.08, 0.01) } else { (0.04, 0.1) };
        probe([0.0; 3], split, rir(48_000, late, split), 1.0)
    });
    let mid = g.sample(&[5.0, 5.0, 5.0]).unwrap();
    assert!((mid.early_late_split_secs - 0.06).abs() < 1e-5);
    let l = mid.late_loudness_db.expect("RIR present");
    assert!((l - -15.0).abs() < 0.1, "{l}");
    let left = g.sample(&[0.0, 5.0, 5.0]).unwrap();
    assert!((left.late_loudness_db.unwrap() + 10.0).abs() < 0.1);
    assert!((left.early_late_split_secs - 0.04).abs() < 1e-5);
}

#[test]
fn baked_resolve_uses_probe_data_not_constants() {
    let q = SpatialQuery { source_position: [2.0, 5.0, 5.0], listener_position: [5.0, 5.0, 5.0], source_id: 0 };
    // With RIRs: level and split come from them.
    let g = grid(|_| probe([0.0; 3], 0.03, rir(48_000, 0.1, 0.03), 1.0));
    let mut h = HybridProbeSampler::new(HybridSamplingStrategy::BakedOnly);
    h.set_probe_grid(g);
    let r = h.resolve(&q, &Mat).unwrap().late_reverb;
    assert!((r.late_loudness_db + 10.0).abs() < 0.1, "{}", r.late_loudness_db);
    assert!((r.early_late_split_secs - 0.03).abs() < 1e-6);

    // Without RIRs: the T60 / volume model (grid spans 10 x 10 x 10 = 1000 m^3).
    let g = grid(|_| probe([0.0; 3], 0.07, Vec::new(), 2.0));
    let mut h = HybridProbeSampler::new(HybridSamplingStrategy::BakedOnly);
    h.set_probe_grid(g);
    let r = h.resolve(&q, &Mat).unwrap().late_reverb;
    assert!((r.late_loudness_db - late_loudness_from_t60_volume(2.0, 1000.0)).abs() < 1e-4);
    assert!((r.early_late_split_secs - 0.07).abs() < 1e-6);
    // Longer T60 in the same volume = louder late field.
    let g = grid(|_| probe([0.0; 3], 0.07, Vec::new(), 4.0));
    let mut h2 = HybridProbeSampler::new(HybridSamplingStrategy::BakedOnly);
    h2.set_probe_grid(g);
    assert!(h2.resolve(&q, &Mat).unwrap().late_reverb.late_loudness_db > r.late_loudness_db + 2.9);
}
