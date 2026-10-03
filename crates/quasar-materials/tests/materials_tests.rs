use quasar_core::bands::Band8;
use quasar_core::rays::RayInteractionContext;
use quasar_materials::delany_bazley::PorousDelanyBazleyEvaluator;
use quasar_materials::evaluator::{AcousticResponse8Band, IAcousticMaterialEvaluator};
use quasar_materials::gpu_pipeline::{GpuMaterialDescriptor, GpuMaterialLayout, MaterialInstanceInfo};
use quasar_materials::instance::{AcousticMaterialInstance, MaterialModelId, MaterialParameterBuffer};
use quasar_materials::registry::AcousticMaterialRegistry;
use quasar_materials::resonant_panel::ResonantPanelEvaluator;
use quasar_materials::tabular::Tabular8BandEvaluator;

// ── tabular_evaluator_absorption ──────────────────────────────────────

#[test]
fn tabular_evaluator_absorption() {
    let absorption = Band8::new([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.6, 0.4]);
    let params = Tabular8BandEvaluator::create_params(absorption, Band8::zeros(), Band8::zeros());
    let evaluator = Tabular8BandEvaluator::new();
    let ctx = RayInteractionContext::default();
    let response = evaluator.evaluate(&params, &ctx);
    for i in 0..8 {
        assert!((response.absorption.0[i] - absorption.0[i]).abs() < 1e-6);
    }
}

// ── tabular_create_params_roundtrip ───────────────────────────────────

#[test]
fn tabular_create_params_roundtrip() {
    let absorption = Band8::new([0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.55, 0.30]);
    let scattering = Band8::splat(0.1);
    let transmission = Band8::splat(0.0);

    let params = Tabular8BandEvaluator::create_params(absorption, scattering, transmission);
    let evaluator = Tabular8BandEvaluator::new();
    let ctx = RayInteractionContext::default();
    let r = evaluator.evaluate(&params, &ctx);

    for i in 0..8 {
        assert!((r.absorption.0[i] - absorption.0[i]).abs() < 1e-6);
        assert!((r.scattering.0[i] - scattering.0[i]).abs() < 1e-6);
        assert!((r.transmission.0[i] - transmission.0[i]).abs() < 1e-6);
    }
}

// ── delany_bazley_absorption_range ────────────────────────────────────

#[test]
fn delany_bazley_absorption_range() {
    let params = PorousDelanyBazleyEvaluator::create_params(10000.0, 0.05);
    let evaluator = PorousDelanyBazleyEvaluator::new();
    let ctx = RayInteractionContext::default();
    let r = evaluator.evaluate(&params, &ctx);
    for i in 0..8 {
        assert!(
            r.absorption.0[i] >= 0.0 && r.absorption.0[i] <= 1.0,
            "band {i} absorption {} out of [0,1]",
            r.absorption.0[i]
        );
    }
}

// ── delany_bazley_increasing_with_frequency ───────────────────────────

#[test]
fn delany_bazley_increasing_with_frequency() {
    let params = PorousDelanyBazleyEvaluator::create_params(20000.0, 0.05);
    let ctx = RayInteractionContext::default();
    let evaluator = PorousDelanyBazleyEvaluator::new();
    let r = evaluator.evaluate(&params, &ctx);

    let centres = quasar_core::bands::FREQ_BAND_CENTRES;
    for i in 1..8 {
        if r.absorption.0[i] > 0.0 && r.absorption.0[i - 1] > 0.0 {
            assert!(
                r.absorption.0[i] >= r.absorption.0[i - 1] - 0.15,
                "absorption decreased from band {} ({:.1} Hz) to band {} ({:.1} Hz): {:.3} -> {:.3}",
                i - 1, centres[i - 1], i, centres[i], r.absorption.0[i - 1], r.absorption.0[i]
            );
        }
    }
}

// ── delany_bazley_high_flow_resistivity ───────────────────────────────

#[test]
fn delany_bazley_high_flow_resistivity() {
    let params = PorousDelanyBazleyEvaluator::create_params(100000.0, 0.01);
    let evaluator = PorousDelanyBazleyEvaluator::new();
    let ctx = RayInteractionContext::default();
    let r = evaluator.evaluate(&params, &ctx);
    // #118: the original assertion (every band < 0.3, "concrete-like") was wrong, not the model.
    // sigma = 1e5 rayls/m, d = 1 cm is a thin dense felt, not concrete: its flow resistance
    // sigma*d = 1000 rayls is ~2.4 Z0, near the optimum for a thin resistive layer, and from
    // 250 Hz up X = rho0 f / sigma is inside the Delany-Bazley validity range (0.01..1), where the
    // model (checked against an independent complex-arithmetic evaluation below) gives
    // alpha ~ 0.48 at 2 kHz. What must hold physically: a 1 cm layer is acoustically tiny at low
    // frequencies (< 0.3 up to 250 Hz) and follows by the independent evaluation (a 1 cm layer near its quarter-wave frequency can legitimately absorb strongly).
    for i in 0..3 {
        assert!(r.absorption.0[i] < 0.3, "thin layer must absorb little at low band {i}: {}", r.absorption.0[i]);
    }
    for i in 0..8 {
        let a = r.absorption.0[i];
        let f = quasar_core::bands::FREQ_BAND_CENTRES[i];
        assert!((0.0..=1.0).contains(&a));
        let reference = db_reference(f as f64, 1e5, 0.01, 0.0).clamp(0.0, 1.0);
        assert!((a as f64 - reference).abs() < 2e-3, "band {i}: {a} vs independent {reference}");
    }
}

// ── resonant_panel_peak_near_resonance ────────────────────────────────

#[test]
fn resonant_panel_peak_near_resonance() {
    // m = 2.3 kg/m², d = 0.1 m gives f0 ≈ 60 / sqrt(2.3 * 0.1) ≈ 125 Hz
    let alpha_125 = ResonantPanelEvaluator::absorption_at_freq(125.0, 2.3, 0.1);
    let alpha_62 = ResonantPanelEvaluator::absorption_at_freq(62.5, 2.3, 0.1);
    let alpha_250 = ResonantPanelEvaluator::absorption_at_freq(250.0, 2.3, 0.1);

    assert!(
        alpha_125 > alpha_62,
        "absorption should peak near resonance (125 Hz) vs 62.5 Hz"
    );
    assert!(
        alpha_125 > alpha_250,
        "absorption should peak near resonance (125 Hz) vs 250 Hz"
    );
    assert!(alpha_125 > 0.3);
}

// ── resonant_panel_absorption_zero_at_extremes ────────────────────────

#[test]
fn resonant_panel_absorption_zero_at_extremes() {
    let alpha_low = ResonantPanelEvaluator::absorption_at_freq(10.0, 5.0, 0.2);
    assert!(
        alpha_low < 0.05,
        "very low frequency should have near-zero absorption"
    );
    let alpha_high = ResonantPanelEvaluator::absorption_at_freq(20000.0, 5.0, 0.2);
    assert!(
        alpha_high < 0.05,
        "very high frequency should have near-zero absorption"
    );
}

// ── material_registry_register_and_evaluate ───────────────────────────

#[test]
fn material_registry_register_and_evaluate() {
    let reg = AcousticMaterialRegistry::new();
    reg.register_evaluator(Box::new(Tabular8BandEvaluator::new()));

    let params = Tabular8BandEvaluator::create_params(
        Band8::new([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        Band8::zeros(),
        Band8::zeros(),
    );
    let instance = AcousticMaterialInstance::new(
        quasar_materials::tabular::TABULAR_MODEL_ID,
        params,
    );
    let handle = reg.add_instance(instance);

    let ctx = RayInteractionContext::default();
    let response = reg.evaluate(handle, &ctx).unwrap();
    assert!((response.absorption.0[0] - 0.8).abs() < 1e-6);
    assert!((response.absorption.0[7] - 0.1).abs() < 1e-6);
}

// ── material_registry_hot_swap ────────────────────────────────────────

#[test]
fn material_registry_hot_swap() {
    let reg = AcousticMaterialRegistry::new();
    reg.register_evaluator(Box::new(Tabular8BandEvaluator::new()));

    let params1 = Tabular8BandEvaluator::create_params(
        Band8::splat(0.2),
        Band8::zeros(),
        Band8::zeros(),
    );
    let handle = reg.add_instance(AcousticMaterialInstance::new(
        quasar_materials::tabular::TABULAR_MODEL_ID,
        params1,
    ));

    let ctx = RayInteractionContext::default();
    let r1 = reg.evaluate(handle, &ctx).unwrap();
    assert!((r1.absorption.0[0] - 0.2).abs() < 1e-6);

    let params2 = Tabular8BandEvaluator::create_params(
        Band8::splat(0.9),
        Band8::zeros(),
        Band8::zeros(),
    );
    reg.update_instance(handle, params2).unwrap();

    let r2 = reg.evaluate(handle, &ctx).unwrap();
    assert!((r2.absorption.0[0] - 0.9).abs() < 1e-6);
}

// ── material_registry_invalid_handle ──────────────────────────────────

#[test]
fn material_registry_invalid_handle() {
    let reg = AcousticMaterialRegistry::new();
    let ctx = RayInteractionContext::default();
    let result = reg.evaluate(999, &ctx);
    assert!(result.is_err());
    match result {
        Err(quasar_core::error::SpatialAudioError::Material(_)) => {}
        _ => panic!("expected Material error"),
    }

    let update_result = reg.update_instance(999, MaterialParameterBuffer::empty());
    assert!(update_result.is_err());
}

// ── material_registry_remove_and_reindex ──────────────────────────────

#[test]
fn material_registry_remove_and_reindex() {
    let reg = AcousticMaterialRegistry::new();
    reg.register_evaluator(Box::new(Tabular8BandEvaluator::new()));

    let h0 = reg.add_instance(AcousticMaterialInstance::new(
        quasar_materials::tabular::TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.1), Band8::zeros(), Band8::zeros()),
    ));
    let _h1 = reg.add_instance(AcousticMaterialInstance::new(
        quasar_materials::tabular::TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.5), Band8::zeros(), Band8::zeros()),
    ));

    assert_eq!(reg.instance_count(), 2);

    assert!(reg.remove_instance(h0));
    assert_eq!(reg.instance_count(), 1);

    // #69: handles are stable. The old test evaluated handle 0 and expected the former last
    // instance (swap_remove moved it there); now h1 keeps meaning, h0 is dead.
    let ctx = RayInteractionContext::default();
    let r = reg.evaluate(_h1, &ctx).unwrap();
    assert!((r.absorption.0[0] - 0.5).abs() < 1e-6);
    assert!(reg.evaluate(h0, &ctx).is_err());
}

// ── gpu_material_layout_build ─────────────────────────────────────────

#[test]
fn gpu_material_layout_build() {
    let instances = vec![
        MaterialInstanceInfo {
            model_id: MaterialModelId(1),
            parameters: MaterialParameterBuffer::new(vec![1u8, 2, 3, 4]),
        },
        MaterialInstanceInfo {
            model_id: MaterialModelId(2),
            parameters: MaterialParameterBuffer::new(vec![5u8, 6, 7, 8]),
        },
    ];

    let layout = GpuMaterialLayout::build(&instances);
    assert_eq!(layout.descriptors.len(), 2);

    // First descriptor: offset 0
    assert_eq!(layout.descriptors[0].model_id, 1);
    assert_eq!(layout.descriptors[0].param_offset, 0);
    assert_eq!(layout.descriptors[0].param_size, 4);

    // Second descriptor starts after first + padding
    let offset1 = layout.descriptors[1].param_offset as usize;
    assert!(offset1 >= 4);
    assert_eq!(layout.descriptors[1].model_id, 2);
    assert_eq!(layout.descriptors[1].param_size, 4);

    assert!(layout.storage_size() > 0);
    assert!(layout.descriptor_size() >= 2 * std::mem::size_of::<GpuMaterialDescriptor>());
}

// ── acoustic_response8band_construction ───────────────────────────────

#[test]
fn acoustic_response8band_construction() {
    let default = AcousticResponse8Band::default();
    for i in 0..8 {
        assert_eq!(default.absorption.0[i], 0.0);
        assert_eq!(default.scattering.0[i], 0.0);
        assert_eq!(default.transmission.0[i], 0.0);
    }

    let air = AcousticResponse8Band::air();
    for i in 0..8 {
        assert_eq!(air.transmission.0[i], 1.0);
    }

    let void = AcousticResponse8Band::void();
    for i in 0..8 {
        assert_eq!(void.absorption.0[i], 1.0);
    }
}

// ── #69 robustness: validation, stable handles, explicit default ──────

use quasar_materials::delany_bazley::DELANY_BAZLEY_MODEL_ID;
use quasar_materials::resonant_panel::RESONANT_PANEL_MODEL_ID;
use quasar_materials::tabular::TABULAR_MODEL_ID;
use quasar_core::backend::MaterialProvider;

fn full_registry() -> AcousticMaterialRegistry {
    let reg = AcousticMaterialRegistry::new();
    reg.register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    reg.register_evaluator(Box::new(PorousDelanyBazleyEvaluator::new()));
    reg.register_evaluator(Box::new(ResonantPanelEvaluator::new()));
    reg
}

fn tab(a: f32) -> MaterialParameterBuffer {
    Tabular8BandEvaluator::create_params(Band8::splat(a), Band8::zeros(), Band8::zeros())
}

#[test]
fn malformed_buffers_are_rejected_at_creation_and_update() {
    let reg = full_registry();
    let bad_len = MaterialParameterBuffer::new(vec![0u8; 95]);
    assert!(reg.try_add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, bad_len.clone())).is_err());
    assert!(reg.try_add_instance(AcousticMaterialInstance::new(DELANY_BAZLEY_MODEL_ID, MaterialParameterBuffer::empty())).is_err());
    assert!(reg.try_add_instance(AcousticMaterialInstance::new(RESONANT_PANEL_MODEL_ID, MaterialParameterBuffer::new(vec![0; 9]))).is_err());
    // NaN / non-physical values.
    assert!(reg.try_add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, tab(f32::NAN))).is_err());
    assert!(reg.try_add_instance(AcousticMaterialInstance::new(DELANY_BAZLEY_MODEL_ID, PorousDelanyBazleyEvaluator::create_params(-1.0, 0.05))).is_err());
    assert!(reg.try_add_instance(AcousticMaterialInstance::new(RESONANT_PANEL_MODEL_ID, ResonantPanelEvaluator::create_params(2.0, 0.0))).is_err());
    assert_eq!(reg.instance_count(), 0, "rejected instances must not be stored");

    let h = reg.add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, tab(0.3)));
    assert!(reg.update_instance(h, bad_len).is_err());
    let r = reg.evaluate(h, &RayInteractionContext::default()).unwrap();
    assert!((r.absorption.0[0] - 0.3).abs() < 1e-6, "failed update must keep the old parameters");
}

#[test]
#[should_panic(expected = "invalid parameters")]
fn add_instance_panics_with_a_clear_message_on_bad_params() {
    let reg = full_registry();
    reg.add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, MaterialParameterBuffer::new(vec![1, 2, 3])));
}

#[test]
fn evaluators_never_panic_on_malformed_buffers() {
    let ctx = RayInteractionContext::default();
    let evs: [Box<dyn IAcousticMaterialEvaluator>; 3] = [
        Box::new(Tabular8BandEvaluator::new()),
        Box::new(PorousDelanyBazleyEvaluator::new()),
        Box::new(ResonantPanelEvaluator::new()),
    ];
    for e in evs.iter() {
        for len in [0usize, 1, 7, 9, 95, 97, 200] {
            let buf = MaterialParameterBuffer::new(vec![0xAB; len]);
            if e.validate(&buf).is_ok() {
                continue;
            }
            assert_eq!(e.evaluate(&buf, &ctx), AcousticResponse8Band::default(), "len {len}");
        }
    }
}

#[test]
fn evaluators_read_unaligned_buffers() {
    // A buffer whose data starts at an odd address inside a larger allocation can't be cast in
    // place; read_value copies, so any alignment works.
    let params = tab(0.25);
    let mut raw = vec![0u8; params.len() + 1];
    raw[1..].copy_from_slice(&params.data);
    let shifted = MaterialParameterBuffer::new(raw[1..].to_vec());
    let r = Tabular8BandEvaluator::new().evaluate(&shifted, &RayInteractionContext::default());
    assert!((r.absorption.0[3] - 0.25).abs() < 1e-6);
}

#[test]
fn bad_handle_and_missing_evaluator_use_the_explicit_default_and_are_counted() {
    let reg = AcousticMaterialRegistry::new();
    let ctx = RayInteractionContext::default();
    assert_eq!(reg.error_count(), 0);
    // Invalid handle -> default 0.9 absorption, opaque; observable.
    for _ in 0..5 {
        assert_eq!(reg.evaluate_material(42, &ctx), Band8::splat(0.9));
    }
    assert_eq!(reg.evaluate_transmission(42, &ctx), Band8::zeros());
    assert_eq!(reg.error_count(), 6);
    // Missing evaluator.
    let h = reg.add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, tab(0.2)));
    assert_eq!(reg.evaluate_material(h, &ctx), Band8::splat(0.9));
    assert_eq!(reg.error_count(), 7);
    // Registering the evaluator fixes it (buffer is validated lazily and accepted).
    reg.register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    assert_eq!(reg.evaluate_material(h, &ctx), Band8::splat(0.2));
    assert_eq!(reg.error_count(), 7);
    // The default is configurable.
    reg.set_default_response(AcousticResponse8Band::new(Band8::splat(0.05), Band8::zeros(), Band8::splat(0.5)));
    assert_eq!(reg.evaluate_material(99, &ctx), Band8::splat(0.05));
    assert_eq!(reg.evaluate_transmission(99, &ctx), Band8::splat(0.5));
}

#[test]
fn lazily_validated_bad_buffer_falls_back_to_the_default() {
    let reg = AcousticMaterialRegistry::new();
    let h = reg.add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, MaterialParameterBuffer::new(vec![0; 5])));
    reg.register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let ctx = RayInteractionContext::default();
    assert!(reg.evaluate(h, &ctx).is_err());
    assert_eq!(reg.evaluate_material(h, &ctx), Band8::splat(0.9));
    assert_eq!(reg.error_count(), 2);
}

#[test]
fn removing_an_instance_never_changes_another_handle() {
    let reg = full_registry();
    let ctx = RayInteractionContext::default();
    let hs: Vec<u32> = (0..5).map(|i| reg.add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, tab(0.1 * (i + 1) as f32)))).collect();
    assert!(reg.remove_instance(hs[1]));
    assert!(!reg.remove_instance(hs[1]), "double remove is a no-op");
    for (i, &h) in hs.iter().enumerate() {
        if i == 1 {
            assert!(reg.evaluate(h, &ctx).is_err());
            assert!(reg.get_instance(h).is_none());
            assert!(reg.update_instance(h, tab(0.5)).is_err());
        } else {
            let r = reg.evaluate(h, &ctx).unwrap();
            assert!((r.absorption.0[0] - 0.1 * (i + 1) as f32).abs() < 1e-6, "handle {h} changed meaning");
        }
    }
    // Slot reuse: the new instance gets a distinct handle; the stale one stays dead.
    let h_new = reg.add_instance(AcousticMaterialInstance::new(TABULAR_MODEL_ID, tab(0.77)));
    assert_ne!(h_new, hs[1]);
    assert!((reg.evaluate(h_new, &ctx).unwrap().absorption.0[0] - 0.77).abs() < 1e-6);
    assert!(reg.evaluate(hs[1], &ctx).is_err(), "stale handle must not alias the new instance");
    assert_eq!(reg.instance_count(), 5);
    assert_eq!(reg.total_parameter_bytes(), 5 * 96);
    // Removing the LAST instance (the old swap_remove hazard) leaves the others intact.
    assert!(reg.remove_instance(hs[4]));
    assert!((reg.evaluate(hs[0], &ctx).unwrap().absorption.0[0] - 0.1).abs() < 1e-6);
}

// ── #67 angle-dependent absorption ────────────────────────────────────

#[derive(Clone, Copy)]
struct Cx(f64, f64);
impl Cx {
    fn add(self, o: Cx) -> Cx { Cx(self.0 + o.0, self.1 + o.1) }
    fn sub(self, o: Cx) -> Cx { Cx(self.0 - o.0, self.1 - o.1) }
    fn mul(self, o: Cx) -> Cx { Cx(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0) }
    fn div(self, o: Cx) -> Cx {
        let d = o.0 * o.0 + o.1 * o.1;
        Cx((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn exp(self) -> Cx { let m = self.0.exp(); Cx(m * self.1.cos(), m * self.1.sin()) }
    fn norm2(self) -> f64 { self.0 * self.0 + self.1 * self.1 }
}

/// Independent evaluation: Zs = Zc coth(j k d) (exponential form), R = (Zs cos - Z0)/(Zs cos + Z0).
/// Written from the published Delany-Bazley regression, X clamped to [0.01, 1].
fn db_reference(f: f64, sigma: f64, d: f64, theta: f64) -> f64 {
    let (rho, c) = (1.204, 343.0);
    let z0 = rho * c;
    let x = (rho * f / sigma).clamp(0.01, 1.0);
    let zc = Cx(z0 * (1.0 + 0.0571 * x.powf(-0.754)), -z0 * 0.087 * x.powf(-0.732));
    let w = 2.0 * std::f64::consts::PI * f / c;
    let k = Cx(w * (1.0 + 0.0978 * x.powf(-0.700)), -w * 0.189 * x.powf(-0.595));
    let jkd = Cx(0.0, 1.0).mul(Cx(k.0 * d, k.1 * d));
    let (ep, em) = (jkd.exp(), Cx(-jkd.0, -jkd.1).exp());
    let coth = ep.add(em).div(ep.sub(em));
    let zs = zc.mul(coth);
    let t = theta.min(89f64.to_radians());
    let zcos = Cx(zs.0 * t.cos(), zs.1 * t.cos());
    let r = zcos.sub(Cx(z0, 0.0)).div(zcos.add(Cx(z0, 0.0)));
    1.0 - r.norm2()
}

fn ctx_at(theta: f32) -> RayInteractionContext {
    RayInteractionContext { incident_angle_rad: theta, ..RayInteractionContext::default() }
}

#[test]
fn delany_bazley_matches_independent_oblique_incidence_formula() {
    let ev = PorousDelanyBazleyEvaluator::new();
    for &(sigma, d) in &[(10_000.0f32, 0.05f32), (20_000.0, 0.03), (5_000.0, 0.1), (100_000.0, 0.01)] {
        let params = PorousDelanyBazleyEvaluator::create_params(sigma, d);
        for deg in [0.0f32, 20.0, 45.0, 70.0, 85.0, 89.0, 89.9] {
            let r = ev.evaluate(&params, &ctx_at(deg.to_radians()));
            for (i, &f) in quasar_core::bands::FREQ_BAND_CENTRES.iter().enumerate() {
                let want = db_reference(f as f64, sigma as f64, d as f64, (deg as f64).to_radians()).clamp(0.0, 1.0);
                assert!((r.absorption.0[i] as f64 - want).abs() < 2e-3, "sigma {sigma} d {d} {deg} deg band {i}: {} vs {want}", r.absorption.0[i]);
            }
        }
    }
}

#[test]
fn delany_bazley_absorption_depends_on_angle_and_is_clamped_at_grazing() {
    let ev = PorousDelanyBazleyEvaluator::new();
    let params = PorousDelanyBazleyEvaluator::create_params(10_000.0, 0.05);
    let a0 = ev.evaluate(&params, &ctx_at(0.0)).absorption;
    let a60 = ev.evaluate(&params, &ctx_at(60f32.to_radians())).absorption;
    let a89 = ev.evaluate(&params, &ctx_at(89f32.to_radians())).absorption;
    let a90 = ev.evaluate(&params, &ctx_at(90f32.to_radians())).absorption;
    assert!((0..8).any(|i| (a0.0[i] - a60.0[i]).abs() > 0.02), "angle must matter: {a0:?} vs {a60:?}");
    for i in 0..8 {
        assert!((a89.0[i] - a90.0[i]).abs() < 1e-6, "angles above 89 deg are clamped");
        assert!(a89.0[i] < 0.4, "grazing absorption must be small: {}", a89.0[i]);
    }
    // Back-face hits (angle > 90 deg) fold onto the front side.
    let back = ev.evaluate(&params, &ctx_at(120f32.to_radians())).absorption;
    for i in 0..8 {
        assert!((back.0[i] - a60.0[i]).abs() < 1e-5);
    }
    // The legacy helper is normal incidence.
    let f = PorousDelanyBazleyEvaluator::absorption_at_freq(1000.0, 10_000.0, 0.05);
    assert!((f - a0.0[4]).abs() < 1e-6);
}

#[test]
fn tabular_96_byte_model_is_angle_independent_and_unchanged() {
    let a = Band8::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let params = Tabular8BandEvaluator::create_params(a, Band8::zeros(), Band8::zeros());
    assert_eq!(params.len(), 96);
    let ev = Tabular8BandEvaluator::new();
    for deg in [0.0f32, 45.0, 80.0] {
        assert_eq!(ev.evaluate(&params, &ctx_at(deg.to_radians())).absorption, a);
    }
}

#[test]
fn tabular_angle_model_interpolates_continuously_from_the_normal_table() {
    use quasar_materials::tabular::{Tabular8BandAngleEvaluator, TABULAR_ANGLE_MODEL_ID};
    let a0 = Band8::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let a60 = Band8::new([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    let tt = 60f32.to_radians();
    let params = Tabular8BandAngleEvaluator::create_params(a0, Band8::splat(0.1), Band8::splat(0.2), a60, tt);
    let ev = Tabular8BandAngleEvaluator::new();
    assert_eq!(ev.model_id(), TABULAR_ANGLE_MODEL_ID);
    assert!(ev.validate(&params).is_ok());
    // Exact at the two tabulated angles.
    let r0 = ev.evaluate(&params, &ctx_at(0.0));
    let rt = ev.evaluate(&params, &ctx_at(tt));
    for i in 0..8 {
        assert!((r0.absorption.0[i] - a0.0[i]).abs() < 1e-6);
        assert!((rt.absorption.0[i] - a60.0[i]).abs() < 1e-5);
        assert_eq!(r0.scattering.0[i], 0.1);
        assert_eq!(r0.transmission.0[i], 0.2);
    }
    // Continuous (no jumps) over a dense sweep, bounded, and ~0 at grazing.
    let mut prev = ev.evaluate(&params, &ctx_at(0.0)).absorption;
    let mut deg = 0.1f32;
    while deg <= 85.0 {
        let cur = ev.evaluate(&params, &ctx_at(deg.to_radians())).absorption;
        for i in 0..8 {
            assert!((cur.0[i] - prev.0[i]).abs() < 0.1, "jump at {deg} deg band {i}");
            assert!((0.0..=1.0).contains(&cur.0[i]));
        }
        prev = cur;
        deg += 0.1;
    }
    let graze = ev.evaluate(&params, &ctx_at(90f32.to_radians())).absorption;
    for i in 0..8 {
        assert!(graze.0[i] < 0.01, "grazing absorption {}", graze.0[i]);
    }
    // A material that follows the locally-reacting real-impedance law is reproduced beyond theta_t.
    let zeta = 3.0f32;
    let alpha = |th: f32| { let x = zeta * th.cos(); 1.0 - ((x - 1.0) / (x + 1.0)).powi(2) };
    let p = Tabular8BandAngleEvaluator::create_params(Band8::splat(alpha(0.0)), Band8::zeros(), Band8::zeros(), Band8::splat(alpha(tt)), tt);
    let r = ev.evaluate(&p, &ctx_at(80f32.to_radians()));
    assert!((r.absorption.0[0] - alpha(80f32.to_radians())).abs() < 1e-4);
}

#[test]
fn tabular_angle_model_validates_buffers() {
    use quasar_materials::tabular::Tabular8BandAngleEvaluator;
    let ev = Tabular8BandAngleEvaluator::new();
    assert!(ev.validate(&MaterialParameterBuffer::new(vec![0; 96])).is_err());
    let bad_angle = Tabular8BandAngleEvaluator::create_params(Band8::splat(0.1), Band8::zeros(), Band8::zeros(), Band8::splat(0.2), 0.0);
    assert!(ev.validate(&bad_angle).is_err());
    assert_eq!(ev.evaluate(&bad_angle, &ctx_at(0.3)), AcousticResponse8Band::default());
    let reg = AcousticMaterialRegistry::new();
    reg.register_evaluator(Box::new(ev));
    assert!(reg
        .try_add_instance(AcousticMaterialInstance::new(quasar_materials::tabular::TABULAR_ANGLE_MODEL_ID, bad_angle))
        .is_err());
}

// ── #118 / #70 (first bullet): Delany-Bazley validity range ───────────

#[test]
fn delany_bazley_exposes_and_clamps_its_validity_range() {
    use quasar_materials::delany_bazley::DELANY_BAZLEY_X_RANGE;
    assert_eq!(DELANY_BAZLEY_X_RANGE, (0.01, 1.0));
    // X = 1.204 f / sigma: 1 kHz at 10 000 rayls/m -> 0.12 (inside); 62.5 Hz -> 0.0075 (below);
    // 8 kHz at 5000 rayls/m -> 1.93 (above).
    assert!(PorousDelanyBazleyEvaluator::x_in_validity_range(1000.0, 10_000.0));
    assert!(!PorousDelanyBazleyEvaluator::x_in_validity_range(62.5, 10_000.0));
    assert!(!PorousDelanyBazleyEvaluator::x_in_validity_range(8000.0, 5_000.0));
    // Out-of-range input stays finite and bounded (X is clamped, not extrapolated).
    for &(f, s) in &[(62.5f32, 1e6f32), (8000.0, 500.0), (20.0, 1e7)] {
        let a = PorousDelanyBazleyEvaluator::absorption_at_freq(f, s, 0.05);
        assert!(a.is_finite() && (0.0..=1.0).contains(&a), "{f} Hz, {s}: {a}");
    }
}
