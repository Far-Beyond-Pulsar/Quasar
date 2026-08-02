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
    for i in 0..8 {
        assert!(
            r.absorption.0[i] < 0.3,
            "high-resistivity concrete-like material should have low absorption at band {i}: {}",
            r.absorption.0[i]
        );
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

    let ctx = RayInteractionContext::default();
    let r = reg.evaluate(0, &ctx).unwrap();
    assert!((r.absorption.0[0] - 0.5).abs() < 1e-6);
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
