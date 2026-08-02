use std::f32::consts::PI;

use quasar_core::bands::{Band8, FREQ_BAND_CENTRES};
use quasar_core::rays::RayInteractionContext;

use crate::evaluator::{AcousticResponse8Band, IAcousticMaterialEvaluator};
use crate::instance::{MaterialModelId, MaterialParameterBuffer};

/// Material model ID for the resonant panel absorber.
pub const RESONANT_PANEL_MODEL_ID: MaterialModelId = MaterialModelId(3);

/// Parameter buffer for the resonant panel model: `[panel_mass_kgm2, cavity_depth_m]` (8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ResonantPanelParams {
    panel_mass_kgm2: f32,
    cavity_depth_m: f32,
}

/// Resonant panel absorber material evaluator.
///
/// Models a mass-spring system where the panel mass and air cavity create a resonant absorber.
///
/// Parameter buffer: `[panel_mass_kgm2: f32, cavity_depth_m: f32]` (8 bytes).
///
/// - `panel_mass_kgm2` — surface density in kg/m² (typically 1–20)
/// - `cavity_depth_m` — air gap behind the panel in meters (typically 0.02–0.5)
pub struct ResonantPanelEvaluator;

impl ResonantPanelEvaluator {
    /// Create a new `ResonantPanelEvaluator`.
    pub fn new() -> Self {
        Self
    }

    /// Create a parameter buffer from panel mass and cavity depth.
    pub fn create_params(panel_mass_kgm2: f32, cavity_depth_m: f32) -> MaterialParameterBuffer {
        let params = ResonantPanelParams {
            panel_mass_kgm2,
            cavity_depth_m,
        };
        MaterialParameterBuffer::new(bytemuck::bytes_of(&params).to_vec())
    }

    /// Compute absorption at a single frequency using a resonant panel absorber model.
    ///
    /// The model uses:
    /// - Resonant frequency: `f₀ ≈ 60 / sqrt(m · d)` (approximate empirical formula)
    /// - A Q-factor relating to panel damping
    /// - A Lorentzian absorption peak centered at `f₀`
    pub fn absorption_at_freq(freq: f32, panel_mass: f32, cavity_depth_m: f32) -> f32 {
        if freq <= 0.0 || panel_mass <= 0.0 || cavity_depth_m <= 0.0 {
            return 0.0;
        }

        // Empirical resonant frequency for a panel absorber:
        //   f₀ ≈ (c₀ / (2π)) * sqrt(ρ₀ / (m · d))
        // With ρ₀ = 1.204, c₀ = 343, this gives roughly 60 / sqrt(m·d).
        let f0 = (C_0 / (2.0 * PI)) * (RHO_0 / (panel_mass * cavity_depth_m)).sqrt();

        if f0 <= 0.0 {
            return 0.0;
        }

        // Q-factor: higher mass and deeper cavity give a sharper resonance.
        // Typical Q ranges from 2 to 20 for panel absorbers.
        let q = 2.0 + 4.0 * (panel_mass / 5.0).min(1.0) + 4.0 * (cavity_depth_m / 0.2).min(1.0);

        // Normalised frequency ratio
        let eta = freq / f0;

        // Lorentzian absorption profile
        // α_max ≈ 0.85 – 0.99 at resonance; we use 0.95 as peak absorption.
        let alpha_max = 0.95_f32;
        let alpha = alpha_max / (1.0 + q * q * (eta - 1.0 / eta).powi(2));

        alpha.clamp(0.0, 1.0)
    }
}

/// Air density at 20°C (kg/m³).
const RHO_0: f32 = 1.204;

/// Speed of sound at 20°C (m/s).
const C_0: f32 = 343.0;

impl IAcousticMaterialEvaluator for ResonantPanelEvaluator {
    fn model_id(&self) -> MaterialModelId {
        RESONANT_PANEL_MODEL_ID
    }

    fn evaluate(
        &self,
        params: &MaterialParameterBuffer,
        _context: &RayInteractionContext,
    ) -> AcousticResponse8Band {
        let p: &ResonantPanelParams = params
            .as_value::<ResonantPanelParams>()
            .expect("ResonantPanelEvaluator: parameter buffer must be exactly 8 bytes");

        let mut absorption = [0.0_f32; 8];
        for (i, &freq) in FREQ_BAND_CENTRES.iter().enumerate() {
            absorption[i] = Self::absorption_at_freq(freq, p.panel_mass_kgm2, p.cavity_depth_m);
        }

        AcousticResponse8Band {
            absorption: Band8(absorption),
            scattering: Band8::zeros(),
            transmission: Band8::zeros(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonant_peak_near_predicted() {
        // Panel: m = 2.3 kg/m², d = 0.1 m => f0 ≈ 60 / sqrt(2.3 * 0.1) ≈ 125 Hz
        // This places resonance near the 125 Hz octave band centre
        let alpha_125 = ResonantPanelEvaluator::absorption_at_freq(125.0, 2.3, 0.1);
        let alpha_250 = ResonantPanelEvaluator::absorption_at_freq(250.0, 2.3, 0.1);
        // The peak should be higher at 125 Hz than at 250 Hz
        assert!(
            alpha_125 > alpha_250,
            "expected absorption to be higher at 125 Hz than 250 Hz"
        );
        assert!(alpha_125 > 0.3, "expected significant absorption near resonance");
    }

    #[test]
    fn test_absorption_range() {
        for freq in &FREQ_BAND_CENTRES {
            let alpha = ResonantPanelEvaluator::absorption_at_freq(*freq, 8.0, 0.15);
            assert!(
                (0.0..=1.0).contains(&alpha),
                "absorption out of range at {freq} Hz: {alpha}"
            );
        }
    }

    #[test]
    fn test_zero_params() {
        assert_eq!(ResonantPanelEvaluator::absorption_at_freq(500.0, 0.0, 0.1), 0.0);
        assert_eq!(ResonantPanelEvaluator::absorption_at_freq(500.0, 5.0, 0.0), 0.0);
    }
}
