use std::f32::consts::PI;

use quasar_core::bands::{Band8, FREQ_BAND_CENTRES};
use quasar_core::rays::RayInteractionContext;

use crate::evaluator::{AcousticResponse8Band, IAcousticMaterialEvaluator};
use crate::instance::{MaterialModelId, MaterialParameterBuffer};

/// Material model ID for the Delany-Bazley porous absorber.
pub const DELANY_BAZLEY_MODEL_ID: MaterialModelId = MaterialModelId(2);

/// Parameter buffer for the Delany-Bazley model: `[flow_resistivity, thickness_m]` (8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DelanyBazleyParams {
    flow_resistivity: f32,
    thickness_m: f32,
}

/// Air density at 20°C (kg/m³).
const RHO_0: f32 = 1.204;

/// Speed of sound at 20°C (m/s).
const C_0: f32 = 343.0;

/// Characteristic impedance of air (rayls).
const Z_0: f32 = RHO_0 * C_0; // ≈ 413.0

/// Delany-Bazley porous absorber material evaluator.
///
/// Parameter buffer: `[flow_resistivity: f32, thickness_m: f32]` (8 bytes).
///
/// Implements the empirical Delany-Bazley model for fibrous porous absorbers.
/// Flow resistivity is in Rayls/m (typically 1000 – 100 000), thickness in meters.
pub struct PorousDelanyBazleyEvaluator;

impl PorousDelanyBazleyEvaluator {
    /// Create a new `PorousDelanyBazleyEvaluator`.
    pub fn new() -> Self {
        Self
    }

    /// Create a parameter buffer from flow resistivity and thickness.
    pub fn create_params(flow_resistivity: f32, thickness_m: f32) -> MaterialParameterBuffer {
        let params = DelanyBazleyParams {
            flow_resistivity,
            thickness_m,
        };
        MaterialParameterBuffer::new(bytemuck::bytes_of(&params).to_vec())
    }

    /// Compute absorption coefficient for a single frequency using the Delany-Bazley model.
    ///
    /// # Parameters
    /// - `freq` — frequency in Hz
    /// - `flow_resistivity` — flow resistivity in Rayls/m
    /// - `thickness_m` — material thickness in meters
    pub fn absorption_at_freq(freq: f32, flow_resistivity: f32, thickness_m: f32) -> f32 {
        if freq <= 0.0 || flow_resistivity <= 0.0 || thickness_m <= 0.0 {
            return 0.0;
        }

        // Dimensionless parameter E = ρ₀ · f / R_s
        let e = RHO_0 * freq / flow_resistivity;

        // Characteristic impedance Zc (complex)
        let zc_real = Z_0 * (1.0 + 0.0571 * e.powf(-0.754));
        let zc_imag = -Z_0 * 0.087 * e.powf(-0.732);

        // Propagation constant k (complex)
        let omega = 2.0 * PI * freq;
        let omega_over_c0 = omega / C_0;
        let k_real = omega_over_c0 * (1.0 + 0.0978 * e.powf(-0.700));
        let k_imag = -omega_over_c0 * 0.189 * e.powf(-0.595);

        // Surface impedance Zs = -j · Zc · cot(k · d)
        // cot(x) = cos(x) / sin(x); handle the complex number arithmetic manually.
        let kd_real = k_real * thickness_m;
        let kd_imag = k_imag * thickness_m;

        // Compute cot(kd) for complex kd = kd_real + j * kd_imag
        // cot(z) = cos(z) / sin(z)
        // cos(a + jb) = cos(a)cosh(b) - j·sin(a)sinh(b)
        // sin(a + jb) = sin(a)cosh(b) + j·cos(a)sinh(b)
        let cos_kd_real = kd_real.cos() * kd_imag.cosh();
        let cos_kd_imag = -(kd_real.sin() * kd_imag.sinh());
        let sin_kd_real = kd_real.sin() * kd_imag.cosh();
        let sin_kd_imag = kd_real.cos() * kd_imag.sinh();

        // cot(z) = (cos_real + j·cos_imag) / (sin_real + j·sin_imag)
        let denom = sin_kd_real * sin_kd_real + sin_kd_imag * sin_kd_imag;
        if denom.abs() < 1e-12 {
            return 1.0; // near singularity — fully absorbed
        }
        let cot_real = (cos_kd_real * sin_kd_real + cos_kd_imag * sin_kd_imag) / denom;
        let cot_imag = (cos_kd_imag * sin_kd_real - cos_kd_real * sin_kd_imag) / denom;

        // Zs = -j · Zc · cot(kd)
        // -j · (zc_real + j·zc_imag) · (cot_real + j·cot_imag)
        // = -j · [ (zc_real·cot_real - zc_imag·cot_imag) + j·(zc_real·cot_imag + zc_imag·cot_real) ]
        let a = zc_real * cot_real - zc_imag * cot_imag;
        let b = zc_real * cot_imag + zc_imag * cot_real;
        // -j · (a + jb) = -j·a + b  =>  real = b, imag = -a
        let zs_real = b;
        let zs_imag = -a;

        // Reflection coefficient R = (Zs - Z₀) / (Zs + Z₀)
        let r_num_real = zs_real - Z_0;
        let r_num_imag = zs_imag;
        let r_den_real = zs_real + Z_0;
        let r_den_imag = zs_imag;
        let r_den_mag2 = r_den_real * r_den_real + r_den_imag * r_den_imag;
        if r_den_mag2.abs() < 1e-12 {
            return 1.0;
        }
        let r_real = (r_num_real * r_den_real + r_num_imag * r_den_imag) / r_den_mag2;
        let r_imag = (r_num_imag * r_den_real - r_num_real * r_den_imag) / r_den_mag2;

        // Absorption α = 1 - |R|²
        let r_mag2 = r_real * r_real + r_imag * r_imag;
        (1.0 - r_mag2).clamp(0.0, 1.0)
    }
}

impl IAcousticMaterialEvaluator for PorousDelanyBazleyEvaluator {
    fn model_id(&self) -> MaterialModelId {
        DELANY_BAZLEY_MODEL_ID
    }

    fn evaluate(
        &self,
        params: &MaterialParameterBuffer,
        _context: &RayInteractionContext,
    ) -> AcousticResponse8Band {
        let p: &DelanyBazleyParams = params
            .as_value::<DelanyBazleyParams>()
            .expect("PorousDelanyBazleyEvaluator: parameter buffer must be exactly 8 bytes");

        let mut absorption = [0.0_f32; 8];
        for (i, &freq) in FREQ_BAND_CENTRES.iter().enumerate() {
            absorption[i] = Self::absorption_at_freq(freq, p.flow_resistivity, p.thickness_m);
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
    fn test_absorption_at_freq_rockwool() {
        // Rockwool: R_s ≈ 10000 Rayls/m, d = 0.05 m
        let alpha = PorousDelanyBazleyEvaluator::absorption_at_freq(500.0, 10000.0, 0.05);
        assert!(alpha >= 0.0 && alpha <= 1.0, "absorption out of range: {alpha}");
        // A reasonable rockwool sample at 500 Hz should show moderate absorption
        assert!(alpha > 0.2, "expected moderate absorption, got {alpha}");
    }

    #[test]
    fn test_absorption_at_freq_high_resistivity() {
        // Very dense material: R_s = 100000, thin panel
        let alpha = PorousDelanyBazleyEvaluator::absorption_at_freq(125.0, 100000.0, 0.01);
        assert!(alpha >= 0.0 && alpha <= 1.0);
    }

    #[test]
    fn test_absorption_at_freq_zero_thickness() {
        let alpha = PorousDelanyBazleyEvaluator::absorption_at_freq(500.0, 10000.0, 0.0);
        assert_eq!(alpha, 0.0);
    }
}
