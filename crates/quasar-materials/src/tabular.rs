use quasar_core::bands::Band8;
use quasar_core::rays::RayInteractionContext;

use crate::evaluator::{AcousticResponse8Band, IAcousticMaterialEvaluator};
use crate::instance::{MaterialModelId, MaterialParameterBuffer};

/// Material model ID for tabular 8-band materials.
pub const TABULAR_MODEL_ID: MaterialModelId = MaterialModelId(1);

/// Parameter buffer type for tabular data: 3 × 8 f32 values (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TabularParams {
    absorption: [f32; 8],
    scattering: [f32; 8],
    transmission: [f32; 8],
}

/// Simple 8-band lookup table evaluator.
///
/// Parameter buffer layout: `absorption[8] | scattering[8] | transmission[8]` (24 f32s = 96 bytes).
pub struct Tabular8BandEvaluator;

impl Tabular8BandEvaluator {
    /// Create a new `Tabular8BandEvaluator`.
    pub fn new() -> Self {
        Self
    }

    /// Helper to create a parameter buffer from explicit 8-band values.
    pub fn create_params(
        absorption: Band8,
        scattering: Band8,
        transmission: Band8,
    ) -> MaterialParameterBuffer {
        let params = TabularParams {
            absorption: absorption.0,
            scattering: scattering.0,
            transmission: transmission.0,
        };
        MaterialParameterBuffer::new(bytemuck::bytes_of(&params).to_vec())
    }
}

impl IAcousticMaterialEvaluator for Tabular8BandEvaluator {
    fn model_id(&self) -> MaterialModelId {
        TABULAR_MODEL_ID
    }

    fn validate(&self, params: &MaterialParameterBuffer) -> Result<(), String> {
        let p = params.read_value::<TabularParams>().ok_or_else(|| {
            format!("tabular buffer must be exactly {} bytes, got {}", std::mem::size_of::<TabularParams>(), params.len())
        })?;
        if p.absorption.iter().chain(&p.scattering).chain(&p.transmission).any(|v| !v.is_finite()) {
            return Err("tabular buffer contains non-finite values".into());
        }
        Ok(())
    }

    fn evaluate(
        &self,
        params: &MaterialParameterBuffer,
        _context: &RayInteractionContext,
    ) -> AcousticResponse8Band {
        // Malformed buffer (rejected by `validate`): documented default, never a panic.
        let Some(tabular) = params.read_value::<TabularParams>() else {
            return AcousticResponse8Band::default();
        };

        AcousticResponse8Band {
            absorption: Band8(tabular.absorption),
            scattering: Band8(tabular.scattering),
            transmission: Band8(tabular.transmission),
        }
    }
}

/// Material model ID for angle-aware tabular 8-band materials (see [`Tabular8BandAngleEvaluator`]).
pub const TABULAR_ANGLE_MODEL_ID: MaterialModelId = MaterialModelId(4);

/// Buffer of [`Tabular8BandAngleEvaluator`]: the 96-byte tabular block followed by the
/// absorption measured at `table_angle_rad`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TabularAngleParams {
    absorption: [f32; 8],
    scattering: [f32; 8],
    transmission: [f32; 8],
    /// Absorption at `table_angle_rad`.
    absorption_at_angle: [f32; 8],
    /// Angle (radians from the normal) of the second table; must be in `(0.05, pi/2 - 0.05)`.
    table_angle_rad: f32,
    _pad: f32,
}

/// Tabular 8-band material with an optional second absorption table at one oblique angle.
///
/// Buffer layout (136 bytes): `absorption(0 deg)[8] | scattering[8] | transmission[8] |
/// absorption(table_angle)[8] | table_angle_rad | pad`. The 96-byte [`Tabular8BandEvaluator`]
/// (model id 1) is unchanged and stays angle independent.
///
/// Absorption at incidence angle `theta` (from the normal, folded into `[0, pi/2]`):
/// * `theta <= theta_t`: linear in `1 - cos(theta)` between `alpha(0)` and `alpha(theta_t)`.
/// * `theta > theta_t`: locally-reacting law with a real normalised impedance `zeta` fitted at
///   `theta_t`: `|R_t| = sqrt(1 - alpha_t)`, `zeta cos(theta_t) = (1 + |R_t|) / (1 - |R_t|)`,
///   `alpha(theta) = 1 - ((zeta cos(theta) - 1) / (zeta cos(theta) + 1))^2`, which is continuous
///   at `theta_t` and tends to 0 at grazing incidence.
///
/// Scattering and transmission are angle independent. At `theta = 0` the result equals the
/// normal-incidence table exactly.
pub struct Tabular8BandAngleEvaluator;

impl Tabular8BandAngleEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Build a parameter buffer.
    pub fn create_params(
        absorption: Band8,
        scattering: Band8,
        transmission: Band8,
        absorption_at_angle: Band8,
        table_angle_rad: f32,
    ) -> MaterialParameterBuffer {
        let p = TabularAngleParams {
            absorption: absorption.0,
            scattering: scattering.0,
            transmission: transmission.0,
            absorption_at_angle: absorption_at_angle.0,
            table_angle_rad,
            _pad: 0.0,
        };
        MaterialParameterBuffer::new(bytemuck::bytes_of(&p).to_vec())
    }

    /// Absorption at `theta_rad` for one band (see the type docs).
    pub fn absorption_at_angle(alpha0: f32, alpha_t: f32, theta_t: f32, theta_rad: f32) -> f32 {
        let pi = std::f32::consts::PI;
        let mut th = theta_rad.abs() % (2.0 * pi);
        if th > pi {
            th = 2.0 * pi - th;
        }
        if th > pi / 2.0 {
            th = pi - th;
        }
        let a0 = alpha0.clamp(0.0, 1.0);
        let at = alpha_t.clamp(0.0, 1.0);
        if th <= theta_t {
            let g = (1.0 - th.cos()) / (1.0 - theta_t.cos());
            return (a0 + (at - a0) * g).clamp(0.0, 1.0);
        }
        // Real impedance fitted at theta_t; at(1) would be total absorption (r = 0 -> zeta cos = 1).
        let r = (1.0 - at).sqrt();
        let zc_t = if r >= 1.0 - 1e-6 { 1e6 } else { (1.0 + r) / (1.0 - r) };
        let zeta = zc_t / theta_t.cos();
        let x = zeta * th.cos();
        let refl = (x - 1.0) / (x + 1.0);
        (1.0 - refl * refl).clamp(0.0, 1.0)
    }
}

impl IAcousticMaterialEvaluator for Tabular8BandAngleEvaluator {
    fn model_id(&self) -> MaterialModelId {
        TABULAR_ANGLE_MODEL_ID
    }

    fn validate(&self, params: &MaterialParameterBuffer) -> Result<(), String> {
        let p = params.read_value::<TabularAngleParams>().ok_or_else(|| {
            format!(
                "angle tabular buffer must be exactly {} bytes, got {}",
                std::mem::size_of::<TabularAngleParams>(),
                params.len()
            )
        })?;
        let all = p
            .absorption
            .iter()
            .chain(&p.scattering)
            .chain(&p.transmission)
            .chain(&p.absorption_at_angle);
        if all.clone().any(|v| !v.is_finite()) || !p.table_angle_rad.is_finite() {
            return Err("angle tabular buffer contains non-finite values".into());
        }
        if !(p.table_angle_rad > 0.05 && p.table_angle_rad < std::f32::consts::FRAC_PI_2 - 0.05) {
            return Err(format!("table angle must be in (0.05, pi/2 - 0.05) rad, got {}", p.table_angle_rad));
        }
        Ok(())
    }

    fn evaluate(&self, params: &MaterialParameterBuffer, context: &RayInteractionContext) -> AcousticResponse8Band {
        if self.validate(params).is_err() {
            return AcousticResponse8Band::default();
        }
        let Some(p) = params.read_value::<TabularAngleParams>() else {
            return AcousticResponse8Band::default();
        };
        let mut a = [0.0_f32; 8];
        for b in 0..8 {
            a[b] = Self::absorption_at_angle(
                p.absorption[b],
                p.absorption_at_angle[b],
                p.table_angle_rad,
                context.incident_angle_rad,
            );
        }
        AcousticResponse8Band {
            absorption: Band8(a),
            scattering: Band8(p.scattering),
            transmission: Band8(p.transmission),
        }
    }
}
