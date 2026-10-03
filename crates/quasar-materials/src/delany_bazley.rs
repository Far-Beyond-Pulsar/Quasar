use std::sync::atomic::{AtomicBool, Ordering};

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
const RHO_0: f64 = 1.204;

/// Speed of sound at 20°C (m/s).
const C_0: f64 = 343.0;

/// Characteristic impedance of air (rayls).
const Z_0: f64 = RHO_0 * C_0; // ≈ 413.0

/// Validity range of the Delany-Bazley regression in the dimensionless variable
/// `X = rho0 * f / sigma` (`f` in Hz, `sigma` = flow resistivity in SI rayls/m = Pa s/m^2):
/// `0.01 <= X <= 1.0` (the original fit; equivalently `f/sigma_cgs` of roughly 0.01..1 in
/// kHz / cgs rayls units). Outside it the power laws extrapolate silently; this evaluator
/// **clamps `X` to the range inside the empirical `Zc` / `k` power laws** (the frequency
/// dependence `omega` of `k` and of the layer response is kept) and warns once per process
/// (see [`PorousDelanyBazleyEvaluator::x_in_validity_range`]).
pub const DELANY_BAZLEY_X_RANGE: (f32, f32) = (0.01, 1.0);

/// Largest incidence angle used for the oblique-incidence law (89 degrees): at exactly 90 degrees
/// `cos(theta) = 0` and the reflection coefficient is exactly -1 (zero absorption), which would
/// make absorption discontinuous in the tracer's grazing hits.
pub const DELANY_BAZLEY_MAX_ANGLE_RAD: f32 = 89.0 * std::f32::consts::PI / 180.0;

static WARNED_RANGE: AtomicBool = AtomicBool::new(false);

/// Minimal complex number (f64) for the layer impedance chain.
#[derive(Clone, Copy)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Delany-Bazley porous absorber material evaluator.
///
/// Parameter buffer: `[flow_resistivity: f32, thickness_m: f32]` (8 bytes).
///
/// Implements the empirical Delany-Bazley model for fibrous porous absorbers (rigid backing)
/// treated as **locally reacting**: the normal-incidence surface impedance
/// `Zs = -j Zc cot(k d)` is used at every angle, with the oblique-incidence reflection
/// coefficient
///
/// `R(theta) = (Zs cos(theta) - Z0) / (Zs cos(theta) + Z0)`, `alpha(theta) = 1 - |R(theta)|^2`,
///
/// where `theta` is `RayInteractionContext::incident_angle_rad` (angle from the surface normal,
/// folded into `[0, pi/2]` and clamped to [`DELANY_BAZLEY_MAX_ANGLE_RAD`] = 89 degrees).
/// Flow resistivity is in Rayls/m (typically 1000 - 100 000), thickness in meters. Validity
/// range: see [`DELANY_BAZLEY_X_RANGE`].
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

    /// Whether `X = rho0 f / sigma` lies in the Delany-Bazley validity range
    /// [`DELANY_BAZLEY_X_RANGE`].
    pub fn x_in_validity_range(freq: f32, flow_resistivity: f32) -> bool {
        if freq <= 0.0 || flow_resistivity <= 0.0 {
            return false;
        }
        let x = RHO_0 as f32 * freq / flow_resistivity;
        x >= DELANY_BAZLEY_X_RANGE.0 && x <= DELANY_BAZLEY_X_RANGE.1
    }

    /// Normal-incidence surface impedance `Zs` (rayls, complex as `(re, im)`) of a rigidly
    /// backed layer, or `None` for degenerate input.
    fn surface_impedance(freq: f64, sigma: f64, d: f64) -> Option<C64> {
        // X clamped to the validity range inside the power laws.
        let e = (RHO_0 * freq / sigma).clamp(DELANY_BAZLEY_X_RANGE.0 as f64, DELANY_BAZLEY_X_RANGE.1 as f64);

        // Characteristic impedance Zc and propagation constant k (e^{+j w t} convention).
        let zc = C64::new(Z_0 * (1.0 + 0.0571 * e.powf(-0.754)), -Z_0 * 0.087 * e.powf(-0.732));
        let w_c = 2.0 * std::f64::consts::PI * freq / C_0;
        let kd = C64::new(w_c * (1.0 + 0.0978 * e.powf(-0.700)) * d, -w_c * 0.189 * e.powf(-0.595) * d);

        // cot(a + jb) = cos/sin with
        // cos(a+jb) = cos a cosh b - j sin a sinh b ; sin(a+jb) = sin a cosh b + j cos a sinh b.
        let cos = C64::new(kd.re.cos() * kd.im.cosh(), -(kd.re.sin() * kd.im.sinh()));
        let sin = C64::new(kd.re.sin() * kd.im.cosh(), kd.re.cos() * kd.im.sinh());
        if sin.norm_sqr() < 1e-24 || !cos.re.is_finite() || !sin.re.is_finite() {
            return None;
        }
        let cot = cos.div(sin);
        // Zs = -j * Zc * cot(kd)
        let p = zc.mul(cot);
        Some(C64::new(p.im, -p.re))
    }

    /// Absorption coefficient at normal incidence (see [`Self::absorption_at_freq_angle`]).
    pub fn absorption_at_freq(freq: f32, flow_resistivity: f32, thickness_m: f32) -> f32 {
        Self::absorption_at_freq_angle(freq, flow_resistivity, thickness_m, 0.0)
    }

    /// Absorption coefficient at incidence angle `theta_rad` (from the normal) using the
    /// locally-reacting oblique-incidence reflection coefficient (see the type docs).
    ///
    /// # Parameters
    /// - `freq` — frequency in Hz
    /// - `flow_resistivity` — flow resistivity in Rayls/m
    /// - `thickness_m` — material thickness in meters
    /// - `theta_rad` — angle of incidence from the surface normal (folded into `[0, pi/2]`,
    ///   clamped to 89 degrees)
    pub fn absorption_at_freq_angle(freq: f32, flow_resistivity: f32, thickness_m: f32, theta_rad: f32) -> f32 {
        if !(freq > 0.0 && flow_resistivity > 0.0 && thickness_m > 0.0) || !theta_rad.is_finite() {
            return 0.0;
        }
        let mut theta = (theta_rad as f64).abs() % (2.0 * std::f64::consts::PI);
        if theta > std::f64::consts::PI {
            theta = 2.0 * std::f64::consts::PI - theta;
        }
        if theta > std::f64::consts::FRAC_PI_2 {
            theta = std::f64::consts::PI - theta;
        }
        let theta = theta.min(DELANY_BAZLEY_MAX_ANGLE_RAD as f64);

        let Some(zs) = Self::surface_impedance(freq as f64, flow_resistivity as f64, thickness_m as f64) else {
            return 1.0; // near singularity of cot: fully absorbed (as before)
        };
        // R = (Zs cos - Z0) / (Zs cos + Z0)
        let c = theta.cos();
        let zc = C64::new(zs.re * c, zs.im * c);
        let den = C64::new(zc.re + Z_0, zc.im);
        if den.norm_sqr() < 1e-24 {
            return 1.0;
        }
        let r = C64::new(zc.re - Z_0, zc.im).div(den);
        let alpha = 1.0 - r.norm_sqr();
        if alpha.is_finite() {
            alpha.clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    }
}

impl IAcousticMaterialEvaluator for PorousDelanyBazleyEvaluator {
    fn model_id(&self) -> MaterialModelId {
        DELANY_BAZLEY_MODEL_ID
    }

    fn validate(&self, params: &MaterialParameterBuffer) -> Result<(), String> {
        let p = params.read_value::<DelanyBazleyParams>().ok_or_else(|| {
            format!("Delany-Bazley buffer must be exactly 8 bytes, got {}", params.len())
        })?;
        if !(p.flow_resistivity.is_finite() && p.flow_resistivity > 0.0) {
            return Err(format!("flow resistivity must be finite and > 0, got {}", p.flow_resistivity));
        }
        if !(p.thickness_m.is_finite() && p.thickness_m > 0.0) {
            return Err(format!("thickness must be finite and > 0, got {}", p.thickness_m));
        }
        Ok(())
    }

    fn evaluate(
        &self,
        params: &MaterialParameterBuffer,
        context: &RayInteractionContext,
    ) -> AcousticResponse8Band {
        // Malformed buffer (rejected by `validate`): documented default, never a panic.
        let Some(p) = params.read_value::<DelanyBazleyParams>() else {
            return AcousticResponse8Band::default();
        };

        let mut absorption = [0.0_f32; 8];
        let mut out_of_range = false;
        for (i, &freq) in FREQ_BAND_CENTRES.iter().enumerate() {
            out_of_range |= !Self::x_in_validity_range(freq, p.flow_resistivity);
            absorption[i] =
                Self::absorption_at_freq_angle(freq, p.flow_resistivity, p.thickness_m, context.incident_angle_rad);
        }
        if out_of_range && !WARNED_RANGE.swap(true, Ordering::Relaxed) {
            eprintln!(
                "quasar-materials: Delany-Bazley used outside its validity range ({} <= rho0*f/sigma <= {}) \
                 for flow resistivity {} rayls/m; X is clamped (warned once)",
                DELANY_BAZLEY_X_RANGE.0, DELANY_BAZLEY_X_RANGE.1, p.flow_resistivity
            );
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
