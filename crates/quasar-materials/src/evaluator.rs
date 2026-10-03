use quasar_core::bands::Band8;
use quasar_core::rays::RayInteractionContext;
use crate::instance::{MaterialModelId, MaterialParameterBuffer};

/// The acoustic response of a surface across 8 octave bands.
#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct AcousticResponse8Band {
    /// Absorption coefficient [0.0 - 1.0] — 1.0 = fully absorbed
    pub absorption: Band8,
    /// Scattering coefficient [0.0 - 1.0] — 0.0 = purely specular, 1.0 = purely diffuse
    pub scattering: Band8,
    /// Transmission gain (linear **amplitude**, not energy) — 0.0 = fully blocked, 1.0 = fully
    /// transmitted. Applied once per surface crossed (no thickness model).
    pub transmission: Band8,
}

impl AcousticResponse8Band {
    /// Create a new `AcousticResponse8Band` from per-band absorption, scattering, and transmission.
    pub fn new(absorption: Band8, scattering: Band8, transmission: Band8) -> Self {
        Self {
            absorption,
            scattering,
            transmission,
        }
    }

    /// Default response for a non-interacting surface (fully reflective, specular).
    pub fn default() -> Self {
        Self {
            absorption: Band8::zeros(),
            scattering: Band8::zeros(),
            transmission: Band8::zeros(),
        }
    }

    /// Default response for air (fully transmissive).
    pub fn air() -> Self {
        Self {
            absorption: Band8::zeros(),
            scattering: Band8::zeros(),
            transmission: Band8::splat(1.0),
        }
    }

    /// Dead surface (fully absorptive).
    pub fn void() -> Self {
        Self {
            absorption: Band8::splat(1.0),
            scattering: Band8::zeros(),
            transmission: Band8::zeros(),
        }
    }
}

/// A material evaluator computes the acoustic response of a material model
/// given its parameter buffer and the interaction context (incident angle, etc.).
pub trait IAcousticMaterialEvaluator: Send + Sync {
    /// Unique identifier for this material model.
    fn model_id(&self) -> MaterialModelId;

    /// Check a parameter buffer (size, finite values, sane ranges). Called by the registry at
    /// `add_instance` / `update_instance` so `evaluate` cannot meet a malformed buffer. Default: accept.
    fn validate(&self, _params: &MaterialParameterBuffer) -> Result<(), String> {
        Ok(())
    }

    /// Evaluate the acoustic response for the given parameters and interaction context.
    ///
    /// Must not panic: a buffer that fails [`Self::validate`] yields the documented default
    /// response [`AcousticResponse8Band::default`] (fully reflective, opaque).
    fn evaluate(
        &self,
        params: &MaterialParameterBuffer,
        context: &RayInteractionContext,
    ) -> AcousticResponse8Band;
}
