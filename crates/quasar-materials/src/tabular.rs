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

    fn evaluate(
        &self,
        params: &MaterialParameterBuffer,
        _context: &RayInteractionContext,
    ) -> AcousticResponse8Band {
        let tabular: &TabularParams = params
            .as_value::<TabularParams>()
            .expect("Tabular8BandEvaluator: parameter buffer must be exactly 96 bytes (TabularParams)");

        AcousticResponse8Band {
            absorption: Band8(tabular.absorption),
            scattering: Band8(tabular.scattering),
            transmission: Band8(tabular.transmission),
        }
    }
}
