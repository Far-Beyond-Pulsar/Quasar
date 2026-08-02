use crate::instance::{MaterialModelId, MaterialParameterBuffer};

/// A GPU-compatible material descriptor (mirror of the WGSL struct).
///
/// Used for building the material indirect buffer.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuMaterialDescriptor {
    /// Material model identifier.
    pub model_id: u32,
    /// Byte offset into the parameter storage buffer.
    pub param_offset: u32,
    /// Byte size of the parameter block.
    pub param_size: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

/// Describes the layout of material data on the GPU.
pub struct GpuMaterialLayout {
    /// All material parameter buffers concatenated into one byte array.
    pub parameter_storage: Vec<u8>,
    /// One `GpuMaterialDescriptor` per material instance.
    pub descriptors: Vec<GpuMaterialDescriptor>,
}

/// Alignment requirement for GPU storage buffer parameters (16 bytes — std430).
const GPU_ALIGNMENT: usize = 16;

impl GpuMaterialLayout {
    /// Build a GPU material layout from a slice of material instances.
    ///
    /// Each instance's parameter buffer is appended to the storage with alignment
    /// padding. The descriptors array maps instance index to
    /// `{model_id, param_offset, param_size}`.
    pub fn build(instances: &[MaterialInstanceInfo]) -> Self {
        let mut storage = Vec::new();
        let mut descriptors = Vec::with_capacity(instances.len());

        for info in instances {
            let param_offset = storage.len() as u32;
            let param_size = info.parameters.len() as u32;

            // Append parameter data
            storage.extend_from_slice(&info.parameters.data);

            // Pad to alignment
            let remainder = storage.len() % GPU_ALIGNMENT;
            if remainder != 0 {
                let padding = GPU_ALIGNMENT - remainder;
                storage.extend(std::iter::repeat(0u8).take(padding));
            }

            descriptors.push(GpuMaterialDescriptor {
                model_id: info.model_id.0,
                param_offset,
                param_size,
                _pad: 0,
            });
        }

        GpuMaterialLayout {
            parameter_storage: storage,
            descriptors,
        }
    }

    /// Get the total storage buffer size in bytes.
    pub fn storage_size(&self) -> usize {
        self.parameter_storage.len()
    }

    /// Get the total descriptor buffer size in bytes.
    pub fn descriptor_size(&self) -> usize {
        self.descriptors.len() * size_of::<GpuMaterialDescriptor>()
    }
}

/// Minimal info needed for GPU layout building.
pub struct MaterialInstanceInfo {
    /// Material model identifier.
    pub model_id: MaterialModelId,
    /// Parameter payload for the material instance.
    pub parameters: MaterialParameterBuffer,
}

/// Return the WGSL source snippet for the material evaluation dispatch function.
///
/// The returned code should be included in ray tracing shaders to enable
/// GPU-side material evaluation.
pub fn wgsl_material_eval_source() -> &'static str {
    include_str!("../shaders/material_eval.wgsl")
}
