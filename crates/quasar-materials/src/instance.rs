use std::mem::size_of;

/// Unique identifier for a material evaluation model (e.g., PorousAbsorber, Resonator, Layered).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialModelId(pub u32);

/// Raw, typed or untyped parameter payload for a material instance.
///
/// Designed for direct blitting to GPU storage buffers or CPU cache lines.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct MaterialParameterBuffer {
    /// Shader/SIMD alignment-friendly byte array (e.g., std430 layout for GPU).
    pub data: Vec<u8>,
}

impl MaterialParameterBuffer {
    /// Create a new parameter buffer from a byte vector.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Create an empty parameter buffer.
    pub fn empty() -> Self {
        Self { data: Vec::new() }
    }

    /// Number of bytes in the parameter buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the parameter buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Interpret the buffer as a slice of a specific type `T`.
    ///
    /// Returns `None` if the buffer size is not a multiple of `size_of::<T>()`.
    pub fn as_slice<T: bytemuck::Pod>(&self) -> Option<&[T]> {
        let count = self.data.len() / size_of::<T>();
        if count == 0 || self.data.len() % size_of::<T>() != 0 {
            return None;
        }
        bytemuck::try_cast_slice(&self.data).ok()
    }

    /// Interpret the buffer as a single value of type `T`.
    ///
    /// Returns `None` if the buffer size does not match `size_of::<T>()`.
    pub fn as_value<T: bytemuck::Pod>(&self) -> Option<&T> {
        if self.data.len() != size_of::<T>() {
            return None;
        }
        bytemuck::try_from_bytes(&self.data).ok()
    }
}

/// A material instance couples a model ID with its parameter buffer.
#[derive(Clone, Debug)]
pub struct AcousticMaterialInstance {
    /// Identifier of the material model (evaluator).
    pub model_id: MaterialModelId,
    /// Parameter payload for this instance.
    pub parameters: MaterialParameterBuffer,
}

impl AcousticMaterialInstance {
    /// Create a new material instance.
    pub fn new(model_id: MaterialModelId, parameters: MaterialParameterBuffer) -> Self {
        Self {
            model_id,
            parameters,
        }
    }
}
