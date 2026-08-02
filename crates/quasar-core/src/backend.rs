use crate::bands::Band8;
use crate::error::SpatialAudioError;
use crate::rays::{Ray, RayHit, RayInteractionContext};
use crate::scene::AcousticScene;

/// A spatial audio query for one source-listener pair.
#[derive(Clone, Debug)]
pub struct SpatialQuery {
    /// World-space position of the sound source.
    pub source_position: [f32; 3],
    /// World-space position of the listener.
    pub listener_position: [f32; 3],
    /// Identifier for the sound source.
    pub source_id: u32,
}

/// Result of a spatial audio query.
#[derive(Clone, Debug)]
pub struct SpatialQueryResult {
    /// Identifier of the source this result corresponds to.
    pub source_id: u32,
    /// Parameters for the direct (line-of-sight) path.
    pub direct_path: DirectPathResult,
    /// Early reflection paths (specular & diffuse).
    pub early_reflections: Vec<EarlyReflection>,
    /// Late reverberation estimate.
    pub late_reverb: LateReverbEstimate,
}

/// Direct path parameters between source and listener.
#[derive(Clone, Debug)]
pub struct DirectPathResult {
    /// Distance attenuation per band (linear gain).
    pub attenuation: Band8,
    /// Fractional delay in samples at the audio thread's sample rate.
    pub delay_samples: f32,
    /// Distance in world units.
    pub distance: f32,
    /// Whether the direct path is occluded.
    pub occluded: bool,
    /// Occlusion factor [0, 1] — 0 = fully occluded, 1 = clear line of sight.
    pub occlusion_factor: f32,
}

/// A single early reflection path.
#[derive(Clone, Debug)]
pub struct EarlyReflection {
    /// Direction from listener toward the reflection point.
    pub direction: [f32; 3],
    /// Delay in samples at the audio thread's sample rate.
    pub delay_samples: f32,
    /// Gain per band (linear).
    pub gain: Band8,
    /// Specular reflection order (0 = direct, 1 = first-order, etc.).
    pub order: u32,
}

/// Late reverberation estimate.
#[derive(Clone, Debug)]
pub struct LateReverbEstimate {
    /// RT60 per band (seconds).
    pub t60: Band8,
    /// Early / late split point (seconds).
    pub early_late_split_secs: f32,
    /// Late reverb loudness relative to direct (dB).
    pub late_loudness_db: f32,
}

/// Provides material acoustic properties for the compute backend.
///
/// Implemented by `AcousticMaterialRegistry` in `quasar-materials`.
pub trait MaterialProvider: Send + Sync {
    /// Evaluate the acoustic material properties at a ray hit point.
    ///
    /// Returns the per-band absorption coefficient(s) or similar acoustic parameter.
    fn evaluate_material(&self, handle: u32, context: &RayInteractionContext) -> Band8;
}

/// Hardware-agnostic spatial compute backend.
///
/// Implementors provide ray tracing, path generation, and reverb estimation.
pub trait IAcousticComputeBackend: Send + Sync {
    /// Query spatial parameters for multiple source-listener pairs.
    ///
    /// Called from the compute thread (15–30 Hz), never from the audio thread.
    fn query_spatial(
        &self,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Vec<SpatialQueryResult>;

    /// Whether this backend supports dynamic scene updates.
    fn supports_dynamic_geometry(&self) -> bool {
        false
    }

    /// Update the scene geometry. May trigger acceleration structure rebuilds.
    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError>;

    /// Trace a single ray through the scene. Returns all hits along the ray.
    fn trace_ray(&self, ray: &Ray) -> Vec<RayHit>;
}
