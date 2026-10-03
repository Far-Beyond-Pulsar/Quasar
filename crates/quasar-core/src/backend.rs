use crate::bands::Band8;
use crate::distance::DistanceModel;
use crate::error::SpatialAudioError;
use crate::rays::{Ray, RayHit, RayInteractionContext};
use crate::scene::AcousticScene;

/// Speed of sound in air (m/s) used when a backend has no configured value.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Sample rate (Hz) assumed until the engine calls
/// [`IAcousticComputeBackend::set_sample_rate`] / `HybridProbeSampler::set_sample_rate`.
pub const DEFAULT_SAMPLE_RATE: f32 = 48_000.0;

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
    /// Broadband occlusion factor [0, 1] — 0 = fully occluded, 1 = clear line of sight
    /// (the mean of [`Self::occlusion`]).
    pub occlusion_factor: f32,
    /// Per-band linear amplitude occlusion [0, 1] (1 = clear, 0 = blocked). Already
    /// multiplied into [`Self::attenuation`]; kept separately so callers can inspect
    /// how much of the direct-path loss is due to geometry.
    pub occlusion: Band8,
}

/// A single early reflection path.
#[derive(Clone, Debug)]
pub struct EarlyReflection {
    /// Arrival direction: the unit vector, in WORLD space, from the listener
    /// toward the **last** reflection point of the path (the bounce nearest the
    /// listener; for a first-order path, the single bounce point). It is the
    /// direction the sound seems to come from, so it is already listener-relative
    /// in origin; rotating it by the listener's heading gives the panning angle.
    pub direction: [f32; 3],
    /// Delay in samples at the audio thread's sample rate: the TOTAL emission ->
    /// listener path length `* fs / c` (same clock as the direct path's
    /// `distance * fs / c`).
    pub delay_samples: f32,
    /// Gain per band (linear amplitude), relative to the un-attenuated dry
    /// signal: the product of the surface reflection coefficients along the path
    /// times the distance law and air absorption of the TOTAL path length. It does
    /// NOT include (or depend on) the direct path's attenuation or occlusion.
    pub gain: Band8,
    /// Specular reflection order (number of bounces; 1 = first-order, etc.).
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

    /// Per-band linear **amplitude** transmission gain of the surface (what is left
    /// of a wave after passing *through* it): 0 = opaque, 1 = transparent.
    ///
    /// Distinct from absorption (energy lost at a reflection). The backend
    /// multiplies it over every surface a direct-path ray crosses. The default is
    /// opaque, so a provider that only knows absorption blocks sound rather than
    /// leaking it; `AcousticMaterialRegistry` returns the material model's own value.
    fn evaluate_transmission(&self, _handle: u32, _context: &RayInteractionContext) -> Band8 {
        Band8::zeros()
    }
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

    /// Tell the backend the audio device sample rate so every `delay_samples` it
    /// returns is expressed in samples at that rate (default: 48 kHz).
    ///
    /// Called by the engine when the backend is installed (and whenever the
    /// engine rate changes), never from the audio thread. Backends that do not
    /// produce delays may ignore it.
    fn set_sample_rate(&mut self, _sample_rate: f32) {}

    /// Install the shared distance-attenuation model used for the direct path
    /// (default: [`DistanceModel::default`], inverse distance with a 1 m
    /// reference). Backends that do not model distance may ignore it.
    fn set_distance_model(&mut self, _model: DistanceModel) {}

    /// Whether this backend supports dynamic scene updates.
    fn supports_dynamic_geometry(&self) -> bool {
        false
    }

    /// Update the scene geometry. May trigger acceleration structure rebuilds.
    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError>;

    /// Trace a single ray through the scene. Returns all hits along the ray.
    fn trace_ray(&self, ray: &Ray) -> Vec<RayHit>;
}
