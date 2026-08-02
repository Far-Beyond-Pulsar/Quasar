use quasar_core::backend::{
    DirectPathResult, IAcousticComputeBackend, LateReverbEstimate, MaterialProvider, SpatialQuery,
    SpatialQueryResult,
};
use quasar_core::bands::Band8;
use quasar_core::error::SpatialAudioError;
use quasar_core::rays::{Ray, RayHit};
use quasar_core::scene::AcousticScene;

/// Stub backend for future DSP/NPU hardware acceleration.
///
/// This backend does NOT perform actual ray tracing. It returns worst-case
/// results (max distance, no occlusion) and serves as a placeholder for
/// dedicated hardware integration.
///
/// Thread-safe: uses only immutable self (no internal state).
pub struct HardwareAcceleratorStub {
    has_scene: bool,
}

impl HardwareAcceleratorStub {
    /// Create a new `HardwareAcceleratorStub` with no scene loaded.
    pub fn new() -> Self {
        Self { has_scene: false }
    }
}

impl IAcousticComputeBackend for HardwareAcceleratorStub {
    fn query_spatial(
        &self,
        queries: &[SpatialQuery],
        _materials: &dyn MaterialProvider,
    ) -> Vec<SpatialQueryResult> {
        queries
            .iter()
            .map(|q| {
                let dx = q.listener_position[0] - q.source_position[0];
                let dy = q.listener_position[1] - q.source_position[1];
                let dz = q.listener_position[2] - q.source_position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                SpatialQueryResult {
                    source_id: q.source_id,
                    direct_path: DirectPathResult {
                        attenuation: Band8::splat(1.0 / (1.0 + dist)),
                        delay_samples: dist * 48_000.0 / 343.0,
                        distance: dist,
                        occluded: false,
                        occlusion_factor: 1.0,
                    },
                    early_reflections: Vec::new(),
                    late_reverb: LateReverbEstimate {
                        t60: Band8::splat(0.5),
                        early_late_split_secs: 0.05,
                        late_loudness_db: -10.0,
                    },
                }
            })
            .collect()
    }

    fn supports_dynamic_geometry(&self) -> bool {
        false
    }

    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError> {
        self.has_scene = !scene.is_empty();
        if scene.is_empty() {
            return Err(SpatialAudioError::InvalidScene(
                "empty scene provided to stub backend".into(),
            ));
        }
        Ok(())
    }

    fn trace_ray(&self, _ray: &Ray) -> Vec<RayHit> {
        vec![]
    }
}
