use crate::backend::{
    DirectPathResult, IAcousticComputeBackend, LateReverbEstimate, MaterialProvider, SpatialQuery,
    SpatialQueryResult,
};
use crate::error::SpatialAudioError;
use crate::probe_grid::AcousticProbeGrid;

/// Strategy for resolving spatial audio queries.
///
/// Controls which data source(s) the hybrid sampler uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridSamplingStrategy {
    /// Only use pre-baked probe grids. Fastest. Suitable for static environments.
    BakedOnly,
    /// Only use real-time ray tracing. Highest fidelity. Suitable for dynamic scenes.
    RealTimeOnly,
    /// Use baked data for late reverb and real-time for early reflections + direct path.
    HybridBlend,
}

/// The hybrid sampler decides which data source to use for spatial audio queries.
///
/// Supports fallback between baked probe grids and real-time ray tracing backends.
pub struct HybridProbeSampler {
    strategy: HybridSamplingStrategy,
    probe_grid: Option<AcousticProbeGrid>,
    realtime_backend: Option<Box<dyn IAcousticComputeBackend>>,
}

impl HybridProbeSampler {
    /// Create a new hybrid sampler with the given strategy.
    pub fn new(strategy: HybridSamplingStrategy) -> Self {
        Self {
            strategy,
            probe_grid: None,
            realtime_backend: None,
        }
    }

    /// Set the baked probe grid data.
    pub fn set_probe_grid(&mut self, grid: AcousticProbeGrid) {
        self.probe_grid = Some(grid);
    }

    /// Set the real-time compute backend.
    pub fn set_realtime_backend(&mut self, backend: Box<dyn IAcousticComputeBackend>) {
        self.realtime_backend = Some(backend);
    }

    /// Get a reference to the probe grid, if set.
    pub fn probe_grid(&self) -> Option<&AcousticProbeGrid> {
        self.probe_grid.as_ref()
    }

    /// Get a reference to the real-time backend, if set.
    pub fn realtime_backend(&self) -> Option<&dyn IAcousticComputeBackend> {
        self.realtime_backend.as_deref()
    }

    /// Get a mutable reference to the real-time backend, if set.
    pub fn realtime_backend_mut(&mut self) -> Option<&mut Box<dyn IAcousticComputeBackend>> {
        self.realtime_backend.as_mut()
    }

    /// Set the sampling strategy.
    pub fn set_strategy(&mut self, strategy: HybridSamplingStrategy) {
        self.strategy = strategy;
    }

    /// Get the current strategy.
    pub fn strategy(&self) -> HybridSamplingStrategy {
        self.strategy
    }

    /// Resolve spatial parameters for one source-listener pair.
    ///
    /// Called from the compute thread (15–30 Hz).
    pub fn resolve(
        &self,
        query: &SpatialQuery,
        materials: &dyn MaterialProvider,
    ) -> Result<SpatialQueryResult, SpatialAudioError> {
        match self.strategy {
            HybridSamplingStrategy::BakedOnly => {
                let grid = self
                    .probe_grid
                    .as_ref()
                    .ok_or_else(|| SpatialAudioError::ProbeGrid("no probe grid configured for BakedOnly strategy".into()))?;

                // Sample the grid at the listener position to get reverb info.
                let sample = grid
                    .sample(&query.listener_position)
                    .ok_or_else(|| SpatialAudioError::ProbeGrid("listener position is outside the probe grid".into()))?;

                let dx = query.source_position[0] - query.listener_position[0];
                let dy = query.source_position[1] - query.listener_position[1];
                let dz = query.source_position[2] - query.listener_position[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                // Simple inverse-distance attenuation (clamped to avoid divide-by-zero).
                let atten = if distance > 1e-6 {
                    1.0 / distance
                } else {
                    1.0
                };
                let attenuations = crate::bands::Band8::splat(atten.min(1.0));

                Ok(SpatialQueryResult {
                    source_id: query.source_id,
                    direct_path: DirectPathResult {
                        attenuation: attenuations,
                        delay_samples: distance * 0.0029, // ~343 m/s → ms, then to fractional samples
                        distance,
                        occluded: false,
                        occlusion_factor: 1.0,
                    },
                    early_reflections: Vec::new(),
                    late_reverb: LateReverbEstimate {
                        t60: sample.t60,
                        early_late_split_secs: 0.05,
                        late_loudness_db: -10.0,
                    },
                })
            }
            HybridSamplingStrategy::RealTimeOnly => {
                let backend = self
                    .realtime_backend
                    .as_ref()
                    .ok_or_else(|| SpatialAudioError::Backend("no real-time backend configured for RealTimeOnly strategy".into()))?;

                let results = backend.query_spatial(&[query.clone()], materials);
                results
                    .into_iter()
                    .next()
                    .ok_or_else(|| SpatialAudioError::Backend("real-time backend returned no results".into()))
            }
            HybridSamplingStrategy::HybridBlend => {
                let backend = self
                    .realtime_backend
                    .as_ref()
                    .ok_or_else(|| SpatialAudioError::Backend("no real-time backend configured for HybridBlend strategy".into()))?;

                let grid = self
                    .probe_grid
                    .as_ref()
                    .ok_or_else(|| SpatialAudioError::ProbeGrid("no probe grid configured for HybridBlend strategy".into()))?;

                // Get real-time result for direct + early reflections.
                let mut result = backend
                    .query_spatial(&[query.clone()], materials)
                    .into_iter()
                    .next()
                    .ok_or_else(|| SpatialAudioError::Backend("real-time backend returned no results".into()))?;

                // Overlay baked late reverb from the probe grid.
                let sample = grid
                    .sample(&query.listener_position)
                    .ok_or_else(|| SpatialAudioError::ProbeGrid("listener position is outside the probe grid".into()))?;

                result.late_reverb = LateReverbEstimate {
                    t60: sample.t60,
                    early_late_split_secs: 0.05,
                    late_loudness_db: -10.0,
                };

                Ok(result)
            }
        }
    }
}
