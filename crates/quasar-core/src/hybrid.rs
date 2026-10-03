use crate::backend::{
    DirectPathResult, DEFAULT_SAMPLE_RATE, SPEED_OF_SOUND, IAcousticComputeBackend, LateReverbEstimate, MaterialProvider, SpatialQuery,
    SpatialQueryResult,
};
use crate::distance::DistanceModel;
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
    /// Audio device sample rate; `delay_samples` the sampler itself produces
    /// (BakedOnly) is `distance * sample_rate / SPEED_OF_SOUND`.
    sample_rate: f32,
    /// Direct-path distance law (shared with the real-time backend).
    distance_model: DistanceModel,
    /// Air temperature (C) / relative humidity (%) for BakedOnly air absorption.
    atmosphere: (f32, f32),
    /// Whether `set_atmosphere` was called (otherwise an installed backend keeps its own config).
    atmosphere_set: bool,
}

impl HybridProbeSampler {
    /// Create a new hybrid sampler with the given strategy.
    pub fn new(strategy: HybridSamplingStrategy) -> Self {
        Self {
            strategy,
            probe_grid: None,
            realtime_backend: None,
            sample_rate: DEFAULT_SAMPLE_RATE,
            distance_model: DistanceModel::default(),
            atmosphere: (20.0, 50.0),
            atmosphere_set: false,
        }
    }

    /// Set the device sample rate (also forwarded to the real-time backend).
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        if let Some(b) = self.realtime_backend.as_mut() {
            b.set_sample_rate(sample_rate);
        }
    }

    /// Set the distance model (also forwarded to the real-time backend), so
    /// BakedOnly, RealTimeOnly and HybridBlend give the same distance gain.
    pub fn set_distance_model(&mut self, model: DistanceModel) {
        self.distance_model = model;
        if let Some(b) = self.realtime_backend.as_mut() {
            b.set_distance_model(model);
        }
    }

    /// Air temperature (C) and relative humidity (%) used for the air absorption
    /// the sampler itself applies (BakedOnly; default 20 C / 50 %). Forwarded to
    /// the real-time backend (and re-applied when one is installed) so every
    /// strategy uses the same atmosphere.
    pub fn set_atmosphere(&mut self, temperature_celsius: f32, humidity_percent: f32) {
        self.atmosphere = (temperature_celsius, humidity_percent);
        self.atmosphere_set = true;
        if let Some(b) = self.realtime_backend.as_mut() {
            b.set_atmosphere(temperature_celsius, humidity_percent);
        }
    }

    /// The active distance model.
    pub fn distance_model(&self) -> DistanceModel {
        self.distance_model
    }

    /// The sample rate delays are expressed in.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Set the baked probe grid data.
    pub fn set_probe_grid(&mut self, grid: AcousticProbeGrid) {
        self.probe_grid = Some(grid);
    }

    /// Set the real-time compute backend.
    pub fn set_realtime_backend(&mut self, mut backend: Box<dyn IAcousticComputeBackend>) {
        backend.set_sample_rate(self.sample_rate);
        backend.set_distance_model(self.distance_model);
        if self.atmosphere_set {
            backend.set_atmosphere(self.atmosphere.0, self.atmosphere.1);
        }
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
    /// Called from the compute thread (15–30 Hz). Equivalent to a one-element
    /// [`resolve_batch`](Self::resolve_batch).
    pub fn resolve(
        &self,
        query: &SpatialQuery,
        materials: &dyn MaterialProvider,
    ) -> Result<SpatialQueryResult, SpatialAudioError> {
        self.resolve_batch(std::slice::from_ref(query), materials)
            .into_iter()
            .next()
            .unwrap_or_else(|| Err(SpatialAudioError::Backend("real-time backend returned no results".into())))
    }

    /// Resolve many source-listener pairs at once (#151).
    ///
    /// RealTimeOnly / HybridBlend issue ONE `query_spatial` for every query (the CPU backend
    /// fans the pairs out over its rayon pool, the GPU backend does a single dispatch) and then
    /// apply the per-query probe overlay; BakedOnly maps per query. Element `i` of the returned
    /// vector is exactly what [`resolve`](Self::resolve) gives for `queries[i]`. A configuration
    /// error (missing grid / backend) is reported for every query.
    pub fn resolve_batch(
        &self,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Vec<Result<SpatialQueryResult, SpatialAudioError>> {
        if queries.is_empty() {
            return Vec::new();
        }
        let fail_all = |make: &dyn Fn() -> SpatialAudioError| -> Vec<Result<SpatialQueryResult, SpatialAudioError>> {
            queries.iter().map(|_| Err(make())).collect()
        };
        match self.strategy {
            HybridSamplingStrategy::BakedOnly => {
                let Some(grid) = self.probe_grid.as_ref() else {
                    return fail_all(&|| SpatialAudioError::ProbeGrid("no probe grid configured for BakedOnly strategy".into()));
                };
                queries.iter().map(|query| self.resolve_baked(grid, query)).collect()
            }
            HybridSamplingStrategy::RealTimeOnly => {
                let Some(backend) = self.realtime_backend.as_ref() else {
                    return fail_all(&|| SpatialAudioError::Backend("no real-time backend configured for RealTimeOnly strategy".into()));
                };
                Self::backend_batch(backend.as_ref(), queries, materials)
            }
            HybridSamplingStrategy::HybridBlend => {
                let Some(backend) = self.realtime_backend.as_ref() else {
                    return fail_all(&|| SpatialAudioError::Backend("no real-time backend configured for HybridBlend strategy".into()));
                };
                let Some(grid) = self.probe_grid.as_ref() else {
                    return fail_all(&|| SpatialAudioError::ProbeGrid("no probe grid configured for HybridBlend strategy".into()));
                };
                let mut out = Self::backend_batch(backend.as_ref(), queries, materials);
                // Overlay baked late reverb from the probe grid.
                // A listener outside the grid (outside the building, or in a corner the grid
                // does not span) keeps the backend's own statistical late estimate instead of
                // failing the whole query, which dropped the direct and early paths too (the
                // caller then never updated that pair).
                for (query, r) in queries.iter().zip(out.iter_mut()) {
                    if let Ok(result) = r {
                        if let Some(sample) = grid.sample(&query.listener_position) {
                            result.late_reverb = baked_late_estimate(&sample, grid);
                        }
                    }
                }
                out
            }
        }
    }

    /// One `query_spatial` for the whole batch; results are matched to queries by position.
    fn backend_batch(
        backend: &dyn IAcousticComputeBackend,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Vec<Result<SpatialQueryResult, SpatialAudioError>> {
        let mut results = backend.query_spatial(queries, materials).into_iter();
        queries
            .iter()
            .map(|_| {
                results
                    .next()
                    .ok_or_else(|| SpatialAudioError::Backend("real-time backend returned no results".into()))
            })
            .collect()
    }

    fn resolve_baked(&self, grid: &AcousticProbeGrid, query: &SpatialQuery) -> Result<SpatialQueryResult, SpatialAudioError> {
        // Sample the grid at the listener position to get reverb info.
        let sample = grid
            .sample(&query.listener_position)
            .ok_or_else(|| SpatialAudioError::ProbeGrid("listener position is outside the probe grid".into()))?;

        let dx = query.source_position[0] - query.listener_position[0];
        let dy = query.source_position[1] - query.listener_position[1];
        let dz = query.source_position[2] - query.listener_position[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Shared distance law and ISO 9613-1 air absorption (same as the
        // real-time backends, so the clear-path gain matches across strategies).
        let attenuations = crate::bands::Band8::splat(self.distance_model.gain(distance))
            .mul(&crate::air::air_absorption_gain(distance, self.atmosphere.0, self.atmosphere.1));

        Ok(SpatialQueryResult {
            source_id: query.source_id,
            direct_path: DirectPathResult {
                attenuation: attenuations,
                delay_samples: distance * self.sample_rate / SPEED_OF_SOUND,
                distance,
                occluded: false,
                occlusion_factor: 1.0,
                occlusion: crate::bands::Band8::splat(1.0),
            },
            early_reflections: Vec::new(),
            late_reverb: baked_late_estimate(&sample, grid),
        })
    }
}

/// Late-reverb estimate from a probe-grid sample (no constants): T60 and the early /
/// late split come from the interpolated probes; the level comes from the baked RIRs
/// when there are any, else from the diffuse-field model
/// ([`crate::reverb_model::late_loudness_from_t60_volume`]) with the mean T60 and the
/// volume spanned by the grid.
fn baked_late_estimate(sample: &crate::probe_grid::AcousticProbeSample, grid: &AcousticProbeGrid) -> LateReverbEstimate {
    LateReverbEstimate {
        t60: sample.t60,
        early_late_split_secs: sample.early_late_split_secs,
        late_loudness_db: sample.late_loudness_db.unwrap_or_else(|| {
            crate::reverb_model::late_loudness_from_t60_volume(sample.t60.mean(), grid.volume_m3())
        }),
    }
}
