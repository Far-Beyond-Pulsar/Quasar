pub use quasar_core;
pub use quasar_materials;
pub use quasar_dsp;
pub use quasar_backends;

pub mod prelude {
    pub use quasar_core::*;
    pub use quasar_dsp::*;
    pub use quasar_materials::*;
    pub use quasar_backends::*;
}

use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::param_exchange::{
    EarlyReflectionCoeffs, ParameterTripleBuffer, SpatialCoefficients,
};
use quasar_core::probe_grid::AcousticProbeGrid;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::crossfader::EqualPowerCrossfader;
use quasar_dsp::node_graph::AudioNodeGraph;
use quasar_materials::registry::AcousticMaterialRegistry;

/// Top-level Quasar spatial audio engine for multiple sources.
///
/// Manages lock-free handoff of per-source spatial coefficients via
/// `ParameterTripleBuffer`, per-source crossfaders, and a shared DSP graph.
///
/// **Compute thread** (15–30 Hz): call [`update_spatial`] for each source.
/// **Audio thread** (48 kHz): call [`process_audio`] with one input per source.
pub struct SpatialAudioEngine {
    hybrid_sampler: HybridProbeSampler,
    triple_buffers: ParameterTripleBuffer,
    material_registry: AcousticMaterialRegistry,
    dsp_graph: AudioNodeGraph,
    crossfaders: Vec<EqualPowerCrossfader>,
    num_sources: usize,
    sample_rate: f32,
    fade_ms: f32,
}

impl SpatialAudioEngine {
    /// Create a new engine with the given number of sources.
    pub fn new(num_sources: usize, sample_rate: f32, fade_ms: f32) -> Self {
        let initial = SpatialCoefficients {
            source_id: 0,
            direct_gain: quasar_core::bands::Band8::splat(0.0),
            direct_delay_samples: 0.0,
            early_reflections: Vec::new(),
            late_t60: quasar_core::bands::Band8::splat(0.5),
            late_gain_db: -10.0,
            version: 0,
        };

        let triple_buffers = ParameterTripleBuffer::new(num_sources, initial.clone());

        let crossfaders = (0..num_sources)
            .map(|_| EqualPowerCrossfader::new(fade_ms, sample_rate, initial.clone()))
            .collect();

        Self {
            hybrid_sampler: HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly),
            triple_buffers,
            material_registry: AcousticMaterialRegistry::new(),
            dsp_graph: AudioNodeGraph::new(),
            crossfaders,
            num_sources,
            sample_rate,
            fade_ms,
        }
    }

    /// Set the real-time compute backend.
    pub fn set_backend(&mut self, backend: Box<dyn IAcousticComputeBackend>) {
        self.hybrid_sampler.set_realtime_backend(backend);
    }

    /// Set baked probe grid data.
    pub fn set_probe_grid(&mut self, grid: AcousticProbeGrid) {
        self.hybrid_sampler.set_probe_grid(grid);
    }

    /// Set hybrid sampling strategy.
    pub fn set_strategy(&mut self, strategy: HybridSamplingStrategy) {
        self.hybrid_sampler.set_strategy(strategy);
    }

    /// Get a reference to the material registry.
    pub fn materials(&self) -> &AcousticMaterialRegistry {
        &self.material_registry
    }

    /// Get a mutable reference to the material registry.
    pub fn materials_mut(&mut self) -> &mut AcousticMaterialRegistry {
        &mut self.material_registry
    }

    /// Get the DSP graph for configuring audio routing.
    pub fn dsp_graph(&mut self) -> &mut AudioNodeGraph {
        &mut self.dsp_graph
    }

    /// Number of sources configured.
    pub fn num_sources(&self) -> usize {
        self.num_sources
    }

    /// Run a spatial update cycle for one source (called from compute thread).
    ///
    /// Resolves `query` through the hybrid sampler and publishes the resulting
    /// [`SpatialCoefficients`] to that source's slot in the lock-free triple buffer.
    pub fn update_spatial(&self, query: &SpatialQuery) {
        let result = self
            .hybrid_sampler
            .resolve(query, &self.material_registry);

        if let Ok(res) = result {
            let early_reflections: Vec<_> = res
                .early_reflections
                .iter()
                .map(|er| EarlyReflectionCoeffs {
                    azimuth: 0.0,
                    elevation: 0.0,
                    delay_samples: er.delay_samples,
                    gain: er.gain,
                })
                .collect();

            let coeffs = SpatialCoefficients {
                source_id: query.source_id,
                direct_gain: res.direct_path.attenuation,
                direct_delay_samples: res.direct_path.delay_samples,
                early_reflections,
                late_t60: res.late_reverb.t60,
                late_gain_db: res.late_reverb.late_loudness_db,
                version: 0,
            };

            let src = query.source_id as usize;
            if src < self.triple_buffers.num_sources() {
                unsafe {
                    *self.triple_buffers.begin_write(src) = coeffs;
                }
                self.triple_buffers.end_write(src);
            }
        }
    }

    /// Process one audio block (called from the audio thread).
    ///
    /// `inputs`: one [`AudioBuffer`] per source.
    /// `output`: final mixed output buffer.
    ///
    /// # Safety
    ///
    /// NEVER allocates, locks, or blocks.
    pub fn process_audio(
        &mut self,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        self.triple_buffers.update();

        let params: Vec<SpatialCoefficients> = (0..inputs.len().min(self.num_sources))
            .map(|src| {
                let latest = unsafe { self.triple_buffers.read(src) };
                self.crossfaders[src].set_target(latest.clone());
                self.crossfaders[src].current_coefficients().clone()
            })
            .collect();

        self.dsp_graph.process(inputs, &params, output);

        for src in 0..params.len() {
            self.crossfaders[src].advance();
        }
    }

    /// Reset all DSP state.
    pub fn reset(&mut self) {
        self.dsp_graph.reset_all();
    }
}
