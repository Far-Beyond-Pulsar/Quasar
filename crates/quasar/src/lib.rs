// quasar-audio: Facade crate — re-exports all public API from sub-crates
// and provides the top-level SpatialAudioEngine.

pub use quasar_core;
pub use quasar_materials;
pub use quasar_dsp;
pub use quasar_backends;

/// High-level convenience alias for the full workspace.
pub mod prelude {
    pub use quasar_core::*;
    pub use quasar_dsp::*;
    pub use quasar_materials::*;
    pub use quasar_backends::*;
}

use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::param_exchange::{ParameterTripleBuffer, SpatialCoefficients};
use quasar_core::probe_grid::AcousticProbeGrid;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::crossfader::EqualPowerCrossfader;
use quasar_dsp::node_graph::AudioNodeGraph;
use quasar_materials::registry::AcousticMaterialRegistry;

/// Top-level Quasar spatial audio engine.
///
/// Owns all subsystems and manages the compute → audio thread data flow.
/// The *compute thread* (15–30 Hz) calls [`update_spatial`] to query the
/// backend / probe grid and publish fresh spatial coefficients via the
/// lock-free triple buffer.  The *audio thread* (48 kHz) calls
/// [`process_audio`] to read those coefficients, smoothly crossfade
/// between old and new parameters, and run the DSP node graph.
///
/// NEVER allocates, locks, or blocks on the audio-thread hot path.
pub struct SpatialAudioEngine {
    hybrid_sampler: HybridProbeSampler,
    triple_buffer: ParameterTripleBuffer,
    material_registry: AcousticMaterialRegistry,
    dsp_graph: AudioNodeGraph,
    crossfader: EqualPowerCrossfader,
    sample_rate: f32,
    fade_ms: f32,
}

impl SpatialAudioEngine {
    /// Create a new engine with default configuration.
    ///
    /// `sample_rate` — audio sample rate in Hz (e.g. 48 000.0).
    /// `fade_ms`    — crossfade duration in milliseconds (typically 10–20 ms).
    pub fn new(sample_rate: f32, fade_ms: f32) -> Self {
        let initial = SpatialCoefficients {
            source_id: 0,
            direct_gain: quasar_core::bands::Band8::splat(0.0),
            direct_delay_samples: 0.0,
            early_reflections: Vec::new(),
            late_t60: quasar_core::bands::Band8::splat(0.5),
            late_gain_db: -10.0,
            version: 0,
        };

        Self {
            hybrid_sampler: HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly),
            triple_buffer: ParameterTripleBuffer::new(initial.clone()),
            material_registry: AcousticMaterialRegistry::new(),
            dsp_graph: AudioNodeGraph::new(),
            crossfader: EqualPowerCrossfader::new(fade_ms, sample_rate, initial),
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

    /// Run a spatial update cycle (called from the compute thread at 15–30 Hz).
    ///
    /// Resolves every [`SpatialQuery`] through the hybrid sampler (baked probe
    /// grid, real-time ray tracing, or a blend of both) and publishes the
    /// resulting [`SpatialCoefficients`] to the lock-free triple buffer so the
    /// audio thread can pick them up on its next [`process_audio`] call.
    pub fn update_spatial(
        &self,
        queries: &[SpatialQuery],
    ) {
        for query in queries {
            let result = self
                .hybrid_sampler
                .resolve(query, &self.material_registry);

            if let Ok(res) = result {
                let early_reflections: Vec<_> = res
                    .early_reflections
                    .iter()
                    .map(|er| quasar_core::param_exchange::EarlyReflectionCoeffs {
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

                // SAFETY: called from the compute thread (single writer).
                unsafe {
                    *self.triple_buffer.begin_write() = coeffs;
                }
                self.triple_buffer.end_write();
            }
        }
    }

    /// Process one audio block (called from the audio thread at 48 kHz).
    ///
    /// Reads the latest [`SpatialCoefficients`] from the triple buffer,
    /// applies an equal-power crossfade, then runs the entire audio node
    /// graph to produce the final output.
    ///
    /// # Safety
    ///
    /// NEVER allocates, locks, or blocks.
    pub fn process_audio(
        &mut self,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        // Pick up the latest published coefficients.
        self.triple_buffer.update();
        // SAFETY: called from the audio thread (single reader).
        let latest = unsafe { self.triple_buffer.read() };

        // Feed the new target into the crossfader so it begins blending.
        self.crossfader.set_target(latest.clone());

        // Build the param slice for the DSP graph (one param set per input).
        let params: Vec<SpatialCoefficients> = (0..inputs.len())
            .map(|_| self.crossfader.current_coefficients().clone())
            .collect();

        self.dsp_graph.process(inputs, &params, output);

        // Advance the crossfader by one frame.
        self.crossfader.advance();
    }

    /// Reset all DSP state (delay lines, filters, etc.).
    pub fn reset(&mut self) {
        self.dsp_graph.reset_all();
    }
}
