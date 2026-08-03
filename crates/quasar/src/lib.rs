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
use quasar_core::bands::Band8;
use quasar_core::error::SpatialAudioError;
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::param_exchange::{
    EarlyReflectionCoeffs, ParameterTripleBuffer, SpatialCoefficients,
};
use quasar_core::probe_grid::AcousticProbeGrid;
use quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig,
    SceneOutputId, SourceConfig, SourceId,
};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE, MAX_AUDIO_CHANNELS};
use quasar_dsp::crossfader::EqualPowerCrossfader;
use quasar_dsp::early_reflections::EarlyReflectionDelayNode;
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::master_decoder::{layout_positions, vbap_gains, SpeakerLayout};
use quasar_dsp::node_graph::{AudioNode, AudioNodeGraph};
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;
use quasar_dsp::patch_bay::{PatchBayNode, PatchEntry};
use quasar_materials::registry::AcousticMaterialRegistry;

/// One triple-buffer slot per (listener × scene output), flat index =
/// `listener * n_out + output`. Contains the per-pair smoothing crossfaders and
/// all preallocated DSP node chains and scratch buffers for the scene pipeline.
///
/// All scratch buffers are struct fields (heap), never `process()` locals —
/// an `AudioBuffer` is ~32 KB and would blow the audio-thread stack.
struct SceneRenderState {
    /// Patch bay: one mono output per scene output.
    patch_bay: PatchBayNode,
    /// One triple-buffer slot per (listener × scene output), flat index = listener * n_out + output.
    triple_buffers: ParameterTripleBuffer,
    /// One crossfader per (listener × scene output), same flat indexing.
    crossfaders: Vec<EqualPowerCrossfader>,
    last_versions: Vec<u64>,
    /// Per scene output DSP chain (mono). Reference listener for occ/early/rev = listener 0.
    occ: Vec<AirAbsorptionOcclusionNode>,
    early: Vec<EarlyReflectionDelayNode>,
    rev: Vec<FdnReverbNode>,
    /// Per-listener cached resolved speaker positions (from listener.physical_layout).
    listener_layouts: Vec<Vec<[f32; 3]>>,
    /// Preallocated scratch (all mono unless noted), sized to DEFAULT_BLOCK_SIZE.
    mixed: Vec<AudioBuffer>,        // patch bay output per scene output
    filtered: Vec<AudioBuffer>,     // occ output per scene output
    reflections: Vec<AudioBuffer>,  // early output (MONO) per scene output
    reverb: Vec<AudioBuffer>,       // rev output per scene output
    combined: Vec<AudioBuffer>,     // per-output final mono = filtered + reflections + reverb
    block: usize,                   // samples per block (from last process call, for crossfader advance)
    /// Sample rate captured at construction; mirrors `SpatialAudioEngine::sample_rate`.
    #[allow(dead_code)] // stored per the locked SceneRenderState layout; engine reads its own field
    sample_rate: f32,
}

impl SceneRenderState {
    /// Create an empty scene-render state (no outputs, no listeners).
    ///
    /// The registry starts empty, so this equals the state `rebuild_scene_render`
    /// would produce. Real node chains are built by
    /// [`SpatialAudioEngine::rebuild_scene_render`] as content is registered.
    fn empty(sample_rate: f32) -> Self {
        let initial = initial_scene_coeffs();
        Self {
            patch_bay: PatchBayNode::new(0),
            triple_buffers: ParameterTripleBuffer::new(0, initial.clone()),
            crossfaders: Vec::new(),
            last_versions: Vec::new(),
            occ: Vec::new(),
            early: Vec::new(),
            rev: Vec::new(),
            listener_layouts: Vec::new(),
            mixed: Vec::new(),
            filtered: Vec::new(),
            reflections: Vec::new(),
            reverb: Vec::new(),
            combined: Vec::new(),
            block: DEFAULT_BLOCK_SIZE,
            sample_rate,
        }
    }
}

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
    last_versions: Vec<u64>,
    num_sources: usize,
    sample_rate: f32,
    fade_ms: f32,

    // ── P2 scene pipeline (channel pulling, zero-alloc audio thread) ─────
    /// Render state for the scene pipeline added in P2.
    scene: SceneRenderState,

    // ── P1 content model (data model that later phases render) ────────────
    /// Loaded multi-channel sources, indexed by `SourceId`.
    sources: Vec<SourceConfig>,
    /// Positioned world emitters, indexed by `SceneOutputId`.
    scene_outputs: Vec<SceneOutputConfig>,
    /// Listener configurations, indexed by `ListenerId`.
    listeners: Vec<ListenerConfig>,
    /// Next ID to hand out for a freshly loaded source.
    next_source_id: u32,
    /// Next ID to hand out for a freshly added scene output.
    next_scene_output_id: u32,
    /// Next ID to hand out for a freshly added listener.
    next_listener_id: u32,
}

impl SpatialAudioEngine {
    /// Create a new engine with the given number of sources.
    pub fn new(num_sources: usize, sample_rate: f32, fade_ms: f32) -> Self {
        let initial = SpatialCoefficients {
            source_id: 0,
            direct_gain: quasar_core::bands::Band8::splat(1.0),
            direct_delay_samples: 0.0,
            direct_azimuth: 0.0, direct_elevation: 0.0,
            early_reflections: Vec::new(),
            late_t60: quasar_core::bands::Band8::splat(0.5),
            late_gain_db: 0.0,
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
            last_versions: vec![0; num_sources],
            num_sources,
            sample_rate,
            fade_ms,
            scene: SceneRenderState::empty(sample_rate),
            sources: Vec::new(),
            scene_outputs: Vec::new(),
            listeners: Vec::new(),
            next_source_id: 0,
            next_scene_output_id: 0,
            next_listener_id: 0,
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

    /// Loaded source registry (the P1 content model).
    pub fn sources(&self) -> &[SourceConfig] {
        &self.sources
    }

    /// Scene output registry (the P1 content model).
    pub fn scene_outputs(&self) -> &[SceneOutputConfig] {
        &self.scene_outputs
    }

    /// Listener registry (the P1 content model).
    pub fn listeners(&self) -> &[ListenerConfig] {
        &self.listeners
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

            // Compute direct-path azimuth/elevation from query geometry
            let dx = query.source_position[0] - query.listener_position[0];
            let dy = query.source_position[1] - query.listener_position[1];
            let dz = query.source_position[2] - query.listener_position[2];
            let direct_azimuth = dx.atan2(-dz);
            let direct_elevation = dy.atan2((dx * dx + dz * dz).sqrt());

            let coeffs = SpatialCoefficients {
                source_id: query.source_id,
                direct_gain: res.direct_path.attenuation,
                direct_delay_samples: res.direct_path.delay_samples,
                direct_azimuth,
                direct_elevation,
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
                let ver = self.triple_buffers.read_version(src);
                if ver > self.last_versions[src] {
                    self.last_versions[src] = ver;
                    let latest = unsafe { self.triple_buffers.read(src) };
                    self.crossfaders[src].set_target(latest.clone());
                }
                self.crossfaders[src].current_coefficients().clone()
            })
            .collect();

        self.dsp_graph.process(inputs, &params, output);

        for src in 0..params.len() {
            self.crossfaders[src].advance(output.samples() as usize);
        }
    }

    // ── Scene pipeline (P2: channel pulling end-to-end) ──────────────────

    /// Resolve every `(SceneOutput, Listener)` pair through the hybrid sampler
    /// and publish per-pair [`SpatialCoefficients`].
    ///
    /// Compute thread (15–30 Hz). Mirrors the legacy [`update_spatial`] math for
    /// the direct path and derives early-reflection azimuth/elevation from each
    /// reflection's world direction.
    pub fn update_scene_spatial(&self) {
        let n_out = self.scene_outputs.len();
        let n_lis = self.listeners.len();
        // Guard: the engine may have been rebuilt between calls, so the
        // triple-buffer size may differ from n_out * n_lis. Clamp to the min
        // to avoid out-of-bounds writes.
        let n_pairs = n_out
            .saturating_mul(n_lis)
            .min(self.scene.triple_buffers.num_sources());

        for l in 0..n_lis {
            for o in 0..n_out {
                let idx = (l * n_out + o) as u32;
                if idx as usize >= n_pairs {
                    continue;
                }
                let query = SpatialQuery {
                    source_position: self.scene_outputs[o].position,
                    listener_position: self.listeners[l].position,
                    source_id: idx,
                };
                if let Ok(res) = self.hybrid_sampler.resolve(&query, &self.material_registry) {
                    let early_reflections: Vec<_> = res
                        .early_reflections
                        .iter()
                        .map(|er| EarlyReflectionCoeffs {
                            azimuth: er.direction[0].atan2(-er.direction[2]),
                            elevation: er.direction[1].atan2(
                                (er.direction[0] * er.direction[0]
                                    + er.direction[2] * er.direction[2])
                                    .sqrt(),
                            ),
                            delay_samples: er.delay_samples,
                            gain: er.gain,
                        })
                        .collect();

                    // Direct-path azimuth/elevation from query geometry,
                    // mirroring the legacy update_spatial math.
                    let dx = query.source_position[0] - query.listener_position[0];
                    let dy = query.source_position[1] - query.listener_position[1];
                    let dz = query.source_position[2] - query.listener_position[2];

                    let coeffs = SpatialCoefficients {
                        source_id: idx,
                        direct_gain: res.direct_path.attenuation,
                        direct_delay_samples: res.direct_path.delay_samples,
                        direct_azimuth: dx.atan2(-dz),
                        direct_elevation: dy.atan2((dx * dx + dz * dz).sqrt()),
                        early_reflections,
                        late_t60: res.late_reverb.t60,
                        late_gain_db: res.late_reverb.late_loudness_db,
                        version: 0,
                    };

                    unsafe {
                        *self.scene.triple_buffers.begin_write(idx as usize) = coeffs;
                    }
                    self.scene.triple_buffers.end_write(idx as usize);
                }
            }
        }
    }

    /// Process one audio block through the scene pipeline. ZERO ALLOCATION.
    ///
    /// `sources`: one buffer per Source (each with its own channel count).
    /// `listener_outputs`: one buffer per Listener (exact match required, else
    /// explicit panic).
    ///
    /// Pipeline per block:
    ///   1. publish latest triple-buffer data;
    ///   2. smooth per-pair coefficients through the crossfaders;
    ///   3. patch bay sums the configured pulls into one mono buffer per scene output;
    ///   4. per scene output (once, reference listener 0): occlusion, early
    ///      reflections, late reverb → `combined`;
    ///   5. per listener: VBAP-decode every scene output's mono onto the
    ///      listener's physical layout and sum into that listener's output;
    ///   6. advance all crossfaders.
    ///
    /// # Panics
    ///
    /// Panics (with a clear message) if the caller's listener output count
    /// differs from the registered listener count, or if any buffer exceeds
    /// `DEFAULT_BLOCK_SIZE` samples.
    pub fn process_audio_scene(&mut self, sources: &[&AudioBuffer], listener_outputs: &mut [AudioBuffer]) {
        let n_out = self.scene_outputs.len();
        let n_lis = self.listeners.len();

        assert_eq!(
            listener_outputs.len(),
            n_lis,
            "process_audio_scene: expected {} listener output buffer(s) (one per registered listener), got {}",
            n_lis,
            listener_outputs.len()
        );

        for s in sources {
            assert!(
                (s.samples() as usize) <= DEFAULT_BLOCK_SIZE,
                "process_audio_scene: source buffer has {} samples, exceeding DEFAULT_BLOCK_SIZE ({DEFAULT_BLOCK_SIZE})",
                s.samples()
            );
        }
        for l in listener_outputs.iter() {
            assert!(
                (l.samples() as usize) <= DEFAULT_BLOCK_SIZE,
                "process_audio_scene: listener output buffer has {} samples, exceeding DEFAULT_BLOCK_SIZE ({DEFAULT_BLOCK_SIZE})",
                l.samples()
            );
        }

        if n_out == 0 || n_lis == 0 {
            for l in listener_outputs.iter_mut() {
                l.clear();
            }
            return;
        }

        let block = listener_outputs[0].samples() as usize;
        self.scene.block = block;

        // 1. Publish latest coefficients from the compute thread.
        self.scene.triple_buffers.update();

        // 2. Smooth per-pair coefficients: re-target ONLY on a strictly newer
        //    version. The triple buffer rotates its read slot through 3 slots
        //    per `update()`, so a snapshot the compute thread wrote once will be
        //    re-read with its *old* version on later blocks; retargeting on any
        //    change would oscillate back to the initial coefficients.
        let n_pairs = n_out
            .saturating_mul(n_lis)
            .min(self.scene.triple_buffers.num_sources())
            .min(self.scene.crossfaders.len());
        for idx in 0..n_pairs {
            let ver = self.scene.triple_buffers.read_version(idx);
            if ver > self.scene.last_versions[idx] {
                self.scene.last_versions[idx] = ver;
                let latest = unsafe { self.scene.triple_buffers.read(idx) };
                self.scene.crossfaders[idx].set_target(latest.clone());
            }
        }

        // 3. Patch bay: sum pulls into one mono buffer per scene output.
        self.scene.patch_bay.process(sources, &mut self.scene.mixed);

        // 4. Spatial render ONCE per scene output. Reference listener = 0, so
        //    the flat index for (listener 0, output o) is simply `o`.
        let n_out_proc = n_out
            .min(self.scene.occ.len())
            .min(self.scene.early.len())
            .min(self.scene.rev.len())
            .min(self.scene.mixed.len())
            .min(self.scene.filtered.len())
            .min(self.scene.reflections.len())
            .min(self.scene.reverb.len())
            .min(self.scene.combined.len())
            .min(self.scene.crossfaders.len());
        for o in 0..n_out_proc {
            let coeff = self.scene.crossfaders[o].current_coefficients();

            // (a) Occlusion / air absorption / direct-path delay.
            self.scene.occ[o].update_occlusion(&coeff.direct_gain, coeff.direct_delay_samples);
            self.scene.occ[o].process(&self.scene.mixed[o], &mut self.scene.filtered[o], coeff);

            // (b) Early reflection taps (mono contribution; spatialized in P3).
            self.scene.early[o].update_reflections(&coeff.early_reflections);
            self.scene.early[o].process(&self.scene.mixed[o], &mut self.scene.reflections[o], coeff);

            // (c) Late reverb: T60 from the room, dry/wet split derived from the
            //     resolved late gain. Wet-only tap; the dry path is handled by
            //     the occlusion stage above.
            self.scene.rev[o].set_t60(&coeff.late_t60);
            let wet = db_to_linear(coeff.late_gain_db);
            self.scene.rev[o].set_mix(wet, 0.0);
            self.scene.rev[o].process(&self.scene.mixed[o], &mut self.scene.reverb[o], coeff);

            // (d) Final mono for this output = filtered + reflections + reverb.
            self.scene.combined[o].copy_from(&self.scene.filtered[o]);
            self.scene.combined[o].add_from(&self.scene.reflections[o]);
            self.scene.combined[o].add_from(&self.scene.reverb[o]);
        }

        // 5. Per-listener decode: VBAP the rendered mono onto the physical
        //    layout and sum into the listener's output bus.
        let n_lis_proc = n_lis
            .min(self.scene.listener_layouts.len())
            .min(listener_outputs.len());
        let mut gains = [0.0_f32; MAX_AUDIO_CHANNELS];
        for l in 0..n_lis_proc {
            let out = &mut listener_outputs[l];
            out.clear();
            let n_speakers = out.channels() as usize;
            for o in 0..n_out_proc {
                let coeff = self.scene.crossfaders[l * n_out + o].current_coefficients();
                let n = vbap_gains(
                    &self.scene.listener_layouts[l],
                    coeff.direct_azimuth,
                    coeff.direct_elevation,
                    &mut gains,
                );
                let combined_ch = self.scene.combined[o].channel(0);
                for sp in 0..n.min(n_speakers) {
                    let g = gains[sp];
                    if g == 0.0 {
                        continue;
                    }
                    let ch = out.channel_mut(sp as u16);
                    for i in 0..block {
                        ch[i] += combined_ch[i] * g;
                    }
                }
            }
        }

        // 6. Advance all crossfaders by the block size (P1 unit fix: fades
        //    complete in ~fade_ms of real time, not per-sample).
        for c in 0..n_pairs {
            self.scene.crossfaders[c].advance(block);
        }
    }

    /// Rebuild the scene-render state from the current registry.
    ///
    /// API/config thread only (requires `&mut self`). In the demo the engine is
    /// behind `Arc<Mutex<..>>`, so this is mutually exclusive with the audio
    /// thread's [`process_audio_scene`] and the compute thread's
    /// [`update_scene_spatial`].
    fn rebuild_scene_render(&mut self) {
        let n_out = self.scene_outputs.len();
        let n_lis = self.listeners.len();
        let sr = self.sample_rate;
        let fade_ms = self.fade_ms;

        // Patch bay: repopulate pulls (API surface is dB; DSP is linear).
        let mut patch_bay = PatchBayNode::new(n_out);
        for (o, output) in self.scene_outputs.iter().enumerate() {
            for pull in &output.pulls {
                patch_bay.set_pull(
                    o,
                    PatchEntry {
                        source_idx: pull.source_id.0 as usize,
                        channel: pull.channel as usize,
                        gain_linear: db_to_linear(pull.gain_db),
                    },
                );
            }
        }

        // One triple-buffer slot + smoothing crossfader per (listener × output).
        let initial = initial_scene_coeffs();
        let triple_buffers = ParameterTripleBuffer::new(n_out * n_lis, initial.clone());
        let crossfaders = (0..n_out * n_lis)
            .map(|_| EqualPowerCrossfader::new(fade_ms, sr, initial.clone()))
            .collect();

        // Per-output mono DSP chain. Reference listener for occ/early/rev = 0.
        let occ = (0..n_out)
            .map(|_| AirAbsorptionOcclusionNode::new(1, sr, 0.1))
            .collect();
        let early = (0..n_out)
            .map(|_| EarlyReflectionDelayNode::new(1, sr, 0.2, 16))
            .collect();
        let rev = (0..n_out).map(|_| FdnReverbNode::new(1, sr)).collect();

        let listener_layouts = self
            .listeners
            .iter()
            .map(|l| layout_positions(&physical_to_speaker_layout(&l.physical_layout)))
            .collect();

        let mono = |count: usize| -> Vec<AudioBuffer> {
            (0..count)
                .map(|_| AudioBuffer::new(1, DEFAULT_BLOCK_SIZE as u16))
                .collect()
        };
        let mixed = mono(n_out);
        let filtered = mono(n_out);
        let reflections = mono(n_out);
        let reverb = mono(n_out);
        let combined = mono(n_out);

        self.scene = SceneRenderState {
            patch_bay,
            triple_buffers,
            crossfaders,
            last_versions: vec![0; n_out * n_lis],
            occ,
            early,
            rev,
            listener_layouts,
            mixed,
            filtered,
            reflections,
            reverb,
            combined,
            block: self.scene.block,
            sample_rate: sr,
        };
    }

    /// Reset all DSP state.
    pub fn reset(&mut self) {
        self.dsp_graph.reset_all();
    }

    // ── Source registry ─────────────────────────────────────────────────

    /// Load a multi-channel audio source and return its [`SourceId`].
    ///
    /// The source is registered in the content model only; decoding and buffer
    /// management is handled by the game-side audio system (P2 renders it).
    ///
    /// # Errors
    ///
    /// Returns [`SpatialAudioError::InvalidScene`] if `cfg.channels` is zero —
    /// a source must declare at least one channel for a [`ChannelPull`] to tap.
    pub fn load_source(&mut self, cfg: SourceConfig) -> Result<SourceId, SpatialAudioError> {
        if cfg.channels == 0 {
            return Err(SpatialAudioError::InvalidScene(format!(
                "source '{}' must declare at least one channel",
                cfg.path
            )));
        }
        let id = SourceId(self.next_source_id);
        self.next_source_id += 1;
        self.sources.push(cfg);
        self.rebuild_scene_render();
        Ok(id)
    }

    /// Remove a source from the registry.
    ///
    /// Any pulls referencing this source are removed from every scene output
    /// so the patch bay never points at a stale source.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered source.
    ///
    /// # Note
    ///
    /// Order-preserving: surviving sources keep `SourceId == index`, and
    /// surviving pulls referencing sources after `id` are renumbered down by
    /// one so the patch bay's `source_id -> buffer index` mapping stays exact.
    pub fn unload_source(&mut self, id: SourceId) {
        let idx = self.source_index(id);
        self.sources.remove(idx); // order-preserving; keeps ID == index for survivors
        for output in &mut self.scene_outputs {
            output.pulls.retain(|p| p.source_id.0 != id.0);
            for p in output.pulls.iter_mut() {
                if p.source_id.0 > id.0 {
                    p.source_id.0 -= 1; // shift surviving IDs down to stay sequential
                }
            }
        }
        self.next_source_id = self.sources.len() as u32;
        self.rebuild_scene_render();
    }

    // ── Scene output registry ───────────────────────────────────────────

    /// Add a positioned scene output and return its [`SceneOutputId`].
    pub fn add_scene_output(&mut self, cfg: SceneOutputConfig) -> SceneOutputId {
        let id = SceneOutputId(self.next_scene_output_id);
        self.next_scene_output_id += 1;
        self.scene_outputs.push(cfg);
        self.rebuild_scene_render();
        id
    }

    /// Remove a scene output from the registry.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    ///
    /// # Note
    ///
    /// Order-preserving: surviving scene outputs keep `SceneOutputId == index`
    /// and the render state is rebuilt so the patch bay and triple-buffer
    /// flat indexing (`listener * n_out + output`) stay consistent.
    pub fn remove_scene_output(&mut self, id: SceneOutputId) {
        let idx = self.scene_output_index(id);
        self.scene_outputs.remove(idx);
        self.next_scene_output_id = self.scene_outputs.len() as u32;
        self.rebuild_scene_render();
    }

    /// Move a scene output to a new world-space position.
    ///
    /// Content-model only: does NOT rebuild the scene render state. Geometry is
    /// re-resolved by the next [`update_scene_spatial`]; see [`update_listener`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_position(&mut self, id: SceneOutputId, pos: [f32; 3]) {
        let idx = self.scene_output_index(id);
        self.scene_outputs[idx].position = pos;
    }

    // ── Patch bay ───────────────────────────────────────────────────────

    /// Add (or replace) a [`ChannelPull`] on a scene output.
    ///
    /// An identical `(source_id, channel)` tap replaces the existing gain;
    /// otherwise the pull is appended. All gains are smoothed in later phases.
    ///
    /// # Panics
    ///
    /// Panics if `output` does not refer to a registered scene output.
    pub fn connect_pull(&mut self, output: SceneOutputId, pull: ChannelPull) {
        let idx = self.scene_output_index(output);
        let pulls = &mut self.scene_outputs[idx].pulls;
        match pulls
            .iter_mut()
            .find(|p| p.source_id == pull.source_id && p.channel == pull.channel)
        {
            Some(existing) => existing.gain_db = pull.gain_db,
            None => pulls.push(pull),
        }
        self.rebuild_scene_render();
    }

    /// Remove every [`ChannelPull`] tapping `(source, channel)` on an output.
    ///
    /// # Panics
    ///
    /// Panics if `output` does not refer to a registered scene output.
    pub fn disconnect_pull(&mut self, output: SceneOutputId, source: SourceId, channel: u32) {
        let idx = self.scene_output_index(output);
        self.scene_outputs[idx]
            .pulls
            .retain(|p| p.source_id != source || p.channel != channel);
        self.rebuild_scene_render();
    }

    /// Update the gain (dB) of an existing pull.
    ///
    /// If no pull taps `(source, channel)` on this output, this is a no-op
    /// (it does not panic).
    ///
    /// # Panics
    ///
    /// Panics if `output` does not refer to a registered scene output.
    pub fn set_pull_gain(
        &mut self,
        output: SceneOutputId,
        source: SourceId,
        channel: u32,
        gain_db: f32,
    ) {
        let idx = self.scene_output_index(output);
        let pulls = &mut self.scene_outputs[idx].pulls;
        if let Some(existing) = pulls
            .iter_mut()
            .find(|p| p.source_id == source && p.channel == channel)
        {
            existing.gain_db = gain_db;
        }
        self.rebuild_scene_render();
    }

    // ── Listener registry ───────────────────────────────────────────────

    /// Add a listener and return its [`ListenerId`].
    pub fn add_listener(&mut self, cfg: ListenerConfig) -> ListenerId {
        let id = ListenerId(self.next_listener_id);
        self.next_listener_id += 1;
        self.listeners.push(cfg);
        self.rebuild_scene_render();
        id
    }

    /// Remove a listener from the registry.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    ///
    /// # Note
    ///
    /// Order-preserving: surviving listeners keep `ListenerId == index` and
    /// the render state is rebuilt so the triple-buffer flat indexing
    /// (`listener * n_out + output`) stays consistent.
    pub fn remove_listener(&mut self, id: ListenerId) {
        let idx = self.listener_index(id);
        self.listeners.remove(idx);
        self.next_listener_id = self.listeners.len() as u32;
        self.rebuild_scene_render();
    }

    /// Update a listener's world position and heading.
    ///
    /// Content-model only: does NOT rebuild the scene render state. Position and
    /// heading feed [`update_scene_spatial`] (called separately, typically once
    /// per frame), and the render state depends only on structure (counts,
    /// layouts, pulls). Rebuilding here would recreate every crossfader and DSP
    /// delay line on each call, causing level jumps and audible clicks.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn update_listener(&mut self, id: ListenerId, position: [f32; 3], heading: [f32; 3]) {
        let idx = self.listener_index(id);
        self.listeners[idx].position = position;
        self.listeners[idx].heading = heading;
    }

    // ── Index helpers ────────────────────────────────────────────────────

    fn source_index(&self, id: SourceId) -> usize {
        let idx = id.0 as usize;
        assert!(
            idx < self.sources.len(),
            "invalid SourceId {id:?}: no such source"
        );
        idx
    }

    fn scene_output_index(&self, id: SceneOutputId) -> usize {
        let idx = id.0 as usize;
        assert!(
            idx < self.scene_outputs.len(),
            "invalid SceneOutputId {id:?}: no such scene output"
        );
        idx
    }

    fn listener_index(&self, id: ListenerId) -> usize {
        let idx = id.0 as usize;
        assert!(
            idx < self.listeners.len(),
            "invalid ListenerId {id:?}: no such listener"
        );
        idx
    }
}

/// Default `SpatialCoefficients` used to seed scene-pipeline crossfaders.
fn initial_scene_coeffs() -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: 0.0,
        version: 0,
    }
}

/// Convert a dB gain to linear amplitude.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Map a listener's physical output layout to a VBAP [`SpeakerLayout`].
///
/// HRTF decoding is deferred to P3; for now it falls back to Stereo.
fn physical_to_speaker_layout(layout: &PhysicalOutputLayout) -> SpeakerLayout {
    match layout {
        PhysicalOutputLayout::Stereo => SpeakerLayout::Stereo,
        PhysicalOutputLayout::Surround51 => SpeakerLayout::Surround51,
        PhysicalOutputLayout::Surround714 => SpeakerLayout::Surround714,
        PhysicalOutputLayout::Quad => SpeakerLayout::Quad,
        PhysicalOutputLayout::Custom { positions } => {
            SpeakerLayout::Custom {
                positions: positions.clone(),
            }
        }
        PhysicalOutputLayout::Hrtf => SpeakerLayout::Stereo,
    }
}

