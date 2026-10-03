pub use quasar_core;
pub use quasar_materials;
pub use quasar_dsp;
pub use quasar_backends;

/// Streaming source implementations (behind `streaming` feature).
#[cfg(feature = "streaming")]
pub mod streaming_source;

mod output_stage;
mod render;
mod renderer;
pub mod resampler;
pub mod source_resampler;
pub use render::{LFE_CUTOFF_HZ, MAX_LISTENERS, MAX_PROPAGATION_DELAY_SECS, MAX_SCENE_OUTPUTS};
pub use renderer::AudioRenderer;
use render::{
    db_to_linear, initial_scene_coeffs, Command, Garbage, ListenerRender, OutputAdd, OutputRender, PairRender,
    SceneRenderState,
};
use renderer::{RendererShared, COMMAND_QUEUE_CAPACITY, GARBAGE_QUEUE_CAPACITY};

pub mod prelude {
    pub use quasar_core::*;
    pub use quasar_dsp::*;
    pub use quasar_materials::*;
    pub use quasar_backends::*;
}

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery, SPEED_OF_SOUND};
use quasar_core::bands::Band8;
use quasar_core::emitter_pattern::{EmitterModel, EmitterPattern, EmitterShape, EmitterTrace};
use quasar_core::source_directivity::{diffuse_send_gain, first_order_bounce_point};
use quasar_core::distance::DistanceModel;
use quasar_core::error::SpatialAudioError;
use quasar_core::hybrid::{HybridProbeSampler, HybridSamplingStrategy};
use quasar_core::spsc::{spsc_channel, SpscConsumer, SpscProducer};
use quasar_core::param_exchange::{
    EarlyReflectionCoeffs, ParameterTripleBuffer, SpatialCoefficients,
};
use quasar_core::probe_grid::AcousticProbeGrid;
use quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig,
    SceneOutputId, SourceConfig, SourceId,
};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use quasar_dsp::crossfader::EqualPowerCrossfader;
use quasar_dsp::limiter::{OutputMeter, OutputSafetyConfig};
use quasar_dsp::node_graph::AudioNodeGraph;
use quasar_dsp::patch_bay::{PatchBayBus, PatchEntry, MAX_PULLS_PER_OUTPUT};
use quasar_materials::registry::AcousticMaterialRegistry;

/// Atomic instrumentation for `process_audio_scene`.  All fields are relaxed-
/// ordered atomics written from the audio callback and read from the demo's
/// UI thread — no lock needed, no data race.
pub struct AudioTiming {
    /// Cumulative wall-clock nanoseconds spent in `process_audio_scene`.
    pub total_ns: AtomicU64,
    /// Peak single-call duration (nanoseconds).
    pub max_ns: AtomicU64,
    /// Number of `process_audio_scene` calls recorded.
    pub call_count: AtomicU64,
}

impl AudioTiming {
    pub const fn new() -> Self {
        Self {
            total_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
            call_count: AtomicU64::new(0),
        }
    }

    /// Snapshot the current counters (for display, avg calculation, etc.).
    pub fn snapshot(&self, block_size: usize, sample_rate: f32) -> AudioTimingSnapshot {
        let total = self.total_ns.load(Ordering::Relaxed);
        let max = self.max_ns.load(Ordering::Relaxed);
        let count = self.call_count.load(Ordering::Relaxed);
        let block_us = block_size as f64 / sample_rate as f64 * 1e6;
        let avg = if count > 0 { total / count } else { 0 };
        AudioTimingSnapshot {
            total_ns: total,
            max_ns: max,
            avg_ns: avg,
            call_count: count,
            block_us,
            headroom_us: (block_us - max as f64 / 1000.0).max(0.0),
        }
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.total_ns.store(0, Ordering::Relaxed);
        self.max_ns.store(0, Ordering::Relaxed);
        self.call_count.store(0, Ordering::Relaxed);
    }
}

/// Human-readable timing readout from [`AudioTiming::snapshot`].
#[derive(Clone, Copy, Debug)]
pub struct AudioTimingSnapshot {
    pub total_ns: u64,
    pub max_ns: u64,
    pub avg_ns: u64,
    pub call_count: u64,
    /// Available wall-clock time per block (μs).
    pub block_us: f64,
    /// How much time is left after the worst measured call (μs).
    pub headroom_us: f64,
}

/// Top-level Quasar spatial audio engine for multiple sources.
///
/// Manages lock-free handoff of per-source spatial coefficients via
/// `ParameterTripleBuffer`, per-source crossfaders, and a shared DSP graph.
///
/// **Compute thread** (15–30 Hz): call [`update_scene_spatial`] (or the legacy
/// [`update_spatial`] per source).
/// **Audio thread** (48 kHz): call [`process_audio_scene`] (or the legacy [`process_audio`]).
///
/// # Edits do not rebuild the renderer (#73)
///
/// Patch-bay and registry edits mutate the render state in place: pull gains are ramped
/// (attack/release, [`set_pull_ramp_ms`](Self::set_pull_ramp_ms)), and adding / removing an
/// output or listener inserts / removes only that entity's DSP state; delay lines, reverb tails,
/// crossfaders and ramps of everything that survives are untouched. A newly added output (pair)
/// is silent until its first coefficients are published by [`update_scene_spatial`] (#119).
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
    /// The audio-side renderer while the engine is "combined" (before [`audio_handle`](Self::audio_handle)
    /// moves it to the audio thread). `None` once split.
    renderer: Option<AudioRenderer>,
    /// Command ring to the audio thread (used only once split; the combined engine applies edits directly).
    cmd_tx: SpscProducer<Command>,
    /// Retired audio-side DSP state, dropped here.
    garbage_rx: SpscConsumer<Garbage>,
    /// State shared lock-free with the renderer.
    shared: Arc<RendererShared>,
    /// Compute-side writer handles of the per-pair parameter triple buffers, `[listener][output]`
    /// (the audio-side `PairRender` holds the other `Arc`).
    pair_params: Vec<Vec<Arc<ParameterTripleBuffer>>>,
    /// Last resolved source/listener poses, used to skip unchanged pairs (#128).
    last_pair_poses: Vec<Vec<Option<([f32; 3], [f32; 3], [f32; 3])>>>,
    /// The emitter models (#156) must be pushed to the backend before the next resolve.
    emitters_dirty: bool,
    /// `(outputs, listeners)` the backend's emitter table was last built for.
    emitter_sync_shape: (usize, usize),

    // ── P1 content model (data model that later phases render) ────────────
    /// Loaded multi-channel sources, indexed by `SourceId`.
    sources: Vec<SourceConfig>,
    /// Positioned world emitters, indexed by `SceneOutputId`.
    scene_outputs: Vec<SceneOutputConfig>,
    /// Listener configurations, indexed by `ListenerId`.
    listeners: Vec<ListenerConfig>,
    /// LFE send (linear) per scene output, parallel to `scene_outputs`. Kept here,
    /// not in `SceneOutputConfig`, so the config struct stays source compatible.
    lfe_sends: Vec<f32>,
    /// Optional per-output distance-model override, parallel to `scene_outputs`
    /// (`None` = use the engine-wide model). Kept here for the same reason as
    /// `lfe_sends`: `SceneOutputConfig` stays source compatible.
    distance_overrides: Vec<Option<DistanceModel>>,
    /// Next ID to hand out for a freshly loaded source.
    next_source_id: u32,
    /// Next ID to hand out for a freshly added scene output.
    next_scene_output_id: u32,
    /// Next ID to hand out for a freshly added listener.
    next_listener_id: u32,
    /// Output-stage meters of each listener (shared with the renderer), parallel to `listeners`.
    listener_meters: Vec<Arc<OutputMeter>>,
    /// Output-stage configuration of each listener, parallel to `listeners`.
    safety_cfgs: Vec<OutputSafetyConfig>,
    /// User mix trims `[reverb_db, early_db]` of each listener, parallel to `listeners`.
    mix_trims_db: Vec<[f32; 2]>,

    /// Debug stage selector for isolating noise sources:
    ///   0 = silence, 1 = raw pull only, 2 = +occlusion, 3 = +early reflections, 4 = full.
    /// Set via the demo (`1` key cycles stages); `process_audio_scene` gates the
    /// per-output spatial chain accordingly.
    pub debug_audio_stage: u8,

    /// Lock-free timing instrumentation for the audio callback (shared with the [`AudioRenderer`]).
    pub timing: Arc<AudioTiming>,
}

impl SpatialAudioEngine {
    /// Snapshot the audio-thread timing counters (for display or logging).
    pub fn timing_snapshot(&self) -> AudioTimingSnapshot {
        self.timing.snapshot(DEFAULT_BLOCK_SIZE, self.sample_rate)
    }

    /// Number of late-reverb (FDN) nodes in the render state: one per LISTENER,
    /// independent of the number of scene outputs (#62).
    pub fn reverb_node_count(&self) -> usize {
        self.listeners.len()
    }
}

impl SpatialAudioEngine {
    /// Create a new engine with the given number of sources.
    pub fn new(num_sources: usize, sample_rate: f32, fade_ms: f32) -> Self {
        let initial = initial_scene_coeffs();

        let triple_buffers = ParameterTripleBuffer::new(num_sources, initial.clone());

        // Spatial targets normally arrive at 20–30 Hz. Keep parameter ramps alive
        // across that update cadence so delay trajectories do not plateau between
        // short user-configured fades (#120).
        let spatial_fade_ms = fade_ms.max(50.0);
        let crossfaders = (0..num_sources)
            .map(|_| EqualPowerCrossfader::new(spatial_fade_ms, sample_rate, initial.clone()))
            .collect();

        let (cmd_tx, cmd_rx) = spsc_channel::<Command>(COMMAND_QUEUE_CAPACITY);
        let (garbage_tx, garbage_rx) = spsc_channel::<Garbage>(GARBAGE_QUEUE_CAPACITY);
        let shared = Arc::new(RendererShared::new());
        let timing = Arc::new(AudioTiming::new());
        let renderer = AudioRenderer::new(
            SceneRenderState::new(sample_rate),
            cmd_rx,
            garbage_tx,
            Arc::clone(&shared),
            Arc::clone(&timing),
        );

        Self {
            hybrid_sampler: HybridProbeSampler::new(HybridSamplingStrategy::RealTimeOnly),
            triple_buffers,
            material_registry: AcousticMaterialRegistry::new(),
            dsp_graph: AudioNodeGraph::new(),
            crossfaders,
            last_versions: vec![0; num_sources],
            num_sources,
            sample_rate,
            fade_ms: spatial_fade_ms,
            renderer: Some(renderer),
            cmd_tx,
            garbage_rx,
            shared,
            pair_params: Vec::new(),
            last_pair_poses: Vec::new(),
            emitters_dirty: true,
            emitter_sync_shape: (usize::MAX, usize::MAX),
            sources: Vec::new(),
            scene_outputs: Vec::new(),
            listeners: Vec::new(),
            lfe_sends: Vec::new(),
            distance_overrides: Vec::new(),
            next_source_id: 0,
            next_scene_output_id: 0,
            next_listener_id: 0,
            listener_meters: Vec::new(),
            safety_cfgs: Vec::new(),
            mix_trims_db: Vec::new(),
            debug_audio_stage: 4,
            timing,
        }
    }

    /// Set the real-time compute backend.
    pub fn set_backend(&mut self, backend: Box<dyn IAcousticComputeBackend>) {
        // The sampler forwards the engine rate to the backend so every
        // `delay_samples` it reports is in samples at the device rate.
        self.hybrid_sampler.set_sample_rate(self.sample_rate);
        self.hybrid_sampler.set_realtime_backend(backend);
        self.last_pair_poses.clear();
        self.emitters_dirty = true;
    }

    /// Push the emitter table (#156) to the backend when it changed: one entry per pair id
    /// (`listener * n_out + output`, the `source_id` of the queries) for every emitter that has a
    /// non-omnidirectional pattern or a non-default shape. Rebuilt when an emitter changed, when
    /// a backend was installed, or when the number of outputs / listeners changed.
    fn sync_emitters_to_backend(&mut self, n_out: usize, n_lis: usize) {
        if !self.emitters_dirty && self.emitter_sync_shape == (n_out, n_lis) {
            return;
        }
        self.emitters_dirty = false;
        self.emitter_sync_shape = (n_out, n_lis);
        let mut table: Vec<(u32, EmitterTrace)> = Vec::new();
        for o in 0..n_out {
            let cfg = &self.scene_outputs[o];
            let pattern = if cfg.orientation.is_some() { resolved_emitter_pattern(cfg) } else { None };
            if pattern.is_none() && cfg.emitter.shape == EmitterShape::Default {
                continue; // omnidirectional, historic aperture: the backend default
            }
            let trace = EmitterTrace {
                pattern,
                forward: cfg.orientation.unwrap_or([0.0, 0.0, -1.0]),
                shape: cfg.emitter.shape.clone(),
            };
            for l in 0..n_lis {
                table.push(((l * n_out + o) as u32, trace.clone()));
            }
        }
        if let Some(backend) = self.hybrid_sampler.realtime_backend_mut() {
            backend.set_emitters(&table);
        }
    }

    /// Set baked probe grid data.
    pub fn set_probe_grid(&mut self, grid: AcousticProbeGrid) {
        self.hybrid_sampler.set_probe_grid(grid);
    }

    /// Set the engine-wide distance-attenuation model (default: inverse
    /// distance, 1 m reference, 6.02 dB per doubling, clamped at 1 m).
    ///
    /// The same model is used by BakedOnly and the installed real-time
    /// backend, so the strategy does not change the direct-path loudness.
    pub fn set_distance_model(&mut self, model: DistanceModel) {
        self.hybrid_sampler.set_distance_model(model);
        self.last_pair_poses.clear();
    }

    /// Set the air temperature (Celsius) and relative humidity (percent) used for
    /// air absorption by every strategy (baked, real-time, hybrid). Takes effect
    /// on the next compute update (#121).
    pub fn set_atmosphere(&mut self, temperature_celsius: f32, humidity_percent: f32) {
        self.hybrid_sampler.set_atmosphere(temperature_celsius, humidity_percent);
        self.last_pair_poses.clear();
    }

    /// Override the distance model of one scene output (`None` = engine-wide model).
    ///
    /// Applied on the compute thread as the ratio `override.gain(d) / global.gain(d)`
    /// on the resolved direct gain, so it works with any strategy/backend.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_distance_model(&mut self, id: SceneOutputId, model: Option<DistanceModel>) {
        let idx = self.scene_output_index(id);
        self.distance_overrides[idx] = model;
        self.last_pair_poses.clear();
    }

    /// Set hybrid sampling strategy.
    pub fn set_strategy(&mut self, strategy: HybridSamplingStrategy) {
        self.hybrid_sampler.set_strategy(strategy);
        self.last_pair_poses.clear();
    }

    /// Get a reference to the material registry.
    pub fn materials(&self) -> &AcousticMaterialRegistry {
        &self.material_registry
    }

    /// Get a mutable reference to the material registry.
    pub fn materials_mut(&mut self) -> &mut AcousticMaterialRegistry {
        self.last_pair_poses.clear();
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
                .map(|er| {
                    let (azimuth, elevation) = direction_to_angles(er.direction);
                    EarlyReflectionCoeffs {
                        azimuth,
                        elevation,
                        delay_samples: er.delay_samples,
                        gain: er.gain,
                    }
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
                directivity_gain: Band8::splat(1.0),
                direct_azimuth,
                direct_elevation,
                early_reflections,
                late_t60: res.late_reverb.t60,
                late_gain_db: res.late_reverb.late_loudness_db,
                early_late_split_secs: res.late_reverb.early_late_split_secs,
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
    /// NEVER allocates, locks, or blocks (verified by `tests/no_alloc_tests.rs`, #79). The
    /// per-source coefficients are lent to the graph straight from the crossfaders; no
    /// temporary copies are made.
    pub fn process_audio(
        &mut self,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        self.triple_buffers.update();

        let n_src = inputs.len().min(self.num_sources).min(self.crossfaders.len());
        for src in 0..n_src {
            let ver = self.triple_buffers.read_version(src);
            if ver > self.last_versions[src] {
                self.last_versions[src] = ver;
                let latest = unsafe { self.triple_buffers.read(src) };
                self.crossfaders[src].set_target(latest);
            }
        }

        self.dsp_graph
            .process_with_params(&inputs[..n_src], &self.crossfaders[..n_src], output);

        for src in 0..n_src {
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
    pub fn update_scene_spatial(&mut self) {
        let n_out = self.scene_outputs.len();
        let n_lis = self.listeners.len();
        if self.last_pair_poses.len() != n_lis || self.last_pair_poses.iter().any(|r| r.len() != n_out) {
            self.last_pair_poses = vec![vec![None; n_out]; n_lis];
        }
        self.sync_emitters_to_backend(n_out, n_lis);

        // Phase 1: every pair whose pose changed since its last publish.
        let mut pending: Vec<(usize, usize, SpatialQuery, ([f32; 3], [f32; 3], _))> = Vec::new();
        for l in 0..n_lis {
            for o in 0..n_out {
                if self.pair_params.get(l).and_then(|p| p.get(o)).is_none() {
                    continue;
                }
                let idx = (l * n_out + o) as u32;
                let query = SpatialQuery {
                    source_position: self.scene_outputs[o].position,
                    listener_position: self.listeners[l].position,
                    source_id: idx,
                };
                let pose = (query.source_position, query.listener_position, self.listeners[l].heading);
                if self.last_pair_poses[l][o] == Some(pose) { continue; }
                pending.push((l, o, query, pose));
            }
        }
        if pending.is_empty() {
            return;
        }

        // Phase 2: ONE batched resolve for all of them (#151): a single `query_spatial`
        // (rayon fan-out on the CPU backend, one dispatch on the GPU backend) instead of one
        // backend call per pair.
        let queries: Vec<SpatialQuery> = pending.iter().map(|p| p.2.clone()).collect();
        let results = self.hybrid_sampler.resolve_batch(&queries, &self.material_registry);
        drop(queries);

        // Phase 3: publish each result exactly as the per-pair path did.
        for ((l, o, query, pose), result) in pending.into_iter().zip(results) {
            {
                let Some(params) = self.pair_params.get(l).and_then(|p| p.get(o)) else {
                    continue;
                };
                if let Ok(mut res) = result {
                    if let Some(Some(ov)) = self.distance_overrides.get(o) {
                        let global = self.hybrid_sampler.distance_model().gain(res.direct_path.distance);
                        if global > 1e-9 {
                            let ratio = ov.gain(res.direct_path.distance) / global;
                            res.direct_path.attenuation = res.direct_path.attenuation.scale(ratio);
                        }
                    }
                    // Source directivity (#74), per (listener, emitter) pair. An emitter without
                    // an orientation, or with `directivity == 0`, is omnidirectional: every factor
                    // below is then exactly 1.0 / 0 dB and the coefficients are unchanged.
                    let out_cfg = &self.scene_outputs[o];
                    let (e_pos, l_pos) = (query.source_position, query.listener_position);
                    // Pattern (#156): an explicit `emitter.pattern` (omni / cardioid family /
                    // super- and hypercardioid / sound cone / horn), else the legacy cardioid
                    // family from `directivity`. Needs an orientation, as before.
                    let resolved = resolved_emitter_pattern(out_cfg);
                    let pattern_on = out_cfg.orientation.is_some() && resolved.is_some();
                    let pattern_at = |target: Option<[f32; 3]>| -> Band8 {
                        match (&resolved, out_cfg.orientation, target) {
                            (Some(p), Some(fwd), Some(t)) if pattern_on => {
                                p.band_gains(fwd, [t[0] - e_pos[0], t[1] - e_pos[1], t[2] - e_pos[2]])
                            }
                            _ => Band8::splat(1.0),
                        }
                    };
                    // Direct path: emitter -> listener.
                    let directivity_gain = pattern_at(Some(l_pos));
                    // Speed of sound the backend used, recovered from its own direct path.
                    let c_eff = if res.direct_path.delay_samples > 1e-3 && res.direct_path.distance > 1e-3 {
                        res.direct_path.distance * self.sample_rate / res.direct_path.delay_samples
                    } else {
                        SPEED_OF_SOUND
                    };
                    // Reverb send: the late field follows the emitter's total radiated power.
                    let diffuse_db = if pattern_on {
                        let g = if out_cfg.emitter.pattern.is_some() {
                            out_cfg.emitter.diffuse_gain // cached at EmitterModel::new
                        } else {
                            diffuse_send_gain(out_cfg.directivity)
                        };
                        20.0 * g.max(1e-6).log10()
                    } else {
                        0.0
                    };

                    let early_reflections: Vec<_> = res
                        .early_reflections
                        .iter()
                        .map(|er| {
                            let (azimuth, elevation) = direction_to_angles(er.direction);
                            // Each reflection leaves the emitter toward ITS first bounce point
                            // (exact for first order; for higher orders the last bounce point is
                            // used as the best available approximation of the departure).
                            let gain = if pattern_on {
                                let path = er.delay_samples * c_eff / self.sample_rate;
                                let bounce = first_order_bounce_point(e_pos, l_pos, er.direction, path);
                                er.gain.mul(&pattern_at(bounce))
                            } else {
                                er.gain
                            };
                            EarlyReflectionCoeffs { azimuth, elevation, delay_samples: er.delay_samples, gain }
                        })
                        .collect();

                    // Direct-path azimuth/elevation from query geometry,
                    // mirroring the legacy update_spatial math.
                    let dx = query.source_position[0] - query.listener_position[0];
                    let dy = query.source_position[1] - query.listener_position[1];
                    let dz = query.source_position[2] - query.listener_position[2];

                    // `source_id` is a constant 0: a pair's identity must not change when other
                    // outputs / listeners are added or removed (the crossfader snaps on an id
                    // change, which would break the in-place edits of #73).
                    let coeffs = SpatialCoefficients {
                        source_id: 0,
                        direct_gain: res.direct_path.attenuation,
                        direct_delay_samples: res.direct_path.delay_samples,
                        directivity_gain,
                        direct_azimuth: dx.atan2(-dz),
                        direct_elevation: dy.atan2((dx * dx + dz * dz).sqrt()),
                        early_reflections,
                        late_t60: res.late_reverb.t60,
                        late_gain_db: res.late_reverb.late_loudness_db + diffuse_db,
                        early_late_split_secs: res.late_reverb.early_late_split_secs,
                        version: 0,
                    };

                    unsafe {
                        *params.begin_write(0) = coeffs;
                    }
                    params.end_write(0);
                    self.last_pair_poses[l][o] = Some(pose);
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
    ///   1. publish latest triple-buffer data and smooth per-pair coefficients through the
    ///      crossfaders;
    ///   2. patch bay sums the configured (ramped) pulls into one mono buffer per scene output;
    ///   3. per scene output (once, listener-independent): the dry delay line;
    ///   4. per listener: for every output the direct chain (delay + band EQ / occlusion + gain,
    ///      per (listener, output) pair, #77), VBAP / HRTF decode onto the listener's layout,
    ///      early reflections, the shared reverb bus and the LFE bus;
    ///   5. advance all crossfaders.
    ///
    /// # Panics
    ///
    /// Panics (with a clear message) if the caller's listener output count
    /// differs from the registered listener count, or if any buffer exceeds
    /// `DEFAULT_BLOCK_SIZE` samples.
    pub fn process_audio_scene(&mut self, sources: &[&AudioBuffer], listener_outputs: &mut [AudioBuffer]) {
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

        self.shared.stage.store(self.debug_audio_stage, Ordering::Relaxed);
        let renderer = self
            .renderer
            .as_mut()
            .expect("process_audio_scene: the audio side was split off with audio_handle(); call AudioRenderer::process_audio_scene instead");
        renderer.process_audio_scene(sources, listener_outputs);
    }

    /// Move the audio side out of the engine (#75) and return it.
    ///
    /// The [`AudioRenderer`] is `Send` and owns everything the audio thread needs: move it into
    /// the audio callback and call [`AudioRenderer::process_audio_scene`] per block, with no
    /// mutex shared with this object. From now on every configuration call on the engine
    /// (`connect_pull`, `add_listener`, ...) is sent through a lock-free SPSC command queue and
    /// applied by the renderer at the start of its next block, and compute work
    /// ([`update_scene_spatial`](Self::update_scene_spatial)) publishes through per-pair triple
    /// buffers; the engine is then free to be held, locked or busy for as long as it likes without
    /// affecting the audio thread.
    ///
    /// After this, [`process_audio_scene`](Self::process_audio_scene) on the engine panics.
    /// Positions of emitters / listeners and the distance model are compute-side only (the
    /// renderer receives resolved coefficients), only the listener heading, LFE sends and
    /// patch-bay edits travel as commands.
    ///
    /// # Panics
    ///
    /// Panics if the audio side was already taken.
    pub fn audio_handle(&mut self) -> AudioRenderer {
        self.renderer
            .take()
            .expect("audio_handle: the audio side was already split off")
    }

    /// Set the debug stage (see [`debug_audio_stage`](Self::debug_audio_stage)); also works after
    /// [`audio_handle`](Self::audio_handle), where the public field no longer reaches the renderer.
    pub fn set_debug_audio_stage(&mut self, stage: u8) {
        self.debug_audio_stage = stage;
        self.shared.stage.store(stage, Ordering::Relaxed);
    }

    /// Drop the DSP state the audio thread has retired (removed outputs / listeners ...). Called
    /// by every configuration method; call it yourself if you only ever run
    /// [`update_scene_spatial`](Self::update_scene_spatial).
    pub fn reap_retired(&mut self) {
        while self.garbage_rx.pop().is_some() {}
    }

    /// Retired boxes the audio thread had to drop itself because this side did not reap them in
    /// time (should stay 0).
    pub fn garbage_overflow_count(&self) -> u64 {
        self.shared.garbage_overflow.load(Ordering::Relaxed)
    }

    /// Deliver a configuration command: applied at once while the engine still owns the renderer,
    /// otherwise pushed on the lock-free queue (the audio thread applies it at its next block).
    ///
    /// If the queue is full (1024 commands pending: the audio thread is not running) this waits,
    /// sleeping, for up to 5 s and then panics rather than dropping the edit.
    fn send(&mut self, cmd: Command) {
        self.reap_retired();
        if let Some(r) = self.renderer.as_mut() {
            r.apply(cmd);
            self.reap_retired();
            return;
        }
        let mut cmd = cmd;
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match self.cmd_tx.push(cmd) {
                Ok(()) => return,
                Err(back) => {
                    cmd = back;
                    assert!(
                        Instant::now() < deadline,
                        "configuration command queue is full: the audio thread is not consuming commands"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
    }

    /// Reset all DSP state.
    pub fn reset(&mut self) {
        self.dsp_graph.reset_all();
    }

    /// Set the attack / release ramp of patch-bay edits (default 15 ms; 0 = immediate).
    pub fn set_pull_ramp_ms(&mut self, ms: f32) {
        let samples = if ms.is_finite() { (ms.max(0.0) * 0.001 * self.sample_rate).round() as u32 } else { 0 };
        self.send(Command::SetPullRampSamples(samples));
    }

    // ── Source registry ─────────────────────────────────────────────────

    /// Load a multi-channel audio source and return its [`SourceId`].
    ///
    /// The source is registered in the content model only; decoding and buffer
    /// management is handled by the game-side audio system (P2 renders it).
    /// No render state changes: the patch bay indexes the caller's source buffers.
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
    /// The unloaded source's taps are dropped immediately (its audio is gone, so there is
    /// nothing to fade); nothing else in the render state changes.
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
        self.send(Command::RemoveSource(idx));
    }

    // ── Scene output registry ───────────────────────────────────────────

    /// Add a positioned scene output and return its [`SceneOutputId`].
    ///
    /// Only the new output's DSP state is created; existing outputs and listeners are untouched.
    /// The new (listener, output) pairs are silent until [`update_scene_spatial`] publishes their
    /// first coefficients.
    ///
    /// # Panics
    ///
    /// Panics beyond [`MAX_SCENE_OUTPUTS`] outputs or [`MAX_PULLS_PER_OUTPUT`] pulls on `cfg`.
    pub fn add_scene_output(&mut self, cfg: SceneOutputConfig) -> SceneOutputId {
        assert!(
            self.scene_outputs.len() < MAX_SCENE_OUTPUTS,
            "add_scene_output: at most {MAX_SCENE_OUTPUTS} scene outputs are supported"
        );
        assert!(
            cfg.pulls.len() <= MAX_PULLS_PER_OUTPUT,
            "add_scene_output: at most {MAX_PULLS_PER_OUTPUT} pulls per output are supported"
        );
        let sr = self.sample_rate;
        let mut new_params = Vec::with_capacity(self.listeners.len());
        let mut pairs = Vec::with_capacity(self.listeners.len());
        for l in &self.listeners {
            let params = Arc::new(ParameterTripleBuffer::new(1, initial_scene_coeffs()));
            pairs.push(PairRender::new(
                sr,
                self.fade_ms,
                l.physical_layout == PhysicalOutputLayout::Hrtf,
                0.0,
                Arc::clone(&params),
            ));
            new_params.push(params);
        }
        let o = self.scene_outputs.len();
        for (l, params) in new_params.into_iter().enumerate() {
            self.pair_params[l].push(params);
        }
        self.send(Command::AddOutput(Box::new(OutputAdd {
            output: Some(OutputRender::new(sr, 0.0)),
            bus: Some(PatchBayBus::new()),
            pairs,
        })));
        for pull in &cfg.pulls {
            self.send(Command::SetPull { output: o, entry: patch_entry(pull) });
        }
        let id = SceneOutputId(self.next_scene_output_id);
        self.next_scene_output_id += 1;
        self.scene_outputs.push(cfg);
        self.lfe_sends.push(0.0);
        self.distance_overrides.push(None);
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
    /// Order-preserving: surviving scene outputs keep `SceneOutputId == index`. Only the removed
    /// output's DSP state is dropped (immediately: its delay line and any tail of that emitter
    /// end with it); every surviving output / listener keeps its state.
    pub fn remove_scene_output(&mut self, id: SceneOutputId) {
        let idx = self.scene_output_index(id);
        self.scene_outputs.remove(idx);
        self.lfe_sends.remove(idx);
        self.distance_overrides.remove(idx);
        self.next_scene_output_id = self.scene_outputs.len() as u32;
        for pp in self.pair_params.iter_mut() {
            if idx < pp.len() {
                pp.remove(idx);
            }
        }
        self.send(Command::RemoveOutput(idx));
    }

    /// Move a scene output to a new world-space position.
    ///
    /// Content-model only: does NOT touch the scene render state. Geometry is
    /// re-resolved by the next [`update_scene_spatial`]; see [`update_listener`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_position(&mut self, id: SceneOutputId, pos: [f32; 3]) {
        let idx = self.scene_output_index(id);
        self.scene_outputs[idx].position = pos;
    }

    /// Orient a scene output and set how directional it is (#74).
    ///
    /// `orientation`: forward axis of the radiation pattern (any non-zero length; `None` makes the
    /// emitter omnidirectional). `directivity`: `0.0` omnidirectional .. `1.0` max cone (clamped);
    /// see [`quasar_core::source_directivity`] for the pattern. The pattern is evaluated per
    /// (listener, emitter) pair for the direct path, each early reflection (toward its own first
    /// bounce point) and the reverb send (diffuse-field average).
    ///
    /// Compute-side only: takes effect at the next [`update_scene_spatial`] (the new gains then
    /// fade in over the crossfade time).
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_directivity(&mut self, id: SceneOutputId, orientation: Option<[f32; 3]>, directivity: f32) {
        let idx = self.scene_output_index(id);
        self.scene_outputs[idx].orientation = orientation;
        self.scene_outputs[idx].directivity = if directivity.is_finite() { directivity.clamp(0.0, 1.0) } else { 0.0 };
        self.last_pair_poses.clear();
        self.emitters_dirty = true;
    }

    /// Set a scene output's radiation pattern (#156): spherical (`Omni`) or directional speaker.
    ///
    /// `pattern`: `None` returns to the legacy behaviour (cardioid family from the `directivity`
    /// of [`set_scene_output_directivity`]); `Some(Omni)` is an explicit spherical speaker;
    /// `Some(Horn { .. })` etc. are the directional types of [`quasar_core::emitter_pattern`].
    /// The pattern needs an orientation (set it with [`set_scene_output_directivity`], whose
    /// `directivity` is then ignored while a pattern is set); without one the emitter stays
    /// omnidirectional. Evaluated per (listener, emitter) pair for the direct path, each early
    /// reflection (toward its own first bounce point) and the reverb send (diffuse-field
    /// average, cached here), and passed to the backend so it can rank and prune image-source
    /// paths by the pattern.
    ///
    /// Compute-side only: takes effect at the next [`update_scene_spatial`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_pattern(&mut self, id: SceneOutputId, pattern: Option<EmitterPattern>) {
        let idx = self.scene_output_index(id);
        let shape = self.scene_outputs[idx].emitter.shape.clone();
        self.scene_outputs[idx].emitter = EmitterModel::new(pattern, shape);
        self.last_pair_poses.clear();
        self.emitters_dirty = true;
    }

    /// Set a scene output's physical aperture (#156): how the soft-occlusion probes are spread
    /// over the source (`Point`, `Sphere`, `Disc`, `Line`; `Default` = the historic 0.35 m disc).
    /// Compute-side only, applied at the next [`update_scene_spatial`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_shape(&mut self, id: SceneOutputId, shape: EmitterShape) {
        let idx = self.scene_output_index(id);
        self.scene_outputs[idx].emitter.shape = shape;
        self.last_pair_poses.clear();
        self.emitters_dirty = true;
    }

    /// Set a scene output's LFE send (linear gain, `>= 0`; default 0 = none).
    ///
    /// For every listener whose layout has an LFE slot (named 5.1 / 7.1), the
    /// output's rendered mono (already distance attenuated) times `linear` is
    /// summed with the other outputs' sends, low-passed once at
    /// [`LFE_CUTOFF_HZ`] (4th-order Butterworth) and added to the LFE channel.
    /// This is the only way signal reaches an LFE slot: it is never panned to.
    /// The send is independent of the output's normal panning (it is additive),
    /// and the change is ramped per sample over the next block. Listeners
    /// without an LFE slot (stereo, quad, custom, HRTF) ignore it.
    ///
    /// Content-model/config only: does NOT rebuild the render state.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn set_scene_output_lfe_send(&mut self, id: SceneOutputId, linear: f32) {
        let idx = self.scene_output_index(id);
        let g = if linear.is_finite() { linear.max(0.0) } else { 0.0 };
        self.lfe_sends[idx] = g;
        self.send(Command::SetLfeSend { output: idx, gain: g });
    }

    /// Current LFE send (linear) of a scene output.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered scene output.
    pub fn scene_output_lfe_send(&self, id: SceneOutputId) -> f32 {
        self.lfe_sends[self.scene_output_index(id)]
    }

    // ── Patch bay ───────────────────────────────────────────────────────

    /// Add (or replace) a [`ChannelPull`] on a scene output.
    ///
    /// An identical `(source_id, channel)` tap replaces the existing gain (gliding to it over
    /// the pull ramp); otherwise the pull is appended and fades in from silence. Edits made
    /// before the first rendered block take effect immediately. No other state is touched.
    ///
    /// # Panics
    ///
    /// Panics if `output` does not refer to a registered scene output, or if the output would
    /// exceed [`MAX_PULLS_PER_OUTPUT`] pulls.
    pub fn connect_pull(&mut self, output: SceneOutputId, pull: ChannelPull) {
        let idx = self.scene_output_index(output);
        let pulls = &mut self.scene_outputs[idx].pulls;
        match pulls
            .iter_mut()
            .find(|p| p.source_id == pull.source_id && p.channel == pull.channel)
        {
            Some(existing) => existing.gain_db = pull.gain_db,
            None => {
                assert!(
                    pulls.len() < MAX_PULLS_PER_OUTPUT,
                    "connect_pull: at most {MAX_PULLS_PER_OUTPUT} pulls per scene output are supported"
                );
                pulls.push(pull);
            }
        }
        self.send(Command::SetPull { output: idx, entry: patch_entry(&pull) });
    }

    /// Remove every [`ChannelPull`] tapping `(source, channel)` on an output.
    ///
    /// The tap fades out over the pull ramp and is dropped when silent.
    ///
    /// # Panics
    ///
    /// Panics if `output` does not refer to a registered scene output.
    pub fn disconnect_pull(&mut self, output: SceneOutputId, source: SourceId, channel: u32) {
        let idx = self.scene_output_index(output);
        self.scene_outputs[idx]
            .pulls
            .retain(|p| p.source_id != source || p.channel != channel);
        self.send(Command::RemovePull { output: idx, source_idx: source.0 as usize, channel: channel as usize });
    }

    /// Update the gain (dB) of an existing pull (a click-free glide over the pull ramp).
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
            self.send(Command::SetPullGain {
                output: idx,
                source_idx: source.0 as usize,
                channel: channel as usize,
                gain: db_to_linear(gain_db),
            });
        }
    }

    // ── Listener registry ───────────────────────────────────────────────

    /// Add a listener and return its [`ListenerId`].
    ///
    /// Only the new listener's DSP state is created (decoder, reverb bus, per-output pairs);
    /// everything else is untouched. Its pairs are silent until [`update_scene_spatial`]
    /// publishes their first coefficients.
    ///
    /// # Panics
    ///
    /// Panics beyond [`MAX_LISTENERS`] listeners.
    pub fn add_listener(&mut self, cfg: ListenerConfig) -> ListenerId {
        assert!(
            self.listeners.len() < MAX_LISTENERS,
            "add_listener: at most {MAX_LISTENERS} listeners are supported"
        );
        let sr = self.sample_rate;
        let hrtf = cfg.physical_layout == PhysicalOutputLayout::Hrtf;
        let mut params_row = Vec::with_capacity(self.scene_outputs.len());
        let mut pairs = Vec::with_capacity(self.scene_outputs.len());
        for o in 0..self.scene_outputs.len() {
            let params = Arc::new(ParameterTripleBuffer::new(1, initial_scene_coeffs()));
            pairs.push(PairRender::new(sr, self.fade_ms, hrtf, self.lfe_sends[o], Arc::clone(&params)));
            params_row.push(params);
        }
        let mut listener = ListenerRender::new(sr, &cfg);
        self.listener_meters.push(listener.meter());
        self.safety_cfgs.push(OutputSafetyConfig::default());
        self.mix_trims_db.push([0.0, 0.0]);
        listener.set_pairs(pairs);
        self.pair_params.push(params_row);
        self.send(Command::AddListener(listener));
        let id = ListenerId(self.next_listener_id);
        self.next_listener_id += 1;
        self.listeners.push(cfg);
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
    /// Order-preserving: surviving listeners keep `ListenerId == index`. Only the removed
    /// listener's DSP state is dropped; surviving listeners (decoder, reverb tail, pairs) are
    /// untouched.
    pub fn remove_listener(&mut self, id: ListenerId) {
        let idx = self.listener_index(id);
        self.listeners.remove(idx);
        self.next_listener_id = self.listeners.len() as u32;
        self.listener_meters.remove(idx);
        self.safety_cfgs.remove(idx);
        self.mix_trims_db.remove(idx);
        if idx < self.pair_params.len() {
            self.pair_params.remove(idx);
        }
        self.send(Command::RemoveListener(idx));
    }

    // ── Mix trims (reverb / early reflections) ─────────────────────────

    /// Trim the late reverb of a listener by `db` decibels (default 0 = the physical level).
    ///
    /// The engine's reverb is calibrated physically: relative to the direct sound of an emitter at
    /// 1 m it has the diffuse-field level of the room (`rev/direct = 312.2 T60 / (V Q)` in power,
    /// see `quasar_core::reverb_model`), so a distant emitter in a reverberant hall is mostly
    /// reverb, as in reality. Games and installations usually want a different balance; this is
    /// the "reverb send" fader for it. It scales ONLY this listener's late-reverb bus: the direct
    /// sound and the early reflections are untouched.
    ///
    /// Sent through the lock-free command queue and applied by the renderer with a per-sample
    /// linear ramp over the next block (no click, no allocation). `db` is clamped to
    /// `-120 ..= +24`; values at or below `-100` mute the reverb; a NaN is ignored. 0 dB is
    /// bit-identical to the engine without the trim.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn set_reverb_gain_db(&mut self, id: ListenerId, db: f32) {
        let idx = self.listener_index(id);
        if let Some(db) = sanitize_trim_db(db) {
            self.mix_trims_db[idx][0] = db;
            self.send(Command::SetReverbTrim { listener: idx, gain: trim_db_to_linear(db) });
        }
    }

    /// Reverb trim of a listener in dB (see [`set_reverb_gain_db`](Self::set_reverb_gain_db)).
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn reverb_gain_db(&self, id: ListenerId) -> f32 {
        self.mix_trims_db[self.listener_index(id)][0]
    }

    /// Trim the discrete early reflections of a listener by `db` decibels (default 0 = physical).
    ///
    /// Scales the gain of every early-reflection tap of this listener (the traced / image-source
    /// reflections handed to the reflection decoder); the direct sound and the late reverb are
    /// untouched. Same transport, clamping and bit-identical default as
    /// [`set_reverb_gain_db`](Self::set_reverb_gain_db); the per-tap gain ramp of the reflection
    /// decoder makes the change click-free.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn set_early_reflection_gain_db(&mut self, id: ListenerId, db: f32) {
        let idx = self.listener_index(id);
        if let Some(db) = sanitize_trim_db(db) {
            self.mix_trims_db[idx][1] = db;
            self.send(Command::SetEarlyTrim { listener: idx, gain: trim_db_to_linear(db) });
        }
    }

    /// Early-reflection trim of a listener in dB (see
    /// [`set_early_reflection_gain_db`](Self::set_early_reflection_gain_db)).
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn early_reflection_gain_db(&self, id: ListenerId) -> f32 {
        self.mix_trims_db[self.listener_index(id)][1]
    }

    // ── Output safety stage (#80) ──────────────────────────────────────

    /// Configure a listener's output safety stage: pre-limiter headroom, look-ahead peak limiter
    /// (ceiling, look-ahead / attack, release), NaN / inf scrub and metering. See
    /// [`OutputSafetyConfig`] and the `quasar_dsp::limiter` module docs.
    ///
    /// The default is enabled, ceiling -1 dBFS, ZERO latency (instant attack). A look-ahead
    /// (`lookahead_ms > 0`) adds exactly that much latency to this listener's output, reported by
    /// [`output_latency_samples`](Self::output_latency_samples). The limiter is a SAMPLE-peak
    /// limiter (no oversampling): intersample peaks can exceed the ceiling by up to a few dB.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn set_output_safety(&mut self, id: ListenerId, cfg: OutputSafetyConfig) {
        let idx = self.listener_index(id);
        self.safety_cfgs[idx] = cfg;
        self.send(Command::SetSafety { listener: idx, cfg });
    }

    /// Output safety configuration of a listener.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn output_safety(&self, id: ListenerId) -> OutputSafetyConfig {
        self.safety_cfgs[self.listener_index(id)]
    }

    /// Latency (samples at the engine rate) the output stage adds to a listener's bus: the
    /// look-ahead of its limiter (0 by default).
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn output_latency_samples(&self, id: ListenerId) -> usize {
        self.safety_cfgs[self.listener_index(id)].latency_samples(self.sample_rate)
    }

    /// Lock-free meters of a listener's output stage (peak, gain reduction, limited / clipped /
    /// non-finite sample counters). Readable from any thread.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn output_meter(&self, id: ListenerId) -> Arc<OutputMeter> {
        Arc::clone(&self.listener_meters[self.listener_index(id)])
    }

    /// Update a listener's world position and heading.
    ///
    /// Content-model only: does NOT rebuild the scene render state. Position feeds
    /// [`update_scene_spatial`] (called separately, typically once per frame); the heading is
    /// applied by the renderer from the next block.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn update_listener(&mut self, id: ListenerId, position: [f32; 3], heading: [f32; 3]) {
        let idx = self.listener_index(id);
        self.listeners[idx].position = position;
        self.listeners[idx].heading = heading;
        self.send(Command::SetListenerHeading { listener: idx, heading });
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

impl SpatialAudioEngine {
    /// Convert a listener's render to a different DEVICE layout as the last channel stage
    /// (#83, #149): the listener is still rendered in its physical layout (say 7.1), then mapped
    /// by the ITU-R BS.775-3 downmix matrix of `quasar_dsp::channel_matrix` (7.1 -> 5.1, 5.1 ->
    /// stereo, 7.1 -> stereo, quad -> stereo, identity for the same layout) and only THEN
    /// limited: the output safety stage runs on the device channels, so its ceiling holds on what
    /// is actually played. The buffer passed to `process_audio_scene` for this listener must have
    /// the DEVICE channel count.
    ///
    /// `None` removes the conversion. Conversions the matrix module does not define (upmix,
    /// custom layouts) return an error here, nothing is changed and no channel is ever silently
    /// dropped. Sent through the lock-free command queue; a change cross-fades the old and the new
    /// mapping over 10 ms (no click) and allocates only on this thread. Without a call the
    /// output is bit-identical to an engine without this stage.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not refer to a registered listener.
    pub fn set_listener_output_layout(
        &mut self,
        id: ListenerId,
        device: Option<PhysicalOutputLayout>,
    ) -> Result<(), quasar_dsp::channel_matrix::MatrixError> {
        use quasar_dsp::channel_matrix::{downmix_gains, layout_channel_count, DEFAULT_MATRIX_RAMP_MS};
        let idx = self.listener_index(id);
        let from = render::physical_to_speaker_layout(&self.listeners[idx].physical_layout);
        let n_phys = layout_channel_count(&from);
        let sr = self.sample_rate;
        let new = match device {
            None => None,
            Some(d) => {
                let to = render::physical_to_speaker_layout(&d);
                let gains = downmix_gains(&from, &to)?;
                let n_dev = layout_channel_count(&to);
                let identity = n_dev == n_phys
                    && (0..n_phys * n_phys).all(|i| gains[i] == if i % (n_phys + 1) == 0 { 1.0 } else { 0.0 });
                Some(output_stage::ConvStage::new(gains, n_phys, n_dev, sr, DEFAULT_MATRIX_RAMP_MS, identity)?)
            }
        };
        let ident: Vec<f32> = (0..n_phys * n_phys).map(|i| if i % (n_phys + 1) == 0 { 1.0 } else { 0.0 }).collect();
        let spare = output_stage::ConvStage::new(ident, n_phys, n_phys, sr, DEFAULT_MATRIX_RAMP_MS, true)?;
        let scratch = Box::new(AudioBuffer::new(n_phys as u16, DEFAULT_BLOCK_SIZE as u16));
        self.send(Command::SetConversion { listener: idx, swap: output_stage::ConvSwap::new(new, spare, scratch) });
        Ok(())
    }
}

/// Patch-bay entry of a pull (API surface is dB; DSP is linear).
fn patch_entry(pull: &ChannelPull) -> PatchEntry {
    PatchEntry {
        source_idx: pull.source_id.0 as usize,
        channel: pull.channel as usize,
        gain_linear: db_to_linear(pull.gain_db),
    }
}

/// The emitter's effective radiation pattern (#156): the explicit `emitter.pattern` (an omni
/// pattern resolves to `None`), else the legacy cardioid family from `directivity`, else `None`
/// (omnidirectional). `None` means every pattern factor is exactly 1.
fn resolved_emitter_pattern(cfg: &SceneOutputConfig) -> Option<EmitterPattern> {
    match &cfg.emitter.pattern {
        Some(p) if p.is_omni() => None,
        Some(p) => Some(p.clone()),
        None if cfg.directivity > 0.0 => Some(EmitterPattern::CardioidFamily { directivity: cfg.directivity }),
        None => None,
    }
}

/// World-space direction vector (listener -> reflection point / source) to `(azimuth, elevation)`
/// in the engine's convention (azimuth 0 = -Z, +X = right).
fn direction_to_angles(d: [f32; 3]) -> (f32, f32) {
    (d[0].atan2(-d[2]), d[1].atan2((d[0] * d[0] + d[2] * d[2]).sqrt()))
}

/// Clamp a user trim to `-120 ..= +24` dB; `None` for NaN.
fn sanitize_trim_db(db: f32) -> Option<f32> {
    if db.is_nan() {
        None
    } else {
        Some(db.clamp(-120.0, 24.0))
    }
}

/// Linear amplitude of a trim in dB; `<= -100 dB` is exactly silence, 0 dB exactly 1.0.
fn trim_db_to_linear(db: f32) -> f32 {
    if db <= -100.0 {
        0.0
    } else if db == 0.0 {
        1.0
    } else {
        10.0_f32.powf(db / 20.0)
    }
}

#[cfg(test)]
mod legacy_tests {
    use super::*;
    use quasar_core::bands::Band8;
    use quasar_core::backend::{
        DirectPathResult, EarlyReflection, LateReverbEstimate, MaterialProvider, SpatialQueryResult,
    };
    use quasar_core::rays::{Ray, RayHit};
    use quasar_core::scene::AcousticScene;

    /// Backend that reports one reflection arriving from +X (right) and one from +Y (above).
    struct Mock;
    impl IAcousticComputeBackend for Mock {
        fn query_spatial(&self, q: &[SpatialQuery], _m: &dyn MaterialProvider) -> Vec<SpatialQueryResult> {
            q.iter()
                .map(|q| SpatialQueryResult {
                    source_id: q.source_id,
                    direct_path: DirectPathResult {
                        attenuation: Band8::splat(1.0),
                        delay_samples: 10.0,
                        distance: 1.0,
                        occluded: false,
                        occlusion_factor: 1.0,
                        occlusion: Band8::splat(1.0),
                    },
                    early_reflections: vec![
                        EarlyReflection { direction: [1.0, 0.0, 0.0], delay_samples: 100.0, gain: Band8::splat(0.5), order: 1 },
                        EarlyReflection { direction: [0.0, 1.0, 0.0], delay_samples: 200.0, gain: Band8::splat(0.5), order: 1 },
                    ],
                    late_reverb: LateReverbEstimate {
                        t60: Band8::splat(0.5),
                        early_late_split_secs: 0.05,
                        late_loudness_db: -10.0,
                    },
                })
                .collect()
        }
        fn update_scene(&mut self, _s: &AcousticScene) -> Result<(), SpatialAudioError> {
            Ok(())
        }
        fn trace_ray(&self, _r: &Ray) -> Vec<RayHit> {
            Vec::new()
        }
    }

    /// #79: the legacy `update_spatial` derives each reflection's azimuth/elevation from the
    /// reflection direction (as the scene path does) instead of publishing 0/0.
    #[test]
    fn legacy_update_spatial_derives_reflection_angles() {
        let mut e = SpatialAudioEngine::new(1, 48_000.0, 15.0);
        e.set_backend(Box::new(Mock));
        e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
        e.update_spatial(&SpatialQuery { source_position: [0.0, 0.0, -2.0], listener_position: [0.0; 3], source_id: 0 });
        e.triple_buffers.update();
        let c = unsafe { e.triple_buffers.read(0) };
        assert_eq!(c.early_reflections.len(), 2);
        let (r, up) = (&c.early_reflections[0], &c.early_reflections[1]);
        assert!((r.azimuth - std::f32::consts::FRAC_PI_2).abs() < 1e-5, "az {}", r.azimuth);
        assert!(r.elevation.abs() < 1e-5);
        assert!((up.elevation - std::f32::consts::FRAC_PI_2).abs() < 1e-5, "el {}", up.elevation);
    }
}
