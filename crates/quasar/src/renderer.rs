//! The audio-thread half of the engine (#75).
//!
//! [`SpatialAudioEngine`](crate::SpatialAudioEngine) is the compute / configuration side: it owns
//! the registries, the hybrid sampler / backend and the writer ends of the per-pair parameter
//! triple buffers. [`AudioRenderer`] is the audio side: it owns the render state, the reader ends
//! of the triple buffers and the consumer of a lock-free SPSC command queue. It is `Send`, is
//! moved into the audio callback, and never takes a lock, waits, or allocates:
//!
//! ```text
//!   compute / API thread                        audio thread
//!   --------------------                        ------------
//!   SpatialAudioEngine                          AudioRenderer
//!     update_scene_spatial ── triple buffers ──▶  crossfaders (per pair)
//!     connect_pull, add_listener, ... ── SPSC commands ─▶ applied at the start of each block
//!     reap retired DSP state ◀── SPSC garbage ──  (removed / replaced boxes, never freed here)
//! ```
//!
//! Heavy objects (new outputs, listeners, their DSP state) are built on the compute side and
//! shipped in a [`Command`]; objects retired by the audio thread are shipped back and dropped by
//! the compute side, so the audio thread performs no `alloc` / `free`.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use quasar_core::spsc::{SpscConsumer, SpscProducer};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use quasar_dsp::limiter::{set_flush_to_zero, OutputMeter};

use crate::render::{Command, Garbage, SceneRenderState};
use crate::AudioTiming;

/// Capacity of the command ring (compute -> audio).
pub(crate) const COMMAND_QUEUE_CAPACITY: usize = 1024;
/// Capacity of the garbage ring (audio -> compute).
pub(crate) const GARBAGE_QUEUE_CAPACITY: usize = 1024;

/// State shared (lock-free) between the engine and its renderer.
pub(crate) struct RendererShared {
    /// Debug stage selector, see `SpatialAudioEngine::debug_audio_stage`.
    pub(crate) stage: AtomicU8,
    /// Blocks rejected because a buffer exceeded `DEFAULT_BLOCK_SIZE` (output was silenced).
    pub(crate) invalid_blocks: AtomicU64,
    /// Retired boxes the audio thread had to drop itself because the garbage ring was full.
    pub(crate) garbage_overflow: AtomicU64,
}

impl RendererShared {
    pub(crate) fn new() -> Self {
        Self { stage: AtomicU8::new(4), invalid_blocks: AtomicU64::new(0), garbage_overflow: AtomicU64::new(0) }
    }
}

/// Audio-thread renderer. Obtain it from
/// [`SpatialAudioEngine::audio_handle`](crate::SpatialAudioEngine::audio_handle), move it into the
/// audio callback and call [`process_audio_scene`](Self::process_audio_scene) once per block.
///
/// No lock is taken and nothing is allocated or freed in any of its methods (verified by
/// `tests/lockfree_engine_tests.rs`). It is `Send` but not `Sync`.
pub struct AudioRenderer {
    pub(crate) scene: SceneRenderState,
    commands: SpscConsumer<Command>,
    garbage: SpscProducer<Garbage>,
    pub(crate) shared: Arc<RendererShared>,
    timing: Arc<AudioTiming>,
    /// Whether to set FTZ / DAZ on the audio thread at the first block (default true).
    flush_denormals: bool,
    thread_prepared: bool,
}

impl AudioRenderer {
    pub(crate) fn new(
        scene: SceneRenderState,
        commands: SpscConsumer<Command>,
        garbage: SpscProducer<Garbage>,
        shared: Arc<RendererShared>,
        timing: Arc<AudioTiming>,
    ) -> Self {
        Self { scene, commands, garbage, shared, timing, flush_denormals: true, thread_prepared: false }
    }

    /// Prepare the CALLING thread for audio work: enables flush-to-zero / denormals-are-zero
    /// (see [`quasar_dsp::limiter::set_flush_to_zero`]) so decaying tails never hit slow denormal
    /// arithmetic. Called automatically on the first [`process_audio_scene`](Self::process_audio_scene)
    /// unless disabled with [`set_flush_denormals`](Self::set_flush_denormals); the FP mode is
    /// per thread, so call it again if the renderer is moved to another thread. Returns whether
    /// the platform supports it (x86_64, aarch64).
    pub fn prepare_audio_thread(&mut self) -> bool {
        self.thread_prepared = true;
        set_flush_to_zero()
    }

    /// Enable / disable the automatic FTZ / DAZ setup of the audio thread (default on).
    pub fn set_flush_denormals(&mut self, on: bool) {
        self.flush_denormals = on;
    }

    /// Meters of the output stage of listener `index` (renderer-side order).
    pub fn output_meter(&self, index: usize) -> Option<&Arc<OutputMeter>> {
        self.scene.meter(index)
    }

    /// Apply one command (used directly by the combined engine and by the queue drain).
    pub(crate) fn apply(&mut self, cmd: Command) {
        let Self { scene, garbage, shared, .. } = self;
        scene.apply(cmd, &mut |g| {
            if garbage.push(g).is_err() {
                // Ring full (compute side not reaping): drop here rather than block.
                shared.garbage_overflow.fetch_add(1, Ordering::Relaxed);
            }
        });
    }

    /// Apply every queued command. Wait-free.
    pub fn drain_commands(&mut self) {
        while let Some(cmd) = self.commands.pop() {
            self.apply(cmd);
        }
    }

    /// Number of listeners the renderer currently renders (it may briefly lag the engine's
    /// registry while commands are in flight).
    pub fn num_listeners(&self) -> usize {
        self.scene.num_listeners()
    }

    /// Number of scene outputs the renderer currently renders.
    pub fn num_outputs(&self) -> usize {
        self.scene.num_outputs()
    }

    /// Lock-free timing counters of this renderer.
    pub fn timing(&self) -> &AudioTiming {
        &self.timing
    }

    /// Blocks that were silenced because a buffer exceeded `DEFAULT_BLOCK_SIZE`.
    pub fn invalid_block_count(&self) -> u64 {
        self.shared.invalid_blocks.load(Ordering::Relaxed)
    }

    /// Render one block. NEVER blocks, locks, allocates or panics.
    ///
    /// Queued configuration commands are applied first. `listener_outputs` should hold one buffer
    /// per listener; because configuration is asynchronous the renderer's listener count can lag
    /// the engine's by a block, so extra buffers are cleared and missing ones are simply not
    /// rendered, instead of panicking. A buffer longer than `DEFAULT_BLOCK_SIZE` samples silences
    /// the block and increments [`invalid_block_count`](Self::invalid_block_count).
    pub fn process_audio_scene(&mut self, sources: &[&AudioBuffer], listener_outputs: &mut [AudioBuffer]) {
        let t_start = Instant::now();
        if self.flush_denormals && !self.thread_prepared {
            self.prepare_audio_thread();
        }
        self.drain_commands();

        let too_long = sources.iter().any(|s| s.samples() as usize > DEFAULT_BLOCK_SIZE)
            || listener_outputs.iter().any(|l| l.samples() as usize > DEFAULT_BLOCK_SIZE);
        if too_long {
            self.shared.invalid_blocks.fetch_add(1, Ordering::Relaxed);
            for l in listener_outputs.iter_mut() {
                l.clear();
            }
            return;
        }

        let stage = self.shared.stage.load(Ordering::Relaxed);
        let n = self.scene.num_listeners().min(listener_outputs.len());
        let (rendered, rest) = listener_outputs.split_at_mut(n);
        for l in rest.iter_mut() {
            l.clear();
        }
        self.scene.process(sources, rendered, stage);

        // Record timing (relaxed atomics: lock-free, no allocation).
        let ns = t_start.elapsed().as_nanos() as u64;
        let t = &*self.timing;
        t.total_ns.fetch_add(ns, Ordering::Relaxed);
        t.call_count.fetch_add(1, Ordering::Relaxed);
        if ns > t.max_ns.load(Ordering::Relaxed) {
            t.max_ns.store(ns, Ordering::Relaxed);
        }
    }
}
