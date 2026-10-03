use crate::bands::Band8;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Maximum number of spatial audio sources supported.
pub const MAX_SPATIAL_SOURCES: usize = 32;

/// All spatial coefficients needed by the audio thread for one source.
#[derive(Clone, Debug)]
pub struct SpatialCoefficients {
    /// Source identifier.
    pub source_id: u32,
    /// Direct-path linear gain per frequency band.
    pub direct_gain: Band8,
    /// Direct-path fractional delay in samples.
    pub direct_delay_samples: f32,
    /// Source directivity (#74): per-band linear gain of the emitter's radiation pattern toward
    /// this listener (emission angle from the emitter orientation). 1.0 = omnidirectional. It is
    /// a separate factor from [`direct_gain`](Self::direct_gain) (distance, air, occlusion) and the
    /// direct chain applies their product.
    pub directivity_gain: Band8,
    /// Direct-path azimuth in radians (-π = left, 0 = centre, π = right).
    pub direct_azimuth: f32,
    /// Direct-path elevation in radians.
    pub direct_elevation: f32,
    /// Early reflection tap coefficients.
    pub early_reflections: Vec<EarlyReflectionCoeffs>,
    /// Late reverb RT60 per band (seconds).
    pub late_t60: Band8,
    /// Late reverb gain relative to direct (dB).
    pub late_gain_db: f32,
    /// Monotonically increasing version counter for change detection.
    pub version: u64,
}

/// Coefficients for one early reflection tap.
#[derive(Clone, Debug)]
pub struct EarlyReflectionCoeffs {
    /// Azimuth angle of the reflection (radians).
    pub azimuth: f32,
    /// Elevation angle of the reflection (radians).
    pub elevation: f32,
    /// Fractional delay in samples for this tap.
    pub delay_samples: f32,
    /// Per-band linear gain for this tap.
    pub gain: Band8,
}

/// Slot-index mask / "freshly published" flag for the exchange slot.
const SLOT_MASK: u32 = 0b011;
const SLOT_DIRTY: u32 = 0b100;

/// Lock-free triple-buffered parameter exchange for multiple sources.
///
/// Each source has its own set of 3 slots (write / read / staging) and its own
/// set of atomic indices.  The compute thread (producer) calls
/// `begin_write(src)` / `end_write(src)` per source.  The audio thread
/// (consumer) calls `update()` once per block to advance all sources, then
/// `read(src)` to get per-source coefficients.
///
/// No allocations on the hot path after construction.
pub struct ParameterTripleBuffer {
    /// Per-source, per-slot: `[source][slot]`.
    buffers: Vec<[UnsafeCell<SpatialCoefficients>; 3]>,
    /// Per-source write index (producer).
    write_indices: Vec<AtomicU32>,
    /// Per-source read index (consumer).
    read_indices: Vec<AtomicU32>,
    /// Per-source exchange slot: low 2 bits = slot index, `SLOT_DIRTY` set when
    /// the producer has published data the consumer has not taken yet.
    staging_indices: Vec<AtomicU32>,
    /// Global latest version (per-source versions are tracked individually).
    _latest_version: AtomicU64,
}

// SAFETY: The triple-buffer design ensures single-writer, single-reader access
// per source.  The producer and consumer never access the same slot
// simultaneously.
unsafe impl Send for ParameterTripleBuffer {}
unsafe impl Sync for ParameterTripleBuffer {}

impl ParameterTripleBuffer {
    /// Create a new triple buffer with `num_sources` sources, each
    /// initialised to `initial`.
    pub fn new(num_sources: usize, initial: SpatialCoefficients) -> Self {
        let mut buffers = Vec::with_capacity(num_sources);
        let mut write_indices = Vec::with_capacity(num_sources);
        let mut read_indices = Vec::with_capacity(num_sources);
        let mut staging_indices = Vec::with_capacity(num_sources);

        for _ in 0..num_sources {
            buffers.push([
                UnsafeCell::new(initial.clone()),
                UnsafeCell::new(initial.clone()),
                UnsafeCell::new(initial.clone()),
            ]);
            write_indices.push(AtomicU32::new(0));
            read_indices.push(AtomicU32::new(1));
            staging_indices.push(AtomicU32::new(2));
        }

        Self {
            buffers,
            write_indices,
            read_indices,
            staging_indices,
            _latest_version: AtomicU64::new(0),
        }
    }

    /// Number of sources.
    pub fn num_sources(&self) -> usize {
        self.buffers.len()
    }

    /// Begin writing to the producer slot for a given source.
    /// Called from the compute thread.
    ///
    /// # Safety
    /// Must only be called from one thread at a time (the compute thread).
    pub unsafe fn begin_write(&self, source_id: usize) -> &mut SpatialCoefficients {
        let idx = self.write_indices[source_id].load(Ordering::Acquire) as usize;
        &mut *self.buffers[source_id][idx].get()
    }

    /// Finish writing and atomically publish for a given source.
    /// Called from the compute thread.
    pub fn end_write(&self, source_id: usize) {
        let write_idx = self.write_indices[source_id].load(Ordering::Relaxed) as usize;
        let version = self._latest_version.fetch_add(1, Ordering::Release) + 1;
        unsafe {
            (*self.buffers[source_id][write_idx].get()).version = version;
        }
        // Hand our slot to the exchange (marked dirty) and take back whatever
        // slot was parked there. One atomic swap: no window for the consumer.
        let prev = self.staging_indices[source_id]
            .swap(write_idx as u32 | SLOT_DIRTY, Ordering::AcqRel);
        self.write_indices[source_id].store(prev & SLOT_MASK, Ordering::Release);
    }

    /// Advance to the latest published data for all sources.
    /// Called from the audio thread once per block.
    ///
    /// Only takes the exchange slot when the producer has published since the
    /// last call (DIRTY set); otherwise the read slot is left alone, so the
    /// consumer never goes back to older data.
    pub fn update(&self) {
        for src in 0..self.buffers.len() {
            if self.staging_indices[src].load(Ordering::Acquire) & SLOT_DIRTY == 0 {
                continue;
            }
            let read = self.read_indices[src].load(Ordering::Relaxed);
            // Give back our (clean) read slot; the producer can only ever
            // replace the exchange value with another dirty one, so the slot
            // we receive here is always the newest published.
            let taken = self.staging_indices[src].swap(read, Ordering::AcqRel);
            self.read_indices[src].store(taken & SLOT_MASK, Ordering::Release);
        }
    }

    /// Read the current data for a given source.
    /// Called from the audio thread after `update()`.
    ///
    /// # Safety
    /// Must only be called from one thread at a time (the audio thread).
    pub unsafe fn read(&self, source_id: usize) -> &SpatialCoefficients {
        let idx = self.read_indices[source_id].load(Ordering::Acquire) as usize;
        &*self.buffers[source_id][idx].get()
    }

    /// Get the version of the readable slot for a given source (no unsafe access).
    pub fn read_version(&self, source_id: usize) -> u64 {
        let idx = self.read_indices[source_id].load(Ordering::Acquire) as usize;
        unsafe { (*self.buffers[source_id][idx].get()).version }
    }
}
