use crate::bands::Band8;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// All spatial coefficients needed by the audio thread for one source.
#[derive(Clone, Debug)]
pub struct SpatialCoefficients {
    /// Source identifier.
    pub source_id: u32,
    /// Direct-path linear gain per frequency band.
    pub direct_gain: Band8,
    /// Direct-path fractional delay in samples.
    pub direct_delay_samples: f32,
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

/// Lock-free triple-buffered parameter exchange.
///
/// The compute thread (producer) calls `begin_write()` / `end_write()`.
/// The audio thread (consumer) calls `update()` then `read()`.
///
/// No allocations, no mutexes, no blocking.
pub struct ParameterTripleBuffer {
    buffers: [UnsafeCell<SpatialCoefficients>; 3],
    write_index: AtomicU32,
    read_index: AtomicU32,
    staging_index: AtomicU32,
    latest_version: AtomicU64,
}

// SAFETY: The triple-buffer design ensures single-writer, single-reader access.
// The producer and consumer never access the same slot simultaneously.
unsafe impl Send for ParameterTripleBuffer {}
unsafe impl Sync for ParameterTripleBuffer {}

impl ParameterTripleBuffer {
    /// Create a new triple buffer with initial values in all three slots.
    pub fn new(initial: SpatialCoefficients) -> Self {
        Self {
            buffers: [
                UnsafeCell::new(initial.clone()),
                UnsafeCell::new(initial.clone()),
                UnsafeCell::new(initial),
            ],
            write_index: AtomicU32::new(0),
            read_index: AtomicU32::new(1),
            staging_index: AtomicU32::new(2),
            latest_version: AtomicU64::new(0),
        }
    }

    /// Begin writing to the producer slot. Called from the compute thread.
    ///
    /// Returns a mutable reference to the write buffer.
    ///
    /// # Safety
    ///
    /// Must only be called from one thread at a time (the compute thread).
    pub unsafe fn begin_write(&self) -> &mut SpatialCoefficients {
        let idx = self.write_index.load(Ordering::Acquire) as usize;
        &mut *self.buffers[idx].get()
    }

    /// Finish writing and atomically publish. Called from the compute thread.
    ///
    /// Swaps write ↔ staging so the audio thread can pick up new data.
    pub fn end_write(&self) {
        // Bump the version in the write slot first.
        let write_idx = self.write_index.load(Ordering::Relaxed) as usize;
        let version = self.latest_version.fetch_add(1, Ordering::Release) + 1;
        // SAFETY: We are the only writer; the audio thread never accesses the write slot.
        unsafe {
            (*self.buffers[write_idx].get()).version = version;
        }

        // Atomically swap write and staging indices.
        let staging = self.staging_index.swap(write_idx as u32, Ordering::AcqRel);
        self.write_index.store(staging, Ordering::Release);
    }

    /// Advance to the latest published data. Called from the audio thread.
    ///
    /// Swaps read ↔ staging atomically.
    pub fn update(&self) {
        let staging = self.staging_index.load(Ordering::Acquire);
        let read = self.read_index.swap(staging, Ordering::AcqRel);
        self.staging_index.store(read, Ordering::Release);
    }

    /// Read the current data. Called from the audio thread after `update()`.
    ///
    /// Returns an immutable reference to the readable slot.
    ///
    /// # Safety
    ///
    /// Must only be called from one thread at a time (the audio thread).
    pub unsafe fn read(&self) -> &SpatialCoefficients {
        let idx = self.read_index.load(Ordering::Acquire) as usize;
        &*self.buffers[idx].get()
    }

    /// Get the version of the currently readable data.
    pub fn version(&self) -> u64 {
        self.latest_version.load(Ordering::Acquire)
    }
}
