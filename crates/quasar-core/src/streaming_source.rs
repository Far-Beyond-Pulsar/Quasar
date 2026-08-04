/// Memory-management strategy for a streaming source.
///
/// This controls **how much of the source is retained in RAM**, not playback
/// behavior: looping/replay always works regardless of policy (re-reads are
/// served from memory when cached, or from disk when streamed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamingPolicy {
    /// Source is accessed continuously (e.g. looping background ambience,
    /// frequently-played SFX).  The whole file is read from disk once and
    /// cached in RAM; every subsequent pass is served from memory and the
    /// disk is never touched again.
    Common,
    /// Source is unlikely to be re-read (e.g. a one-shot SFX).  Never
    /// cached: it streams through a small ring window and each pass is
    /// re-read from disk.  Memory stays bounded regardless of file size.
    Once,
    /// Start streamed like `Once`; if the consumer re-reads the source
    /// within a heuristic window, promote to `Common` (cache the whole
    /// file in memory).
    Auto,
}

/// A streaming audio source that delivers raw interleaved audio frames on demand.
///
/// This is the fundamental abstraction for large-file streaming.  Implementors
/// read audio data from disk, network, or synthesis in small chunks rather than
/// loading the entire source into memory.
///
/// # Contract
///
/// - `read_frames` must write interleaved `[L, R, L, R, …]` samples.
/// - Multiple calls to `read_frames` advance an internal cursor; there is no
///   random-access requirement beyond `seek_frames`.
/// - All methods must be cheap and non-blocking where possible — the trait is
///   called from the audio callback on many implementations.
pub trait StreamingSource: Send {
    /// Number of audio channels in this source.
    fn channels(&self) -> usize;

    /// Native sample rate of the source in Hz.
    fn sample_rate(&self) -> u32;

    /// Read the next block of interleaved audio frames.
    ///
    /// `frames` is a flat slice of interleaved samples (capacity
    /// `channels × requested_frames`).  Returns the number of complete
    /// frames actually written, which may be less than `requested_frames`
    /// only when the source is exhausted.  Returns 0 to signal EOF.
    fn read_frames(&mut self, frames: &mut [f32]) -> usize;

    /// Seek to an absolute frame index in the source's sample-rate domain.
    ///
    /// After a successful seek, the next call to `read_frames` will begin
    /// at `frame`.  Seeking past the end of the source is a no-op.
    fn seek_frames(&mut self, frame: u64);

    /// Total number of frames in the source, if known (`None` for live streams).
    fn total_frames(&self) -> Option<u64>;

    /// Declared streaming policy for this source.
    ///
    /// The default is `Auto` — the I/O thread will heuristically promote
    /// to `Common` if the source is re-read frequently.
    fn policy(&self) -> StreamingPolicy {
        StreamingPolicy::Auto
    }
}
