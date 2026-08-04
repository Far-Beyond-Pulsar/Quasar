use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use quasar_core::streaming_source::{StreamingPolicy, StreamingSource};

// ── Low-level disk reader ─────────────────────────────────────────────

/// A streaming WAV file reader that implements `StreamingSource`.
pub struct WaveFileStream {
    reader: hound::WavReader<BufReader<File>>,
    spec: hound::WavSpec,
    total_frames: u64,
    position: u64,
}

impl WaveFileStream {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, hound::Error> {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let total_frames = reader.duration() as u64;
        Ok(Self { reader, spec, total_frames, position: 0 })
    }
}

impl StreamingSource for WaveFileStream {
    fn channels(&self) -> usize { self.spec.channels as usize }
    fn sample_rate(&self) -> u32 { self.spec.sample_rate }
    fn total_frames(&self) -> Option<u64> { Some(self.total_frames) }

    fn read_frames(&mut self, frames: &mut [f32]) -> usize {
        let ch = self.channels();
        let requested = frames.len() / ch;
        let remaining = self.total_frames.saturating_sub(self.position);
        let count = (requested as u64).min(remaining) as usize;
        if count == 0 { return 0; }

        match self.spec.bits_per_sample {
            16 => {
                let mut iter = self.reader.samples::<i16>();
                for i in 0..count {
                    for c in 0..ch {
                        frames[i * ch + c] = match iter.next() {
                            Some(Ok(s)) => s as f32 / i16::MAX as f32,
                            _ => 0.0,
                        };
                    }
                }
            }
            24 => {
                let mut iter = self.reader.samples::<i32>();
                for i in 0..count {
                    for c in 0..ch {
                        frames[i * ch + c] = match iter.next() {
                            Some(Ok(s)) => s as f32 / 8_388_608.0,
                            _ => 0.0,
                        };
                    }
                }
            }
            32 => {
                let mut iter = self.reader.samples::<i32>();
                for i in 0..count {
                    for c in 0..ch {
                        frames[i * ch + c] = match iter.next() {
                            Some(Ok(s)) => s as f32 / i32::MAX as f32,
                            _ => 0.0,
                        };
                    }
                }
            }
            b => panic!("WaveFileStream: unsupported bit depth {b}"),
        }

        self.position += count as u64;
        count
    }

    fn seek_frames(&mut self, frame: u64) {
        let frame = frame.min(self.total_frames);
        if self.reader.seek(frame as u32).is_ok() {
            self.position = frame;
        }
    }
}

// ── Explicit policy declaration ───────────────────────────────────────

/// Wraps any `StreamingSource` to declare its `StreamingPolicy` explicitly.
///
/// Use this when the caller already knows the intended playback pattern
/// (e.g. a background ambience loop) instead of waiting for the `Auto`
/// heuristic to infer it after a few seconds of repeated wraps/rewinds.
pub struct PolicyOverride<S> {
    inner: S,
    policy: StreamingPolicy,
}

impl<S: StreamingSource> PolicyOverride<S> {
    pub fn new(inner: S, policy: StreamingPolicy) -> Self {
        Self { inner, policy }
    }
}

impl<S: StreamingSource> StreamingSource for PolicyOverride<S> {
    fn channels(&self) -> usize { self.inner.channels() }
    fn sample_rate(&self) -> u32 { self.inner.sample_rate() }
    fn total_frames(&self) -> Option<u64> { self.inner.total_frames() }
    fn read_frames(&mut self, frames: &mut [f32]) -> usize { self.inner.read_frames(frames) }
    fn seek_frames(&mut self, frame: u64) { self.inner.seek_frames(frame) }
    fn policy(&self) -> StreamingPolicy { self.policy }
}

// ── Policy bit-packing ────────────────────────────────────────────────

/// Encode a `StreamingPolicy` + promotion flag into a `u64` for atomic storage.
fn policy_to_u64(p: StreamingPolicy, promoted: bool) -> u64 {
    let base = match p {
        StreamingPolicy::Common => 0u64,
        StreamingPolicy::Once => 1,
        StreamingPolicy::Auto => 2,
    };
    if promoted { base | 4 } else { base }
}

fn u64_to_policy(v: u64) -> (StreamingPolicy, bool) {
    let p = match v & 3 {
        0 => StreamingPolicy::Common,
        1 => StreamingPolicy::Once,
        _ => StreamingPolicy::Auto,
    };
    (p, v & 4 != 0)
}

// ── Ring buffer ───────────────────────────────────────────────────────

const DEFAULT_RING_CAPACITY_FRAMES: usize = 40960;
const IO_CHUNK_FRAMES: usize = 4096;
const HEURISTIC_WINDOW_SECS: f64 = 10.0;

/// Growable ring-buffer contents: `cap` frames × `ch` interleaved samples.
///
/// The whole struct is behind an `Arc` so that promoting a source to
/// `Common` (caching the full file) can atomically swap in a larger buffer.
struct RingData {
    data: Vec<AtomicU32>,
    cap: usize,
}

struct RingBuf {
    /// Current buffer.  Readers clone the `Arc` (brief lock, never blocks
    /// for data access); the writer holds the lock only to clone or swap.
    storage: Mutex<Arc<RingData>>,
    ch: usize,
    write_frame: AtomicU64,
    read_frame: AtomicU64,
    seek_to: AtomicU64,
    total_frames: u64,
    policy: AtomicU64,
    wrap_count: AtomicU64,
    stop: AtomicBool,
    total_read: AtomicU64,
    /// True once the whole file is resident and the writer serves every
    /// loop from memory without touching disk.
    cached: AtomicBool,
}

impl RingBuf {
    fn new(ch: usize, total_frames: u64, policy: StreamingPolicy) -> Self {
        // Memory management: a `Common` source is cached in full — the ring
        // is sized to the entire file so every loop is served from RAM.
        // `Once` / `Auto` stream through a fixed small window instead.
        let cap = if policy == StreamingPolicy::Common && total_frames != u64::MAX {
            total_frames as usize
        } else {
            DEFAULT_RING_CAPACITY_FRAMES.max(IO_CHUNK_FRAMES * 2)
        };
        Self {
            storage: Mutex::new(Arc::new(RingData {
                data: (0..cap * ch).map(|_| AtomicU32::new(0)).collect(),
                cap,
            })),
            ch,
            write_frame: AtomicU64::new(0),
            read_frame: AtomicU64::new(0),
            seek_to: AtomicU64::new(0),
            total_frames,
            policy: AtomicU64::new(policy_to_u64(policy, false)),
            wrap_count: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            total_read: AtomicU64::new(0),
            cached: AtomicBool::new(false),
        }
    }

    /// Writer-side snapshot: brief lock, only to clone the `Arc`.
    fn snapshot(&self) -> Arc<RingData> {
        self.storage.lock().unwrap().clone()
    }

    /// Reader-side snapshot that never blocks.  Returns `None` only if the
    /// writer is mid-swap (a few nanoseconds, once per promotion).
    fn try_snapshot(&self) -> Option<Arc<RingData>> {
        self.storage.try_lock().ok().map(|g| g.clone())
    }

    /// Write interleaved frames at a cumulative source-frame offset.  Data
    /// access is lock-free (atomics); the lock is only taken to clone the
    /// current buffer `Arc`.
    fn write(&self, src_frame: u64, frames: &[f32]) {
        let rd = self.snapshot();
        let n = frames.len() / self.ch;
        for i in 0..n {
            let idx = ((src_frame + i as u64) as usize % rd.cap) * self.ch;
            for c in 0..self.ch {
                rd.data[idx + c].store(f32::to_bits(frames[i * self.ch + c]), Ordering::Relaxed);
            }
        }
    }

    fn cap(&self) -> usize {
        self.snapshot().cap
    }

    fn status(&self) -> (u64, u64, u64, usize, u64, u64, bool) {
        let wf = self.write_frame.load(Ordering::Acquire);
        let rf = self.read_frame.load(Ordering::Acquire);
        let cap = self.cap();
        let (policy, promoted) = u64_to_policy(self.policy.load(Ordering::Acquire));
        let wc = self.wrap_count.load(Ordering::Acquire);
        (wf, rf, self.total_frames, cap, wc, policy as u64, promoted)
    }
}

// ── Cache the whole source in memory ──────────────────────────────────

/// Read the entire source into a fresh full-size buffer and atomically swap
/// it in, switching the ring to cached mode.  After this the I/O thread never
/// reads from disk again — every loop is served from RAM.
fn cache_source(source: &mut Box<dyn StreamingSource>, ring: &Arc<RingBuf>) {
    let total = ring.total_frames;
    if total == u64::MAX {
        return; // unknown length — cannot pre-size a cache
    }
    source.seek_frames(0);
    let ch = ring.ch;
    let t = total as usize;
    let new_data = RingData {
        data: (0..t * ch).map(|_| AtomicU32::new(0)).collect(),
        cap: t,
    };
    let mut buf = vec![0.0_f32; IO_CHUNK_FRAMES * ch];
    let mut pos = 0usize;
    loop {
        let n = source.read_frames(&mut buf);
        if n == 0 { break; }
        for i in 0..n {
            for c in 0..ch {
                new_data.data[(pos + i) * ch + c]
                    .store(f32::to_bits(buf[i * ch + c]), Ordering::Relaxed);
            }
        }
        pos += n;
        if pos >= t { break; }
    }
    // Swap in one shot; the old buffer keeps serving readers until then.
    {
        let mut g = ring.storage.lock().unwrap();
        *g = Arc::new(new_data);
    }
    let rf = ring.read_frame.load(Ordering::Acquire);
    ring.write_frame.store(rf.wrapping_add(total), Ordering::Release);
    ring.cached.store(true, Ordering::Release);
    ring.policy.store(policy_to_u64(StreamingPolicy::Auto, true), Ordering::Release);
    eprintln!("[quasar-stream] cached {total} frames in memory");
}

// ── Background I/O thread ─────────────────────────────────────────────

fn io_thread(mut source: Box<dyn StreamingSource>, ring: Arc<RingBuf>) {
    let ch = ring.ch;
    let mut buf = vec![0.0_f32; IO_CHUNK_FRAMES * ch];
    let mut first_wrap: Option<Instant> = None;
    let mut first_rewind: Option<Instant> = None;

    let policy_now = || {
        let (p, promoted) = u64_to_policy(ring.policy.load(Ordering::Acquire));
        (p, promoted)
    };

    loop {
        if ring.stop.load(Ordering::Acquire) {
            break;
        }

        // ── Pending seek ────────────────────────────────────────────
        let seek = ring.seek_to.swap(0, Ordering::Acquire);
        if seek > 0 {
            let target = seek - 1;
            let prev_wf = ring.write_frame.swap(target, Ordering::Release);
            ring.total_read.store(target, Ordering::Relaxed);

            if ring.cached.load(Ordering::Acquire) {
                // Data is already resident: seeks are pure random access.
                continue;
            }
            source.seek_frames(target);

            // Detect rewind (consumer jumped backward) — heuristic for Auto.
            if policy_now().0 == StreamingPolicy::Auto && target < prev_wf {
                let now = Instant::now();
                if let Some(t0) = first_rewind {
                    if (now - t0).as_secs_f64() < HEURISTIC_WINDOW_SECS {
                        eprintln!("[quasar-stream] ⤶ rewind within window — caching source");
                        cache_source(&mut source, &ring);
                    } else {
                        first_rewind = Some(now);
                    }
                } else {
                    first_rewind = Some(now);
                }
            }
            continue;
        }

        // ── Cached mode: everything is resident, never touch disk ───
        if ring.cached.load(Ordering::Acquire) {
            let rf = ring.read_frame.load(Ordering::Acquire);
            let wf = ring.write_frame.load(Ordering::Acquire);
            // Keep the writer one full pass ahead of the consumer so the
            // reader's ahead/behind guards never trigger.  The data at
            // `frame % total` never changes between loops, so no re-read.
            let margin = wf.checked_sub(rf).unwrap_or(0);
            if margin < IO_CHUNK_FRAMES as u64 {
                let total = ring.total_frames;
                if total != u64::MAX {
                    ring.write_frame.store(rf.wrapping_add(total), Ordering::Release);
                }
            }
            thread::sleep(std::time::Duration::from_millis(1));
            continue;
        }

        // ── Streamed mode: read chunks from disk ────────────────────
        let wf = ring.write_frame.load(Ordering::Acquire);
        let rf = ring.read_frame.load(Ordering::Acquire);
        let buffered = wf.checked_sub(rf).unwrap_or(0);
        let available = ring.cap().saturating_sub(buffered as usize);
        if available == 0 {
            thread::sleep(std::time::Duration::from_millis(1));
            continue;
        }

        // Never request more than we can store (also handles rings smaller
        // than one IO chunk, e.g. a tiny cached Common file).
        let want = available.min(IO_CHUNK_FRAMES);
        let count = source.read_frames(&mut buf[..want * ch]);
        if count == 0 {
            ring.wrap_count.fetch_add(1, Ordering::Relaxed);
            let (p, promoted) = policy_now();
            match p {
                StreamingPolicy::Common => {
                    // The whole file just finished its first pass into a
                    // full-size ring: it is now resident.  Serve all future
                    // loops purely from memory.
                    let total = ring.total_frames;
                    if total != u64::MAX {
                        let rf2 = ring.read_frame.load(Ordering::Acquire);
                        ring.write_frame.store(rf2.wrapping_add(total), Ordering::Release);
                        ring.cached.store(true, Ordering::Release);
                        eprintln!("[quasar-stream] cached {total} frames in memory");
                    } else {
                        source.seek_frames(0);
                    }
                }
                StreamingPolicy::Once => {
                    // Loop from disk every pass; never cached.
                    source.seek_frames(0);
                }
                StreamingPolicy::Auto => {
                    if promoted {
                        let total = ring.total_frames;
                        if total != u64::MAX {
                            cache_source(&mut source, &ring);
                        } else {
                            source.seek_frames(0); // unknown length, keep streaming
                        }
                    } else {
                        // Heuristic: promote to Common after the second wrap
                        // within the window (frequently re-read source).
                        let now = Instant::now();
                        if let Some(t0) = first_wrap {
                            if (now - t0).as_secs_f64() < HEURISTIC_WINDOW_SECS {
                                eprintln!("[quasar-stream] ↻ repeated re-read — promoting to Common (caching)");
                                ring.policy.store(policy_to_u64(StreamingPolicy::Auto, true), Ordering::Release);
                                continue; // next iteration takes the promoted path
                            }
                            first_wrap = Some(now);
                        } else {
                            first_wrap = Some(now);
                        }
                        source.seek_frames(0); // keep looping from disk meanwhile
                    }
                }
            }
            continue;
        }

        ring.write(wf, &buf[..count * ch]);
        ring.write_frame.store(wf + count as u64, Ordering::Release);
        ring.total_read.fetch_add(count as u64, Ordering::Relaxed);
    }
}

// ── Thread-safe streaming source (consumer side) ──────────────────────

/// Threaded ring-buffered streaming source.
///
/// All disk I/O runs on a background thread.  The consumer (audio callback)
/// reads from the ring buffer and never blocks.
///
/// Memory management is controlled by `StreamingPolicy`:
/// - `Common` — the whole file is read once and cached in RAM; later loops
///   are served from memory and the disk is never touched again.
/// - `Once` — never cached; every loop re-reads from disk through a small
///   ring window.
/// - `Auto` — starts streamed like `Once`; if the consumer re-reads the
///   source repeatedly within a heuristic window it is promoted to `Common`
///   and the file is cached in memory.
pub struct BufferedStream {
    ring: Arc<RingBuf>,
    _handle: Option<JoinHandle<()>>,
    channels: usize,
    sample_rate: u32,
    total_frames: u64,
    read_frame: u64,
}

impl BufferedStream {
    /// Wrap `source` with a background I/O thread and ring buffer.
    /// The source's `policy()` method controls memory management.
    pub fn new(source: Box<dyn StreamingSource>) -> Self {
        let ch = source.channels();
        let sr = source.sample_rate();
        let total = source.total_frames().unwrap_or(u64::MAX);
        let policy = source.policy();

        let ring = Arc::new(RingBuf::new(ch, total, policy));

        let ring_clone = ring.clone();
        let handle = thread::Builder::new()
            .name("quasar-stream-io".into())
            .spawn(move || io_thread(source, ring_clone))
            .expect("spawn streaming I/O thread");

        // Status-logging thread.
        let ring_stat = ring.clone();
        thread::Builder::new()
            .name("quasar-stream-status".into())
            .spawn(move || loop {
                thread::sleep(std::time::Duration::from_secs(2));
                if ring_stat.stop.load(Ordering::Acquire) { break; }
                let (wf, rf, total, cap, _wraps, pol, promoted) = ring_stat.status();
                let (policy, _) = u64_to_policy(pol);
                let cached = ring_stat.cached.load(Ordering::Acquire);
                let loop_num = if total > 0 { rf / total } else { 0 };
                let pos_in_loop = if total > 0 { rf % total } else { rf };
                let ahead = wf.wrapping_sub(rf).min(cap as u64);
                let pct = if total > 0 {
                    pos_in_loop as f64 / total as f64 * 100.0
                } else { 0.0 };

                let pol_label = match policy {
                    StreamingPolicy::Common => "common",
                    StreamingPolicy::Once => "once",
                    StreamingPolicy::Auto => if promoted { "auto→common" } else { "auto" },
                };
                let mem_label = if cached { "cached" } else { "stream" };
                eprintln!(
                    "[quasar-stream] loop={} pos={}/{} ({:.1}%) ahead={} buffer={}/{} mem={} policy={}",
                    loop_num, pos_in_loop, total, pct, ahead, cap, cap, mem_label, pol_label,
                );
            })
            .expect("spawn streaming status thread");

        Self {
            ring,
            _handle: Some(handle),
            channels: ch,
            sample_rate: sr,
            total_frames: total,
            read_frame: 0,
        }
    }

    /// True once the whole file is resident in memory and the I/O thread is
    /// serving every loop from RAM (no further disk reads).
    pub fn is_cached(&self) -> bool {
        self.ring.cached.load(Ordering::Acquire)
    }

    /// Read a single sample at a cumulative source-frame index (non-blocking).
    ///
    /// `frame` is in the cumulative domain (monotonically increasing, same
    /// unit as `write_frame`).  The ring buffer stores data modulo its
    /// capacity; this method translates using `frame % cap`.  Returns 0 if
    /// the frame is too far ahead of the writer or has already been
    /// overwritten.
    ///
    /// A grace window of one IO chunk allows the consumer to read slightly
    /// past `write_frame` during the EOF-transient.
    pub fn sample_at(&self, frame: u64, channel: usize) -> f32 {
        let Some(rd) = self.ring.try_snapshot() else {
            return 0.0; // writer is mid-swap (once per promotion)
        };
        let wf = self.ring.write_frame.load(Ordering::Acquire);
        if frame >= wf {
            // Slightly ahead of the writer — transient at EOF boundary.
            if frame - wf > IO_CHUNK_FRAMES as u64 {
                return 0.0;
            }
        } else {
            let dist = wf - frame;
            if dist > rd.cap as u64 {
                return 0.0;
            }
        }
        let idx = (frame as usize % rd.cap) * self.channels + channel;
        f32::from_bits(rd.data[idx].load(Ordering::Relaxed))
    }

    /// Advance the read cursor after consuming `frames` source-frames.
    ///
    /// Both cursors grow monotonically.  The ring buffer's modulo addressing
    /// and the grace window in `sample_at` handle the EOF transient without
    /// needing seek signals.
    pub fn advance_read(&mut self, frames: u64) {
        self.read_frame += frames;
        self.ring.read_frame.store(self.read_frame, Ordering::Release);
    }

    pub fn channels(&self) -> usize { self.channels }
    pub fn sample_rate(&self) -> u32 { self.sample_rate }
    pub fn total_frames(&self) -> u64 { self.total_frames }
}

impl StreamingSource for BufferedStream {
    fn channels(&self) -> usize { self.channels }
    fn sample_rate(&self) -> u32 { self.sample_rate }
    fn total_frames(&self) -> Option<u64> { Some(self.total_frames) }

    fn read_frames(&mut self, frames: &mut [f32]) -> usize {
        let ch = self.channels;
        let requested = frames.len() / ch;
        let Some(rd) = self.ring.try_snapshot() else {
            return 0;
        };
        let wf = self.ring.write_frame.load(Ordering::Acquire);
        let available = wf.saturating_sub(self.read_frame) as usize;
        let count = requested.min(available);
        if count == 0 { return 0; }
        for i in 0..count {
            let idx = ((self.read_frame + i as u64) as usize % rd.cap) * ch;
            for c in 0..ch {
                let bits = rd.data[idx + c].load(Ordering::Relaxed);
                frames[i * ch + c] = f32::from_bits(bits);
            }
        }
        self.read_frame += count as u64;
        self.ring.read_frame.store(self.read_frame, Ordering::Release);
        count
    }

    fn seek_frames(&mut self, _frame: u64) {}
    fn policy(&self) -> StreamingPolicy {
        let (p, _) = u64_to_policy(self.ring.policy.load(Ordering::Acquire));
        p
    }
}

impl Drop for RingBuf {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_wav(path: &str, samples: &[i16], channels: u16, sample_rate: u32) {
        let spec = hound::WavSpec { channels, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples { writer.write_sample(s).unwrap(); }
        writer.finalize().unwrap();
    }

    #[test]
    fn reads_full_stream() {
        let path = std::env::temp_dir().join("quasar_test_stream_full.wav");
        let p = path.to_str().unwrap().to_string();
        let samples: Vec<i16> = (0..100).collect();
        write_test_wav(&p, &samples, 1, 44100);
        let mut stream = WaveFileStream::open(&path).unwrap();
        assert_eq!(stream.channels(), 1);
        assert_eq!(stream.sample_rate(), 44100);
        assert_eq!(stream.total_frames(), Some(100));
        let mut buf = vec![0.0_f32; 50];
        let n = stream.read_frames(&mut buf);
        assert_eq!(n, 50);
        for i in 0..50 {
            assert!((buf[i] - (i as f32 / i16::MAX as f32)).abs() < 1e-6);
        }
        let n = stream.read_frames(&mut buf);
        assert_eq!(n, 50);
        let n = stream.read_frames(&mut buf);
        assert_eq!(n, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn seek_and_loop() {
        let path = std::env::temp_dir().join("quasar_test_stream_seek.wav");
        let p = path.to_str().unwrap().to_string();
        let samples: Vec<i16> = (0..100).collect();
        write_test_wav(&p, &samples, 1, 48000);
        let mut stream = WaveFileStream::open(&path).unwrap();
        let mut buf = vec![0.0_f32; 10];
        let n = stream.read_frames(&mut buf);
        assert_eq!(n, 10);
        stream.seek_frames(0);
        let n = stream.read_frames(&mut buf);
        assert_eq!(n, 10);
        assert!((buf[0] - 0.0).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn threaded_buffered_read() {
        let path = std::env::temp_dir().join("quasar_test_buffered.wav");
        let p = path.to_str().unwrap().to_string();
        let samples: Vec<i16> = (0..500).map(|i| i as i16).collect();
        write_test_wav(&p, &samples, 1, 48000);
        let source: Box<dyn StreamingSource> = Box::new(WaveFileStream::open(&path).unwrap());
        let mut buf = BufferedStream::new(source);
        thread::sleep(std::time::Duration::from_millis(50));
        let mut out = vec![0.0_f32; 500];
        let mut pos = 0;
        while pos < 500 {
            let n = buf.read_frames(&mut out[pos..]);
            if n == 0 { thread::sleep(std::time::Duration::from_millis(10)); continue; }
            pos += n;
        }
        for i in 0..500 {
            assert!((out[i] - (i as f32 / i16::MAX as f32)).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn once_policy_loops_from_disk() {
        // `Once` means "never cache" — the source still loops (behavior is
        // not tied to the policy), but every pass is re-read from disk and
        // memory never grows to hold the whole file.
        let path = std::env::temp_dir().join("quasar_test_once.wav");
        let p = path.to_str().unwrap().to_string();
        let samples: Vec<i16> = (0..50).collect();
        write_test_wav(&p, &samples, 1, 48000);
        let source: Box<dyn StreamingSource> = Box::new(PolicyOverride::new(
            WaveFileStream::open(&path).unwrap(),
            StreamingPolicy::Once,
        ));
        let buf = BufferedStream::new(source);
        thread::sleep(std::time::Duration::from_millis(100));
        assert!(!buf.is_cached(), "Once sources must not be cached");

        // Read across 5 loops (250 frames) using sample_at like the demo.
        let mut silent = 0;
        for f in 0..250u64 {
            let mut spins = 0;
            let v = loop {
                let v = buf.sample_at(f, 0);
                if v != 0.0 || f % 50 == 0 { break v; }
                spins += 1;
                if spins > 200 { break v; }
                thread::sleep(std::time::Duration::from_millis(2));
            };
            if v == 0.0 && f % 50 != 0 { silent += 1; }
        }
        assert!(silent < 25, "too many silent samples: {silent}/250");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn common_policy_caches_and_survives_many_loops() {
        // Regression test: an explicitly-declared Common source must be
        // cached in memory and keep producing correct audio across many
        // wraps, using `sample_at` random access exactly like the demo's
        // resampler does.
        let path = std::env::temp_dir().join("quasar_test_common_loops.wav");
        let p = path.to_str().unwrap().to_string();
        let samples: Vec<i16> = (0..200).collect();
        write_test_wav(&p, &samples, 1, 48000);
        let source: Box<dyn StreamingSource> = Box::new(PolicyOverride::new(
            WaveFileStream::open(&path).unwrap(),
            StreamingPolicy::Common,
        ));
        let mut buf = BufferedStream::new(source);
        thread::sleep(std::time::Duration::from_millis(50));

        // Walk cumulative frames across 5 full loops (1000 frames), using
        // sample_at + advance_read exactly like the audio callback.  The
        // reader advancing is what lets the writer discover EOF and cache.
        let mut silent_after_first_loop = 0;
        for step in 0..1000u64 {
            let mut spins = 0;
            let val = loop {
                let v = buf.sample_at(step, 0);
                if v != 0.0 || step % 200 == 0 {
                    break v;
                }
                spins += 1;
                if spins > 200 {
                    break v; // give up waiting, record whatever we got
                }
                thread::sleep(std::time::Duration::from_millis(2));
            };
            if step >= 200 && val == 0.0 && step % 200 != 0 {
                silent_after_first_loop += 1;
            }
            buf.advance_read(1);
        }
        assert!(
            buf.is_cached(),
            "Common sources must be cached in memory after the first full pass"
        );
        assert!(
            silent_after_first_loop < 50,
            "too many silent samples after first loop: {silent_after_first_loop}/800"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn auto_policy_promotes_to_cache() {
        // `Auto` starts streamed; after repeated wraps within the heuristic
        // window it promotes to Common and the whole file becomes resident.
        let path = std::env::temp_dir().join("quasar_test_auto_promote.wav");
        let p = path.to_str().unwrap().to_string();
        let samples: Vec<i16> = (0..200).collect();
        write_test_wav(&p, &samples, 1, 48000);
        let source: Box<dyn StreamingSource> = Box::new(WaveFileStream::open(&path).unwrap());
        let mut buf = BufferedStream::new(source);
        thread::sleep(std::time::Duration::from_millis(50));

        // Drive several loops so the writer hits EOF repeatedly and the
        // heuristic promotes the source to a cached Common.
        for step in 0..1000u64 {
            let mut spins = 0;
            let _ = loop {
                let v = buf.sample_at(step, 0);
                if v != 0.0 || step % 200 == 0 { break v; }
                spins += 1;
                if spins > 200 { break 0.0; }
                thread::sleep(std::time::Duration::from_millis(2));
            };
            buf.advance_read(1);
        }
        thread::sleep(std::time::Duration::from_millis(100));
        assert!(
            buf.is_cached(),
            "Auto source should promote to a cached Common after repeated re-reads"
        );
        let _ = std::fs::remove_file(&path);
    }
}
