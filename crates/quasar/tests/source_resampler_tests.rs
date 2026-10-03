//! #76: source-side resampling adapter (`ResampledSource`) and its ring-fill drift controller.
//!
//! * Drift: a simulated producer whose clock is off by +-100 / +-1000 ppm feeds a ring that the
//!   device-clock consumer drains; with the controller on, the ring never underruns or overruns
//!   over a long run and the applied trim stays bounded and converges to the clock offset.
//! * Looping: a looping `BufferedStream` (real WAV in memory, `Common` policy as the demo) is
//!   resampled 44.1 -> 48 kHz across several loop wraps and compared with the ideal continuous
//!   tone: no discontinuity at the seam.
//! * Real-time safety: `render` makes no heap allocation.

mod common;

use common::count_allocs;
use quasar_audio::source_resampler::{DriftController, FrameSource, ResampledSource};
use std::f64::consts::PI;

const TONE_HZ: f64 = 1000.0;

/// Ring model with a drifting producer clock. Content is a continuous tone `sin(2 pi f n / rate)`.
struct SimRing {
    rate: f64,
    written: u64,
    read: u64,
    /// Fractional producer position in frames.
    acc: f64,
    /// Producer writes in chunks of this many frames (like an I/O thread). Deliberately NOT a
    /// multiple of the 256-frame block, so the fill level is a sawtooth the controller must average.
    chunk: u64,
}

impl SimRing {
    fn new(rate: f64, _cap: u64, prefill: u64) -> Self {
        Self { rate, written: prefill, read: 0, acc: prefill as f64, chunk: 1000 }
    }
    /// Advance the producer by `secs` of DEVICE time at a clock `ppm` parts per million fast.
    fn produce(&mut self, secs: f64, ppm: f64) {
        self.acc += secs * self.rate * (1.0 + ppm * 1e-6);
        self.written = (self.acc as u64 / self.chunk) * self.chunk;
    }
}

impl FrameSource for SimRing {
    fn channels(&self) -> usize {
        1
    }
    fn sample_rate(&self) -> u32 {
        self.rate as u32
    }
    fn written_frames(&self) -> u64 {
        self.written
    }
    fn read_cursor(&self) -> u64 {
        self.read
    }
    fn read_block(&self, start: u64, _ch: usize, dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = (2.0 * PI * TONE_HZ * (start + i as u64) as f64 / self.rate).sin() as f32 * 0.5;
        }
    }
    fn advance_read(&mut self, frames: u64) {
        self.read += frames;
    }
}

struct DriftOutcome {
    underruns: u64,
    min_fill: u64,
    max_fill: u64,
    max_abs_trim: f64,
    final_trim: f64,
}

fn run_drift(in_rate: f64, out_rate: f64, ppm: f64, secs: f64) -> DriftOutcome {
    const CAP: u64 = 8192;
    const TARGET: u64 = 4096;
    let ring = SimRing::new(in_rate, CAP, TARGET);
    let mut src = ResampledSource::new(ring, out_rate).unwrap();
    src.enable_drift_control(TARGET as f64);
    let block = 256usize;
    let mut buf = vec![0.0f32; block];
    let blocks = (secs * out_rate / block as f64) as usize;
    let (mut min_fill, mut max_fill, mut max_abs_trim) = (u64::MAX, 0u64, 0.0f64);
    let (mut tail_sum, mut tail_n) = (0.0f64, 0usize);
    for b in 0..blocks {
        src.source_mut().produce(block as f64 / out_rate, ppm);
        let n = src.render(&mut [&mut buf[..]]);
        assert_eq!(n, block, "underrun at block {b}");
        let fill = src.fill_frames();
        min_fill = min_fill.min(fill);
        max_fill = max_fill.max(fill);
        max_abs_trim = max_abs_trim.max(src.applied_trim().abs());
        if b >= blocks - blocks / 10 {
            tail_sum += src.applied_trim();
            tail_n += 1;
        }
        // An overrun is the writer lapping the reader: fill beyond the ring capacity.
        assert!(fill <= CAP, "ring overrun at block {b}: fill {fill}");
    }
    // Mean trim over the last 10 percent of the run.
    DriftOutcome { underruns: src.stats().underruns, min_fill, max_fill, max_abs_trim, final_trim: tail_sum / tail_n.max(1) as f64 }
}

fn check_drift(in_rate: f64, out_rate: f64, ppm: f64, secs: f64) {
    let o = run_drift(in_rate, out_rate, ppm, secs);
    let d = ppm * 1e-6;
    eprintln!(
        "drift {ppm:+} ppm {in_rate}->{out_rate} over {secs}s: fill {}..{}, max|trim| {:.1} ppm, mean final-10% {:.1} ppm, underruns {}",
        o.min_fill,
        o.max_fill,
        o.max_abs_trim * 1e6,
        o.final_trim * 1e6,
        o.underruns
    );
    assert_eq!(o.underruns, 0);
    assert!(o.min_fill >= 1024, "fill came too close to underrun: {}", o.min_fill);
    assert!(o.max_fill <= 7168, "fill came too close to overrun: {}", o.max_fill);
    // Bounded: never beyond the controller limit, and not wildly past the true offset.
    assert!(o.max_abs_trim <= 5000e-6 + 1e-9);
    assert!(o.max_abs_trim <= 2.0 * d.abs() + 1000e-6, "trim overshoot {:.1} ppm", o.max_abs_trim * 1e6);
    // The trim converges to the clock offset.
    assert!((o.final_trim - d).abs() <= 0.2 * d.abs() + 15e-6, "final trim {:.1} ppm vs offset {ppm}", o.final_trim * 1e6);
}

#[test]
fn drift_plus_minus_100_ppm_same_rate() {
    check_drift(48_000.0, 48_000.0, 100.0, 400.0);
    check_drift(48_000.0, 48_000.0, -100.0, 400.0);
}

#[test]
fn drift_plus_minus_1000_ppm_same_rate() {
    check_drift(48_000.0, 48_000.0, 1000.0, 400.0);
    check_drift(48_000.0, 48_000.0, -1000.0, 400.0);
}

#[test]
fn drift_with_rate_conversion() {
    // 44.1 kHz source on a 48 kHz device with the producer 300 ppm slow.
    check_drift(44_100.0, 48_000.0, -300.0, 200.0);
}

#[test]
fn without_a_controller_a_drifting_source_underruns_or_overruns() {
    // Control experiment: proves the drift test above is not vacuous.
    let ring = SimRing::new(48_000.0, 8192, 4096);
    let mut src = ResampledSource::new(ring, 48_000.0).unwrap();
    let mut buf = vec![0.0f32; 256];
    let mut bad = false;
    for _ in 0..(1000.0 * 48_000.0 / 256.0) as usize {
        src.source_mut().produce(256.0 / 48_000.0, -1000.0);
        src.render(&mut [&mut buf[..]]);
        if src.stats().underruns > 0 {
            bad = true;
            break;
        }
    }
    assert!(bad, "-1000 ppm for 1000 s must drain a 4096-frame cushion without control");
}

#[test]
fn controller_is_bounded_and_signed_correctly() {
    let mut c = DriftController::new(1000.0);
    // Ring far too full: consume faster (positive), but clamped.
    for _ in 0..1000 {
        let t = c.update(1_000_000.0, 0.005, 48_000.0);
        assert!(t > 0.0 && t <= 5000e-6 + 1e-12);
    }
    let mut c = DriftController::new(1000.0);
    for _ in 0..1000 {
        let t = c.update(0.0, 0.005, 48_000.0);
        assert!(t < 0.0 && t >= -5000e-6 - 1e-12);
    }
}

#[test]
fn render_never_allocates() {
    let ring = SimRing::new(44_100.0, 8192, 4096);
    let mut src = ResampledSource::new(ring, 48_000.0).unwrap();
    src.enable_drift_control(4096.0);
    let mut buf = vec![0.0f32; 256];
    // Warm-up, then measure.
    for _ in 0..4 {
        src.source_mut().produce(256.0 / 48_000.0, 0.0);
        src.render(&mut [&mut buf[..]]);
    }
    let ((), n) = count_allocs(|| {
        for _ in 0..200 {
            src.source_mut().produce(256.0 / 48_000.0, 50.0);
            src.render(&mut [&mut buf[..]]);
        }
    });
    assert_eq!(n, 0, "ResampledSource::render allocated {n} times");
}

#[cfg(feature = "streaming")]
mod looping {
    use super::*;
    use quasar_audio::streaming_source::{BufferedStream, PolicyOverride, WaveFileStream};
    use quasar_core::streaming_source::StreamingPolicy;
    use std::time::{Duration, Instant};

    const FILE_RATE: u32 = 44_100;
    const LOOP_FRAMES: u32 = 4400;
    /// Exactly 40 cycles per loop (441 Hz at 44.1 kHz): the file itself is seamless when looped.
    const CYCLES: f64 = 40.0;

    fn wav_bytes() -> &'static [u8] {
        let mut cur = std::io::Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec { channels: 1, sample_rate: FILE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
            for n in 0..LOOP_FRAMES {
                let s = (2.0 * PI * CYCLES * n as f64 / LOOP_FRAMES as f64).sin() * 0.5;
                w.write_sample((s * 32767.0).round() as i16).unwrap();
            }
            w.finalize().unwrap();
        }
        Box::leak(cur.into_inner().into_boxed_slice())
    }

    fn wait_for_fill(src: &ResampledSource<BufferedStream>, frames: u64) {
        let t0 = Instant::now();
        while src.fill_frames() < frames {
            assert!(t0.elapsed() < Duration::from_secs(10), "stream never filled (have {})", src.fill_frames());
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn seam_run(policy: StreamingPolicy) {
        let wave = WaveFileStream::from_bytes(wav_bytes()).unwrap();
        let stream = BufferedStream::new(Box::new(PolicyOverride::new(wave, policy)));
        let out_rate = 48_000.0;
        let mut src = ResampledSource::new(stream, out_rate).unwrap();
        let step = FILE_RATE as f64 / out_rate;
        let total_out = (3.6 * LOOP_FRAMES as f64 / step) as usize;
        let mut out = Vec::with_capacity(total_out);
        let mut buf = vec![0.0f32; 256];
        while out.len() < total_out {
            wait_for_fill(&src, (256.0 * step) as u64 + 128);
            let n = src.render(&mut [&mut buf[..]]);
            assert_eq!(n, 256, "underrun while looping at output frame {}", out.len());
            out.extend_from_slice(&buf);
        }
        // Compare with the ideal continuous tone (skip the start-up transient: the resampler
        // timeline starts with silence before frame 0).
        let mut max_err = 0.0f64;
        let mut max_seam_err = 0.0f64;
        for (k, &y) in out.iter().enumerate().skip(64) {
            let t = k as f64 * step; // input frames
            let ideal = (2.0 * PI * CYCLES * t / LOOP_FRAMES as f64).sin() * 0.5;
            let e = (y as f64 - ideal).abs();
            max_err = max_err.max(e);
            let into = t % LOOP_FRAMES as f64;
            if t > LOOP_FRAMES as f64 / 2.0 && (into < 40.0 || into > LOOP_FRAMES as f64 - 40.0) {
                max_seam_err = max_seam_err.max(e);
            }
        }
        eprintln!("{policy:?}: max error vs ideal tone {max_err:.2e}, around the seams {max_seam_err:.2e}");
        assert!(max_err < 1e-3, "max error {max_err}");
        assert!(max_seam_err < 1e-3, "seam error {max_seam_err}");
        assert_eq!(src.stats().underruns, 0);
        // A click would show as a large second difference; a 441 Hz tone at 48 kHz has a
        // second difference of at most 0.5 * (2 pi 441/48000)^2 ~ 1.7e-3.
        let max_d2 = out.windows(3).skip(64).map(|w| (w[2] - 2.0 * w[1] + w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_d2 < 2.5e-3, "second difference {max_d2}");
    }

    #[test]
    fn looping_common_stream_is_continuous_across_the_wrap() {
        seam_run(StreamingPolicy::Common);
    }

    #[test]
    fn looping_once_stream_is_continuous_across_the_wrap() {
        seam_run(StreamingPolicy::Once);
    }
}


#[test]
fn render_block_matches_render_and_never_allocates() {
    let mk = || {
        let mut s = ResampledSource::new(SimRing::new(44_100.0, 8192, 4096), 48_000.0).unwrap();
        s.source_mut().produce(0.0, 0.0);
        s
    };
    let (mut a, mut b) = (mk(), mk());
    let mut buf = vec![0.0f32; 200];
    assert_eq!(a.render(&mut [&mut buf[..]]), 200);
    let ((), n) = count_allocs(|| {
        assert_eq!(b.render_block(200), 200);
    });
    assert_eq!(n, 0);
    assert_eq!(b.block(0, 200), &buf[..]);
    assert!(b.block(5, 200).is_empty());
}
