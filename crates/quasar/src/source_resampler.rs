//! Source-side adapter that turns a cumulative frame source (the [`BufferedStream`] ring, or any
//! [`FrameSource`]) into device-rate planar blocks through the polyphase resampler (#76), with an
//! optional ring-fill drift controller.
//!
//! # Data flow
//!
//! ```text
//! BufferedStream ring  --read_block-->  planar scratch  --PolyphaseResampler-->  device-rate planar
//!   (file rate, cumulative frames,                         (ratio trimmed by the        block handed to
//!    looping is sample-continuous)                          drift controller)            the engine
//! ```
//!
//! * The ring addresses frames in a **cumulative** domain: when a looping file wraps, frame
//!   `total_frames` is simply the first frame of the next pass. The adapter therefore sees one
//!   endless sample-continuous stream, and the resampler's history carries across the seam, so a
//!   loop wrap is exactly as smooth as the content itself (no click, no re-priming).
//! * [`ResampledSource::render`] is real-time safe: every buffer is allocated in
//!   [`ResampledSource::new`]; the call neither locks (blocking), allocates, nor panics. If the
//!   ring runs dry the remainder of the block is zero-filled and the underrun is counted.
//!
//! # Drift control
//!
//! A *file* source has no clock of its own: the I/O thread refills the ring as fast as the consumer
//! drains it, so the ring level carries no information about clock drift and the controller must
//! stay **off** (the default; this is what the example does). A *live* source (network stream,
//! another device) produces at its own clock, so its rate differs from the device clock by some
//! tens to thousands of ppm and the ring slowly fills or drains. [`DriftController`] (enabled with
//! [`ResampledSource::enable_drift_control`]) watches the fill level once per rendered block and
//! steers [`PolyphaseResampler::set_ratio_trim`]:
//!
//! ```text
//! e      = lowpass_1s(fill) - target          (frames; > 0: ring filling up)
//! trim   = e / (tau * rate_in) + I            (P term: remove the error over ~tau seconds)
//! I     += e / (tau * tau_i * rate_in) * dt   (I term: learns the steady drift; clamped)
//! ```
//!
//! With `tau = 6 s`, `tau_i = 24 s` the loop is `w = 0.083 rad/s`, `zeta = 1` (critically damped,
//! no oscillation). The trim is clamped to `+-max_trim` (default 5000 ppm), and the resampler itself
//! slews it at [`TRIM_SLEW_PER_SAMPLE`] per output sample (about 240 ppm per second at 48 kHz), so
//! the correction is a smooth, inaudible pitch ramp.
//!
//! [`BufferedStream`]: crate::streaming_source::BufferedStream
//! [`TRIM_SLEW_PER_SAMPLE`]: crate::resampler::TRIM_SLEW_PER_SAMPLE

use crate::resampler::PolyphaseResampler;

/// Most channels the adapter handles.
pub const MAX_CHANNELS: usize = 16;
/// Input frames fetched per resampler call.
const SCRATCH_FRAMES: usize = 2048;
/// Longest block [`ResampledSource::render_block`] can produce.
pub const MAX_BLOCK_FRAMES: usize = 1024;
/// Time constant of the fill-level smoothing in the drift controller.
const FILL_SMOOTH_SECS: f64 = 1.0;

/// A cumulative-domain frame source (see the module docs).
pub trait FrameSource {
    fn channels(&self) -> usize;
    /// Native sample rate of the source in Hz.
    fn sample_rate(&self) -> u32;
    /// Exclusive end of the readable range (cumulative frame index).
    fn written_frames(&self) -> u64;
    /// The cursor the writer uses to decide how far it may run ahead.
    fn read_cursor(&self) -> u64;
    /// Copy `dst.len()` frames of channel `ch`, starting at cumulative frame `start`.
    fn read_block(&self, start: u64, ch: usize, dst: &mut [f32]);
    /// Tell the writer that `frames` more frames (relative) have been consumed.
    fn advance_read(&mut self, frames: u64);
}

#[cfg(feature = "streaming")]
impl FrameSource for crate::streaming_source::BufferedStream {
    fn channels(&self) -> usize {
        crate::streaming_source::BufferedStream::channels(self)
    }
    fn sample_rate(&self) -> u32 {
        crate::streaming_source::BufferedStream::sample_rate(self)
    }
    fn written_frames(&self) -> u64 {
        crate::streaming_source::BufferedStream::written_frames(self)
    }
    fn read_cursor(&self) -> u64 {
        crate::streaming_source::BufferedStream::read_cursor(self)
    }
    fn read_block(&self, start: u64, ch: usize, dst: &mut [f32]) {
        self.sample_block(start, ch, dst)
    }
    fn advance_read(&mut self, frames: u64) {
        crate::streaming_source::BufferedStream::advance_read(self, frames)
    }
}

/// PI controller steering the resampler ratio from the ring fill level. See the module docs.
#[derive(Clone, Debug)]
pub struct DriftController {
    target: f64,
    tau: f64,
    tau_i: f64,
    max_trim: f64,
    integ: f64,
    /// One-pole smoothed fill (frames) and its state flag: the ring level is a sawtooth at the
    /// writer's chunk rate, which must not reach the trim.
    fill_lp: f64,
    primed: bool,
    trim: f64,
}

impl DriftController {
    /// Controller holding the fill at `target_fill_frames`, with the default time constants
    /// (`tau = 6 s`, `tau_i = 24 s`) and `+-5000 ppm` trim limit.
    pub fn new(target_fill_frames: f64) -> Self {
        Self { target: target_fill_frames.max(1.0), tau: 6.0, tau_i: 24.0, max_trim: 5000e-6, integ: 0.0, fill_lp: 0.0, primed: false, trim: 0.0 }
    }

    /// Override time constants (seconds) and the trim limit (fraction, e.g. `5000e-6`).
    pub fn with_tuning(mut self, tau: f64, tau_i: f64, max_trim: f64) -> Self {
        self.tau = tau.max(0.1);
        self.tau_i = tau_i.max(0.1);
        self.max_trim = max_trim.clamp(0.0, crate::resampler::MAX_RATIO_TRIM);
        self
    }

    /// One control step: `fill` is the current ring fill in frames, `dt` the time since the last
    /// step in seconds, `in_rate` the source rate in Hz. Returns the trim to apply.
    pub fn update(&mut self, fill: f64, dt: f64, in_rate: f64) -> f64 {
        if !self.primed {
            self.fill_lp = fill;
            self.primed = true;
        }
        self.fill_lp += (fill - self.fill_lp) * (1.0 - (-dt / FILL_SMOOTH_SECS).exp());
        let e = self.fill_lp - self.target;
        let rate = in_rate.max(1.0);
        self.integ += e / (self.tau * self.tau_i * rate) * dt;
        self.integ = self.integ.clamp(-self.max_trim, self.max_trim);
        self.trim = (e / (self.tau * rate) + self.integ).clamp(-self.max_trim, self.max_trim);
        self.trim
    }

    /// The last trim returned.
    pub fn trim(&self) -> f64 {
        self.trim
    }

    /// Drop the learned state.
    pub fn reset(&mut self) {
        self.integ = 0.0;
        self.primed = false;
        self.trim = 0.0;
    }
}

/// Counters kept by [`ResampledSource`] (plain integers; read them from the owning thread).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResampledSourceStats {
    /// `render` calls in which the ring ran dry (the block was zero-filled from that point).
    pub underruns: u64,
    /// Output frames written (excluding zero-filled underrun frames).
    pub frames_out: u64,
    /// Output frames zero-filled because of underruns.
    pub frames_dropped: u64,
}

/// Why a [`ResampledSource`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResampledSourceError {
    /// The source reports zero channels or more than [`MAX_CHANNELS`].
    UnsupportedChannelCount(usize),
}

/// Device-rate, resampled, planar view of a [`FrameSource`]. See the module docs.
pub struct ResampledSource<S: FrameSource> {
    src: S,
    rs: PolyphaseResampler,
    ch: usize,
    in_rate: f64,
    out_rate: f64,
    /// Next cumulative source frame to push into the resampler.
    in_pos: u64,
    scratch: Vec<Vec<f32>>,
    /// Internal output blocks for [`render_block`](Self::render_block).
    block_out: Vec<Vec<f32>>,
    ctl: Option<DriftController>,
    stats: ResampledSourceStats,
}

impl<S: FrameSource> ResampledSource<S> {
    /// Wrap `src`, resampling from its native rate to `out_rate` Hz. Allocates.
    pub fn new(src: S, out_rate: f64) -> Result<Self, ResampledSourceError> {
        let ch = src.channels();
        if ch == 0 || ch > MAX_CHANNELS {
            return Err(ResampledSourceError::UnsupportedChannelCount(ch));
        }
        let in_rate = src.sample_rate() as f64;
        let rs = PolyphaseResampler::new(ch, in_rate, out_rate);
        let in_pos = src.read_cursor();
        Ok(Self {
            src,
            rs,
            ch,
            in_rate,
            out_rate: out_rate.max(1.0),
            in_pos,
            scratch: (0..ch).map(|_| vec![0.0; SCRATCH_FRAMES]).collect(),
            block_out: (0..ch).map(|_| vec![0.0; MAX_BLOCK_FRAMES]).collect(),
            ctl: None,
            stats: ResampledSourceStats::default(),
        })
    }

    /// Turn on ring-fill drift control (for live sources, not files; see the module docs).
    pub fn enable_drift_control(&mut self, target_fill_frames: f64) {
        self.ctl = Some(DriftController::new(target_fill_frames));
    }

    /// Turn on drift control with an explicit controller.
    pub fn set_drift_controller(&mut self, ctl: Option<DriftController>) {
        if ctl.is_none() {
            self.rs.set_ratio_trim(0.0);
        }
        self.ctl = ctl;
    }

    pub fn source(&self) -> &S {
        &self.src
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.src
    }

    pub fn resampler(&self) -> &PolyphaseResampler {
        &self.rs
    }

    pub fn channels(&self) -> usize {
        self.ch
    }

    pub fn stats(&self) -> ResampledSourceStats {
        self.stats
    }

    /// Source frames currently buffered ahead of the resampler's input position.
    pub fn fill_frames(&self) -> u64 {
        self.src.written_frames().saturating_sub(self.in_pos)
    }

    /// The drift trim currently applied by the resampler (after slewing).
    pub fn applied_trim(&self) -> f64 {
        self.rs.ratio_trim()
    }

    /// Fill `out` (one slice per channel, equal lengths; extra channels beyond the source's are
    /// zeroed) with device-rate frames. Returns the number of frames produced; the remainder of the
    /// block is zero-filled on an underrun. Never allocates.
    pub fn render(&mut self, out: &mut [&mut [f32]]) -> usize {
        let n = out.iter().map(|s| s.len()).min().unwrap_or(0);
        if n == 0 {
            return 0;
        }
        if let Some(ctl) = self.ctl.as_mut() {
            let trim = ctl.update(self.src.written_frames().saturating_sub(self.in_pos) as f64, n as f64 / self.out_rate, self.in_rate);
            self.rs.set_ratio_trim(trim);
        }
        let ch = self.ch.min(out.len());
        let mut produced = 0usize;
        while produced < n {
            let need = self.rs.input_needed(n - produced);
            let avail = self.src.written_frames().saturating_sub(self.in_pos) as usize;
            let take = need.min(SCRATCH_FRAMES).min(avail);
            for k in 0..ch {
                self.src.read_block(self.in_pos, k, &mut self.scratch[k][..take]);
            }
            let input: [&[f32]; MAX_CHANNELS] =
                std::array::from_fn(|k| if k < ch { &self.scratch[k][..take] } else { &[] });
            let mut outs: [&mut [f32]; MAX_CHANNELS] = {
                let mut it = out.iter_mut();
                std::array::from_fn(|k| match it.next() {
                    Some(s) if k < ch => &mut s[produced..n],
                    _ => &mut [],
                })
            };
            let res = self.rs.process(&input[..ch], &mut outs[..ch]);
            self.in_pos += res.consumed as u64;
            if res.consumed > 0 {
                self.src.advance_read(res.consumed as u64);
            }
            produced += res.produced;
            if res.consumed == 0 && res.produced == 0 {
                break;
            }
        }
        self.stats.frames_out += produced as u64;
        if produced < n {
            self.stats.underruns += 1;
            self.stats.frames_dropped += (n - produced) as u64;
            for s in out.iter_mut() {
                let l = s.len().min(n);
                s[produced..l].fill(0.0);
            }
        }
        for s in out.iter_mut().skip(ch) {
            s.fill(0.0);
        }
        produced
    }
}

impl<S: FrameSource> ResampledSource<S> {
    /// Render `frames` (at most [`MAX_BLOCK_FRAMES`]) into the adapter's own planar buffers and
    /// return the frames produced; read them with [`block`](Self::block). Same real-time
    /// guarantees as [`render`](Self::render). Convenient when the caller cannot hold several
    /// `&mut` channel slices at once (e.g. an `AudioBuffer`).
    pub fn render_block(&mut self, frames: usize) -> usize {
        let n = frames.min(MAX_BLOCK_FRAMES);
        let mut bufs = std::mem::take(&mut self.block_out);
        let produced = {
            let ch = bufs.len().min(MAX_CHANNELS);
            let mut it = bufs.iter_mut();
            let mut outs: [&mut [f32]; MAX_CHANNELS] = std::array::from_fn(|k| match it.next() {
                Some(b) if k < ch => &mut b[..n],
                _ => &mut [],
            });
            self.render(&mut outs[..ch])
        };
        self.block_out = bufs;
        produced
    }

    /// Channel `ch` of the last [`render_block`](Self::render_block) (`frames` long).
    pub fn block(&self, ch: usize, frames: usize) -> &[f32] {
        self.block_out.get(ch).map_or(&[][..], |b| &b[..frames.min(MAX_BLOCK_FRAMES)])
    }
}
