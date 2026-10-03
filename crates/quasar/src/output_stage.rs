//! Optional output format conversion of a listener's render (#83, #149).
//!
//! The listener is rendered in its PHYSICAL layout (say 7.1) into an internal scratch buffer; a
//! [`ChannelMatrix`] (BS.775-3 coefficients) converts that to the DEVICE layout (stereo, 5.1,
//! ...) into the buffer the caller supplied; then the output safety stage (limiter, scrub,
//! meters) runs on the DEVICE channels. The limiter is deliberately LAST: a downmix sums
//! channels (a front-heavy 7.1 mix can add up to +3 dB or more in the two remaining channels), so
//! the ceiling only holds if the limiter sees the channels that are actually played.
//!
//! * Not configured (the default): [`OutputConv::begin`] returns `None`, the renderer writes
//!   straight into the caller's buffer exactly as before, and the result is bit-identical.
//! * Reconfiguration is click-free. A new stage ramps up from silence while the previous one
//!   (or, when there was none, an identity pass-through of the physical layout) ramps down over
//!   the matrix ramp time (10 ms), and the two are summed. Both stages see the same physical
//!   render, so a switch is a linear cross-fade of two correlated mixes (no level dip).
//!   Removing the conversion ramps an identity stage up; once settled it is retired and the
//!   direct path (bit-identical) resumes.
//! * Everything that allocates (matrices, scratch, the swap shell) is built on the API thread and
//!   travels through the command queue; replaced pieces go back to the compute side as garbage.
//!   [`OutputConv::finish`] never allocates, locks or panics.
//! * If the caller's buffer has fewer channels than the device layout the extra matrix rows are
//!   ignored (never wrapped around).

use quasar_dsp::{AudioBuffer, ChannelMatrix, MatrixError};

/// One conversion matrix with the gains it ramps to when it is activated.
pub(crate) struct ConvStage {
    matrix: ChannelMatrix,
    target: Box<[f32]>,
    identity: bool,
}

impl ConvStage {
    /// Stage for `gains` (row-major `out x in`), starting at SILENCE. API thread (allocates).
    pub(crate) fn new(
        gains: Vec<f32>,
        in_ch: usize,
        out_ch: usize,
        sample_rate: f32,
        ramp_ms: f32,
        identity: bool,
    ) -> Result<Box<Self>, MatrixError> {
        let matrix = ChannelMatrix::new(in_ch, out_ch, sample_rate, ramp_ms)?;
        if gains.len() != in_ch * out_ch {
            return Err(MatrixError::BadShape { given: gains.len(), expected: in_ch * out_ch });
        }
        Ok(Box::new(Self { matrix, target: gains.into_boxed_slice(), identity }))
    }
}

/// Command payload: the replacement stage plus the spare pieces the audio side may need. The
/// shell travels back as garbage with whatever was not used.
pub(crate) struct ConvSwap {
    /// The new conversion, `None` = remove the conversion.
    pub(crate) new: Option<Box<ConvStage>>,
    /// Identity stage (physical -> physical) used as the pass-through that is faded out when
    /// there was no conversion before, or faded in when the conversion is removed.
    pub(crate) spare: Option<Box<ConvStage>>,
    /// Physical-layout scratch buffer, installed when the listener has none yet.
    pub(crate) scratch: Option<Box<AudioBuffer>>,
    /// Receives a still-fading older stage that has to be dropped (rapid double switch).
    pub(crate) extra: Option<Box<ConvStage>>,
}

impl ConvSwap {
    pub(crate) fn new(new: Option<Box<ConvStage>>, spare: Box<ConvStage>, scratch: Box<AudioBuffer>) -> Box<Self> {
        Box::new(Self { new, spare: Some(spare), scratch: Some(scratch), extra: None })
    }
}

/// Pieces retired by the audio side, dropped by the compute side.
#[allow(dead_code)]
pub(crate) enum ConvGarbage {
    Stage(Box<ConvStage>),
    Scratch(Box<AudioBuffer>),
    Swap(Box<ConvSwap>),
}

/// Per-listener conversion state (audio thread).
pub(crate) struct OutputConv {
    scratch: Option<Box<AudioBuffer>>,
    cur: Option<Box<ConvStage>>,
    old: Option<Box<ConvStage>>,
    retired: [Option<Box<ConvStage>>; 2],
    retired_scratch: Option<Box<AudioBuffer>>,
}

impl OutputConv {
    pub(crate) fn empty() -> Self {
        Self { scratch: None, cur: None, old: None, retired: [None, None], retired_scratch: None }
    }

    /// True while the listener renders through the conversion (a stage is installed or fading).
    pub(crate) fn active(&self) -> bool {
        self.cur.is_some() || self.old.is_some()
    }

    /// Apply a [`ConvSwap`]. Allocation-free; unused / replaced pieces stay in the shell, which is
    /// handed to `sink`.
    pub(crate) fn swap(&mut self, mut s: Box<ConvSwap>, sink: &mut impl FnMut(ConvGarbage)) {
        if self.scratch.is_none() {
            self.scratch = s.scratch.take();
        }
        let removing = s.new.is_none();
        if removing && self.cur.is_none() {
            sink(ConvGarbage::Swap(s)); // nothing to remove
            return;
        }
        // A still-fading older stage cannot be kept: drop it (rare rapid double switch).
        if let Some(prev_old) = self.old.take() {
            s.extra = Some(prev_old);
        }
        match self.cur.take() {
            Some(mut c) => {
                c.matrix.fade_to_zero();
                self.old = Some(c);
            }
            None => {
                // The direct path was active: fade it out as an identity stage.
                if let Some(mut id) = s.spare.take() {
                    id.matrix.set_matrix_immediate(&id.target).ok();
                    id.matrix.fade_to_zero();
                    self.old = Some(id);
                }
            }
        }
        let mut new = if removing { s.spare.take() } else { s.new.take() };
        if let Some(n) = new.as_mut() {
            n.matrix.set_matrix(&n.target).ok();
        }
        self.cur = new;
        sink(ConvGarbage::Swap(s));
    }

    /// Hand retired pieces (settled identity stages, the scratch buffer) to `sink`.
    pub(crate) fn flush_retired(&mut self, sink: &mut impl FnMut(ConvGarbage)) {
        for slot in self.retired.iter_mut() {
            if let Some(st) = slot.take() {
                sink(ConvGarbage::Stage(st));
            }
        }
        if let Some(sc) = self.retired_scratch.take() {
            sink(ConvGarbage::Scratch(sc));
        }
    }

    /// Before the render: if the conversion is active, take the physical-layout scratch buffer
    /// (sized to `block`) that the render must write into. `None` = render into the caller's
    /// buffer directly (bit-identical default path).
    pub(crate) fn begin(&mut self, block: usize) -> Option<Box<AudioBuffer>> {
        if !self.active() {
            return None;
        }
        let mut sb = self.scratch.take()?;
        sb.set_samples(block as u16);
        sb.clear();
        Some(sb)
    }

    /// After the render: convert `scratch` into `dest` (cleared first), advance the ramps and
    /// retire settled stages.
    pub(crate) fn finish(&mut self, scratch: Box<AudioBuffer>, dest: &mut AudioBuffer) {
        dest.clear();
        if let Some(c) = self.cur.as_mut() {
            c.matrix.process_add(&scratch, dest);
        }
        if let Some(o) = self.old.as_mut() {
            o.matrix.process_add(&scratch, dest);
        }
        self.scratch = Some(scratch);

        // The faded-out stage is done once it is silent.
        if self.old.as_ref().map_or(false, |o| o.matrix.is_silent()) {
            if let Some(slot) = self.retired.iter_mut().find(|s| s.is_none()) {
                *slot = self.old.take();
            }
        }
        // A settled identity stage IS the direct path: leave the conversion (bit-identical).
        if self.old.is_none() && self.cur.as_ref().map_or(false, |c| c.identity && c.matrix.is_settled()) {
            if let Some(slot) = self.retired.iter_mut().find(|s| s.is_none()) {
                *slot = self.cur.take();
                if self.retired_scratch.is_none() {
                    self.retired_scratch = self.scratch.take();
                }
            }
        }
    }
}
