//! Core scene-output content model.
//!
//! Three strictly separated layers:
//! - [`Source`]: raw multi-channel audio content (a file/stream with N channels).
//! - [`SceneOutputConfig`]: a positioned world emitter whose audible content is
//!   the sum of explicit [`ChannelPull`] taps onto loaded sources.
//! - [`ListenerConfig`]: a world position + heading + physical device layout.
//!
//! Channel pulling is the ONLY way audio enters the mix. There is no
//! "play a whole file on an emitter" convenience path.

use crate::scene::Movability;

/// Identifier for a loaded audio source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(pub u32);

/// Identifier for a positioned scene output (world-space emitter/speaker).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneOutputId(pub u32);

/// Identifier for a listener (world position + physical device layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ListenerId(pub u32);

/// ONE tap in the patch bay. Content for a scene output = Σ over its pulls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelPull {
    /// Which loaded source this tap reads from.
    pub source_id: SourceId,
    /// Which channel of that source (`0 .. source.channels - 1`).
    pub channel: u32,
    /// Gain in dB (API surface is dB; converted to linear in DSP).
    pub gain_db: f32,
}

/// Description of a loaded multi-channel audio source.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceConfig {
    /// Path to the source file / stream handle.
    pub path: String,
    /// Number of channels the source carries.
    pub channels: usize,
}

/// Configuration for a positioned scene output (world-space emitter/speaker).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneOutputConfig {
    /// World-space position.
    pub position: [f32; 3],
    /// Heading for directivity (reserved for future phases).
    pub orientation: Option<[f32; 3]>,
    /// Directivity: `0.0` = omnidirectional, `1.0` = max cone.
    pub directivity: f32,
    /// Patch-bay taps that define this output's audible content.
    pub pulls: Vec<ChannelPull>,
    /// How the output's position may change over time.
    pub movability: Movability,
}

/// Physical speaker layout of a listener's output device.
#[derive(Clone, Debug, PartialEq)]
pub enum PhysicalOutputLayout {
    /// Two-channel stereo (L/R).
    Stereo,
    /// 5.1 surround (FL/FR/C/LFE/BL/BR).
    Surround51,
    /// 7.1.4 surround (base 7.1 + 4 height channels).
    Surround714,
    /// Quad (FL/FR/BL/BR).
    Quad,
    /// Arbitrary real-world speaker directions.
    Custom {
        /// Real-world speaker directions (unit vectors).
        positions: Vec<[f32; 3]>,
    },
    /// Headphone rendering via an HRTF.
    Hrtf,
}

/// Configuration for a listener (world position + heading + device layout).
#[derive(Clone, Debug, PartialEq)]
pub struct ListenerConfig {
    /// World-space position.
    pub position: [f32; 3],
    /// Normalized forward vector.
    pub heading: [f32; 3],
    /// Physical device layout the engine renders onto.
    pub physical_layout: PhysicalOutputLayout,
}

impl Default for SceneOutputConfig {
    /// Omnidirectional, static output at the origin with no pulls.
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation: None,
            directivity: 0.0,
            pulls: Vec::new(),
            movability: Movability::Static,
        }
    }
}

impl Default for ListenerConfig {
    /// Stereo listener at the origin facing -Z.
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            heading: [0.0, 0.0, -1.0],
            physical_layout: PhysicalOutputLayout::Stereo,
        }
    }
}

impl ChannelPull {
    /// Create a new patch-bay tap reading `channel` of `source_id` at `gain_db`.
    pub fn new(source_id: SourceId, channel: u32, gain_db: f32) -> Self {
        Self {
            source_id,
            channel,
            gain_db,
        }
    }
}

impl SceneOutputConfig {
    /// Create a scene output at `position` with the given `movability`,
    /// omnidirectional directivity and no pulls.
    pub fn new(position: [f32; 3], movability: Movability) -> Self {
        Self {
            position,
            orientation: None,
            directivity: 0.0,
            pulls: Vec::new(),
            movability,
        }
    }
}
