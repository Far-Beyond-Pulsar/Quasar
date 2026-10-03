//! Device channel-order remap tables (#149).
//!
//! The engine renders every layout in ONE standard order (the WASAPI / SMPTE order):
//!
//! | channels | engine order |
//! |---|---|
//! | 2 | `L R` |
//! | 4 | `FL FR BL BR` |
//! | 6 | `FL FR C LFE BL BR` |
//! | 8 | `FL FR C LFE BL BR SL SR` |
//!
//! Audio APIs do not all use that order for the same channel count. [`DeviceChannelOrder`] names
//! the order of an API and [`ChannelRemap`] turns an engine-order frame into a device-order frame
//! (a pure permutation, allocation-free, bit exact). Channel counts for which no table exists
//! return an error instead of silently passing the channels through in a guessed order.
//!
//! # Sources, and what is an assumption
//!
//! * **WASAPI** ([`DeviceChannelOrder::Wasapi`]): channels follow the `WAVEFORMATEXTENSIBLE`
//!   channel-mask bit order (FL, FR, FC, LFE, BL, BR, ..., SL, SR). For 4, 6 and 8 channels with
//!   the usual masks (`SPEAKER_QUAD`, `SPEAKER_5POINT1`, `SPEAKER_7POINT1_SURROUND`) this IS the
//!   engine order, so the table is the identity. A device exposing a different mask keeps the
//!   same slot order.
//! * **ALSA** ([`DeviceChannelOrder::Alsa`]): the classic ALSA / OSS multichannel order
//!   `FL FR RL RR C LFE (SL SR)` used by the `surround40` / `surround51` / `surround71` plug
//!   devices. Stereo is `L R`. ASSUMPTION: the actual device may report another `snd_pcm_chmap`;
//!   cpal does not expose it, so the table is the default one.
//! * **CoreAudio** ([`DeviceChannelOrder::CoreAudio`]): stereo `L R`; quad `L R Ls Rs`; 5.1
//!   `L R C LFE Ls Rs` (`MPEG_5_1_A`, the same as the engine); 7.1 `L R C LFE Ls Rs Rls Rrs`
//!   (`MPEG_7_1_C`: the SIDE pair comes before the REAR pair). ASSUMPTION: macOS devices can
//!   publish any layout tag; this is the common one and is not queried from the device.
//!
//! Nothing here was verified against hardware.

use crate::audio_buffer::MAX_AUDIO_CHANNELS;

/// Channel order used by an audio API for multichannel output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChannelOrder {
    /// The engine's own order (no remap).
    Engine,
    Wasapi,
    Alsa,
    CoreAudio,
}

impl DeviceChannelOrder {
    /// The order of the platform's default audio API this crate is compiled for (Windows:
    /// WASAPI, macOS / iOS: CoreAudio, other: ALSA).
    pub fn for_current_platform() -> Self {
        if cfg!(target_os = "windows") {
            DeviceChannelOrder::Wasapi
        } else if cfg!(any(target_os = "macos", target_os = "ios")) {
            DeviceChannelOrder::CoreAudio
        } else {
            DeviceChannelOrder::Alsa
        }
    }

    /// `table[device_slot] = engine_channel` for `channels` channels, or `None` when no table is
    /// defined for that count.
    pub fn table(&self, channels: usize) -> Option<&'static [usize]> {
        use DeviceChannelOrder::*;
        const ID2: [usize; 2] = [0, 1];
        const ID4: [usize; 4] = [0, 1, 2, 3];
        const ID6: [usize; 6] = [0, 1, 2, 3, 4, 5];
        const ID8: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        // ALSA: FL FR RL RR C LFE (SL SR).
        const ALSA6: [usize; 6] = [0, 1, 4, 5, 2, 3];
        const ALSA8: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        // CoreAudio 7.1 (MPEG_7_1_C): L R C LFE Ls Rs Rls Rrs = FL FR C LFE SL SR BL BR.
        const CA8: [usize; 8] = [0, 1, 2, 3, 6, 7, 4, 5];
        match (self, channels) {
            (_, 2) => Some(&ID2),
            (_, 4) => Some(&ID4),
            (Engine | Wasapi | CoreAudio, 6) => Some(&ID6),
            (Engine | Wasapi, 8) => Some(&ID8),
            (Alsa, 6) => Some(&ALSA6),
            (Alsa, 8) => Some(&ALSA8),
            (CoreAudio, 8) => Some(&CA8),
            _ => None,
        }
    }
}

/// Why a remap could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelOrderError {
    /// No table exists for this order / channel count (e.g. a custom speaker array).
    UnsupportedChannelCount { order: DeviceChannelOrder, channels: usize },
}

impl std::fmt::Display for ChannelOrderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelOrderError::UnsupportedChannelCount { order, channels } => {
                write!(f, "no {order:?} channel-order table for {channels} channels")
            }
        }
    }
}

impl std::error::Error for ChannelOrderError {}

/// Engine order -> device order permutation of one frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelRemap {
    n: usize,
    /// `map[device_slot] = engine_channel`.
    map: [u8; MAX_AUDIO_CHANNELS],
}

impl ChannelRemap {
    /// Remap for `channels` channels into the `order` of the device.
    pub fn new(order: DeviceChannelOrder, channels: usize) -> Result<Self, ChannelOrderError> {
        let table = order.table(channels).ok_or(ChannelOrderError::UnsupportedChannelCount { order, channels })?;
        let mut map = [0u8; MAX_AUDIO_CHANNELS];
        for (m, &e) in map.iter_mut().zip(table) {
            *m = e as u8;
        }
        Ok(Self { n: channels, map })
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.n
    }

    /// True when the device order equals the engine order (the caller can skip the remap).
    pub fn is_identity(&self) -> bool {
        (0..self.n).all(|d| self.map[d] as usize == d)
    }

    /// Engine channel that feeds device slot `device_slot`.
    pub fn engine_channel(&self, device_slot: usize) -> usize {
        if device_slot < self.n {
            self.map[device_slot] as usize
        } else {
            device_slot
        }
    }

    /// Device slot that carries engine channel `engine_channel`.
    pub fn device_slot(&self, engine_channel: usize) -> usize {
        (0..self.n).find(|&d| self.map[d] as usize == engine_channel).unwrap_or(engine_channel)
    }

    /// Write the device-order frame `dst` from the engine-order frame `src` (`n` samples each).
    /// Never allocates.
    #[inline]
    pub fn apply_frame(&self, src: &[f32], dst: &mut [f32]) {
        for d in 0..self.n.min(dst.len()) {
            dst[d] = src.get(self.map[d] as usize).copied().unwrap_or(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_is_a_permutation() {
        for order in [DeviceChannelOrder::Engine, DeviceChannelOrder::Wasapi, DeviceChannelOrder::Alsa, DeviceChannelOrder::CoreAudio] {
            for n in [2usize, 4, 6, 8] {
                let t = order.table(n).unwrap();
                let mut s = t.to_vec();
                s.sort_unstable();
                assert_eq!(s, (0..n).collect::<Vec<_>>(), "{order:?} {n}");
            }
        }
    }

    #[test]
    fn alsa_51_puts_rear_before_centre() {
        let r = ChannelRemap::new(DeviceChannelOrder::Alsa, 6).unwrap();
        let eng = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]; // FL FR C LFE BL BR
        let mut dev = [0.0; 6];
        r.apply_frame(&eng, &mut dev);
        assert_eq!(dev, [10.0, 11.0, 14.0, 15.0, 12.0, 13.0]); // FL FR RL RR C LFE
        assert!(!r.is_identity());
        for d in 0..6 {
            assert_eq!(r.device_slot(r.engine_channel(d)), d);
        }
    }

    #[test]
    fn wasapi_is_identity_and_unsupported_counts_error() {
        for n in [2usize, 4, 6, 8] {
            assert!(ChannelRemap::new(DeviceChannelOrder::Wasapi, n).unwrap().is_identity());
        }
        assert!(ChannelRemap::new(DeviceChannelOrder::Alsa, 5).is_err());
        assert!(ChannelRemap::new(DeviceChannelOrder::CoreAudio, 12).is_err());
    }

    #[test]
    fn coreaudio_71_swaps_side_and_rear() {
        let r = ChannelRemap::new(DeviceChannelOrder::CoreAudio, 8).unwrap();
        let eng: Vec<f32> = (0..8).map(|i| i as f32).collect(); // FL FR C LFE BL BR SL SR
        let mut dev = [0.0; 8];
        r.apply_frame(&eng, &mut dev);
        assert_eq!(dev, [0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 4.0, 5.0]);
    }
}
