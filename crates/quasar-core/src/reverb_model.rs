//! Statistical late-field model shared by the backends and the hybrid sampler.
//!
//! `late_loudness_db` of a [`crate::backend::LateReverbEstimate`] is the level of
//! the diffuse (reverberant) field relative to the DIRECT sound of the same emitter at
//! the 1 m reference distance (direct gain 1), in dB of amplitude (RMS). In a diffuse
//! field with room constant `R` the reverberant energy density relative to the direct
//! energy at 1 m is `16 pi / R` (`R = S a / (1 - a)`), independent of where emitter and
//! listener stand; the direct sound then falls as `1/r`, so the direct-to-reverberant
//! ratio at distance `r` is `R / (16 pi r^2)` and the critical distance is
//! `sqrt(R / (16 pi))`.

/// Speed-independent Sabine constant (s/m): `T60 = 0.161 V / A`.
pub const SABINE_K: f32 = 0.161;

/// Lowest / highest late level returned (dB re the direct sound at 1 m).
pub const LATE_DB_MIN: f32 = -60.0;
pub const LATE_DB_MAX: f32 = 6.0;

/// Late-field level (dB re direct at 1 m) from a reverberation time and a room volume,
/// using the Sabine relation `R ~ A = 0.161 V / T60` (valid for low absorption):
/// `10 log10(16 pi T60 / (0.161 V)) = 10 log10(312.2 T60 / V)`. This is the form of
/// Barron's `31200 T / V` (which is re the direct sound at 10 m, 20 dB lower).
/// Non-finite or non-positive inputs return [`LATE_DB_MIN`] (no usable room).
pub fn late_loudness_from_t60_volume(t60: f32, volume_m3: f32) -> f32 {
    if !(t60.is_finite() && volume_m3.is_finite() && t60 > 0.0 && volume_m3 > 0.0) {
        return LATE_DB_MIN;
    }
    let ratio = 16.0 * std::f32::consts::PI * t60 / (SABINE_K * volume_m3);
    (10.0 * ratio.log10()).clamp(LATE_DB_MIN, LATE_DB_MAX)
}
