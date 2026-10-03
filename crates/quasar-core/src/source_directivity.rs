//! Source directivity (#74): radiation patterns and the geometry they need.
//!
//! [`SceneOutputConfig::directivity`](crate::scene_output::SceneOutputConfig) `d` in `0..=1` and
//! [`orientation`](crate::scene_output::SceneOutputConfig) select a first-order ("cardioid family")
//! pattern. With `theta` the angle between the emitter's orientation and the direction from the
//! emitter to wherever the sound goes (the listener for the direct path, the first reflection
//! point for a reflection):
//!
//! ```text
//! p(theta)   = 1 - (d / 2) * (1 - cos theta)          0 <= p <= 1,  p(0) = 1 (on axis)
//! g_b(theta) = max(p, MIN_DIRECTIVITY_GAIN) ^ n_b     per octave band b
//! ```
//!
//! * `d = 0`: `p = 1` in every direction: omnidirectional, every band exactly 1.0.
//! * `d = 1`: at 1 kHz (`n = 1`) an exact cardioid: `p = (1 + cos theta) / 2`, side (90 deg) -6 dB,
//!   rear a null (floored at [`MIN_DIRECTIVITY_GAIN`] = -60 dB).
//! * In between the rear gain at 1 kHz is `1 - d` (e.g. `d = 0.5`: sub-cardioid, rear -6 dB).
//! * **Per-band narrowing**: the exponent `n_b` ([`BAND_EXPONENTS`]) is below 1 at low bands
//!   (wider, tending to omni, as real sources are at low frequencies) and above 1 at high bands
//!   (narrower), so the cone is tighter at 8 kHz than at 125 Hz while the rear is never LOUDER at
//!   higher bands.
//!
//! The pattern is applied on the compute side per (listener, emitter) pair: to the direct path
//! (`SpatialCoefficients::directivity_gain`), to every early reflection tap (each one leaves the
//! emitter in its own direction, see [`first_order_bounce_point`]) and, as a diffuse-field average
//! ([`diffuse_send_gain`]), to the reverb send.

use crate::bands::Band8;

/// Pattern exponent per octave band (62.5 Hz .. 8 kHz). 1.0 at 1 kHz.
pub const BAND_EXPONENTS: [f32; 8] = [0.5, 0.65, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8];

/// Floor of the pattern gain (-60 dB): a perfect null would zero a band for the band-EQ designer.
pub const MIN_DIRECTIVITY_GAIN: f32 = 1e-3;

/// Per-band linear amplitude gain of the pattern with parameter `directivity` (clamped to
/// `0..=1`) at emission angle `theta` with `cos_theta = cos(theta)` (clamped to `-1..=1`;
/// non-finite reads as on-axis). Exactly [`Band8::splat`]`(1.0)` for `directivity <= 0`.
pub fn pattern_band_gains(directivity: f32, cos_theta: f32) -> Band8 {
    let d = if directivity.is_finite() { directivity.clamp(0.0, 1.0) } else { 0.0 };
    if d <= 0.0 {
        return Band8::splat(1.0);
    }
    let c = if cos_theta.is_finite() { cos_theta.clamp(-1.0, 1.0) } else { 1.0 };
    let p = (1.0 - 0.5 * d * (1.0 - c)).max(MIN_DIRECTIVITY_GAIN);
    let mut g = [0.0_f32; 8];
    for (b, out) in g.iter_mut().enumerate() {
        *out = p.powf(BAND_EXPONENTS[b]);
    }
    Band8::new(g)
}

/// Per-band mean-square of the pattern over the whole sphere ("diffuse-field power", relative to
/// an omnidirectional emitter of the same on-axis level). Uses the exact `d_omega = 2 pi d(cos)`
/// measure with a 256-point midpoint rule over `cos theta`.
pub fn diffuse_field_power(directivity: f32) -> Band8 {
    let d = if directivity.is_finite() { directivity.clamp(0.0, 1.0) } else { 0.0 };
    if d <= 0.0 {
        return Band8::splat(1.0);
    }
    const N: usize = 256;
    let mut acc = [0.0_f64; 8];
    for i in 0..N {
        let c = -1.0 + 2.0 * (i as f32 + 0.5) / N as f32;
        let g = pattern_band_gains(d, c);
        for b in 0..8 {
            acc[b] += (g.0[b] as f64) * (g.0[b] as f64);
        }
    }
    let mut out = [0.0_f32; 8];
    for b in 0..8 {
        out[b] = (acc[b] / N as f64) as f32;
    }
    Band8::new(out)
}

/// Scalar gain for the reverb send: the late field is excited by the emitter's TOTAL radiated
/// power, so the send follows the diffuse-field average of the pattern, `sqrt` of the mean over
/// bands of [`diffuse_field_power`]. Exactly 1.0 for an omnidirectional emitter, independent of
/// the listener direction.
pub fn diffuse_send_gain(directivity: f32) -> f32 {
    let p = diffuse_field_power(directivity);
    (p.0.iter().sum::<f32>() / 8.0).sqrt()
}

/// `cos` of the angle between `orientation` and the direction `from -> to` (emitter to target).
/// `None` if the orientation is missing / degenerate or the points coincide (treated as
/// omnidirectional by the callers).
pub fn emission_cos(orientation: Option<[f32; 3]>, from: [f32; 3], to: [f32; 3]) -> Option<f32> {
    let o = normalize(orientation?)?;
    let u = normalize([to[0] - from[0], to[1] - from[1], to[2] - from[2]])?;
    Some((o[0] * u[0] + o[1] * u[1] + o[2] * u[2]).clamp(-1.0, 1.0))
}

/// Where a first-order specular reflection touched the wall, from what a backend reports.
///
/// `emitter` / `listener`: world positions. `arrival_dir`: unit vector from the listener toward
/// the reflection point (`EarlyReflection::direction`). `total_path`: the emitter -> point ->
/// listener path length in metres. With `P = L + r * dir`, `|E - P| + r = total_path` gives
/// `r = (T^2 - |v|^2) / (2 (T - v . dir))` for `v = E - L`: exact for one bounce. For higher
/// orders (`direction` points at the LAST bounce) the result is the last bounce point, which only
/// approximates the departure direction from the emitter. `None` for degenerate input.
pub fn first_order_bounce_point(
    emitter: [f32; 3],
    listener: [f32; 3],
    arrival_dir: [f32; 3],
    total_path: f32,
) -> Option<[f32; 3]> {
    let d = normalize(arrival_dir)?;
    let v = [emitter[0] - listener[0], emitter[1] - listener[1], emitter[2] - listener[2]];
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let vd = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
    let t = total_path;
    let denom = 2.0 * (t - vd);
    if !(t.is_finite() && denom > 1e-6) {
        return None;
    }
    let r = (t * t - v2) / denom;
    if !(r.is_finite() && r > 0.0) {
        return None;
    }
    Some([listener[0] + r * d[0], listener[1] + r * d[1], listener[2] + r * d[2]])
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 && l.is_finite() {
        Some([v[0] / l, v[1] / l, v[2] / l])
    } else {
        None
    }
}
