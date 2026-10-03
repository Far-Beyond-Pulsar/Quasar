//! Atmospheric (air) absorption of pure tones, ISO 9613-1:1993.
//!
//! [`air_absorption_db_per_m`] returns the pure-tone attenuation coefficient
//! `alpha` in **dB per metre** (the standard tabulates dB/km; divide by 1000).
//! [`air_absorption_gain`] converts that to a linear *amplitude* gain over a
//! distance, `10^(-alpha d / 20)` (NOT `exp(-alpha d)`, which would treat the
//! dB value as nepers), per Quasar octave band centre.
//!
//! Formulation (T absolute temperature, `T0 = 293.15 K`, `T01 = 273.16 K`,
//! `pr = 101.325 kPa`, `pa` ambient pressure, `hr` relative humidity in %):
//!
//! ```text
//! psat/pr = 10^(-6.8346 (T01/T)^1.261 + 4.6151)
//! h       = hr (psat/pr) (pr/pa)                       molar water-vapour concentration, %
//! frO     = (pa/pr) (24 + 4.04e4 h (0.02 + h) / (0.391 + h))
//! frN     = (pa/pr) (T/T0)^-1/2 (9 + 280 h exp(-4.170 ((T/T0)^-1/3 - 1)))
//! alpha   = 8.686 f^2 { 1.84e-11 (pa/pr)^-1 (T/T0)^1/2
//!             + (T/T0)^-5/2 [ 0.01275 e^(-2239.1/T) / (frO + f^2/frO)
//!                           + 0.1068  e^(-3352.0/T) / (frN + f^2/frN) ] }
//! ```
//!
//! Inputs are clamped to the standard's range (T in [-20, 50] C, hr in
//! [0, 100] %, pa in [50, 200] kPa), so the result is always finite.

use crate::bands::{Band8, FREQ_BAND_CENTRES};

/// Reference ambient pressure (kPa).
pub const REFERENCE_PRESSURE_KPA: f32 = 101.325;

/// Pure-tone air absorption coefficient in dB/m at standard pressure
/// (101.325 kPa). See [`air_absorption_db_per_m_at`] for explicit pressure.
pub fn air_absorption_db_per_m(freq_hz: f32, temp_c: f32, humidity_percent: f32) -> f32 {
    air_absorption_db_per_m_at(freq_hz, temp_c, humidity_percent, REFERENCE_PRESSURE_KPA)
}

/// Pure-tone air absorption coefficient in dB/m (ISO 9613-1).
///
/// `pressure_kpa` is the ambient pressure `pa`. NaN inputs fall back to
/// 20 C / 50 % / 101.325 kPa.
pub fn air_absorption_db_per_m_at(
    freq_hz: f32,
    temp_c: f32,
    humidity_percent: f32,
    pressure_kpa: f32,
) -> f32 {
    let temp_c = if temp_c.is_nan() { 20.0 } else { temp_c.clamp(-20.0, 50.0) };
    let hr = if humidity_percent.is_nan() { 50.0 } else { humidity_percent.clamp(0.0, 100.0) };
    let pa_kpa = if pressure_kpa.is_nan() { REFERENCE_PRESSURE_KPA } else { pressure_kpa.clamp(50.0, 200.0) };
    let f = if freq_hz.is_finite() { freq_hz.max(0.0) as f64 } else { 0.0 };

    let t = temp_c as f64 + 273.15;
    let t0 = 293.15_f64;
    let t01 = 273.16_f64;
    let pa_pr = pa_kpa as f64 / REFERENCE_PRESSURE_KPA as f64;

    let c = -6.8346 * (t01 / t).powf(1.261) + 4.6151;
    let psat_pr = 10.0_f64.powf(c);
    let h = hr as f64 * psat_pr / pa_pr;

    let fr_o = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    let fr_n = pa_pr
        * (t / t0).powf(-0.5)
        * (9.0 + 280.0 * h * (-4.170 * ((t / t0).powf(-1.0 / 3.0) - 1.0)).exp());

    let f2 = f * f;
    let alpha = 8.686
        * f2
        * (1.84e-11 / pa_pr * (t / t0).sqrt()
            + (t / t0).powf(-2.5)
                * (0.01275 * (-2239.1 / t).exp() / (fr_o + f2 / fr_o)
                    + 0.1068 * (-3352.0 / t).exp() / (fr_n + f2 / fr_n)));

    if alpha.is_finite() { alpha.max(0.0) as f32 } else { 0.0 }
}

/// Linear amplitude gain per octave band after `distance_m` metres of air:
/// `10^(-alpha(f_band) * distance / 20)`, in `(0, 1]`.
pub fn air_absorption_gain(distance_m: f32, temp_c: f32, humidity_percent: f32) -> Band8 {
    let d = if distance_m.is_finite() { distance_m.max(0.0) } else { 0.0 };
    let mut v = [1.0_f32; 8];
    for (i, g) in v.iter_mut().enumerate() {
        let alpha = air_absorption_db_per_m(FREQ_BAND_CENTRES[i], temp_c, humidity_percent);
        *g = 10.0_f32.powf(-alpha * d / 20.0);
    }
    Band8::new(v)
}
