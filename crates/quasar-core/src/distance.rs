//! Shared distance-attenuation model.
//!
//! One definition of "how loud is a source at distance `d`" for every backend
//! and sampling strategy (CPU ray tracer, baked probes, hybrid, GPU fallback),
//! so switching strategy never changes the direct-path loudness.
//!
//! The curves follow the OpenAL distance models:
//!
//! * [`DistanceCurve::Inverse`]:     `ref / (ref + rolloff * (d - ref))`
//! * [`DistanceCurve::Linear`]:      `1 - rolloff * (d - ref) / (max - ref)`
//! * [`DistanceCurve::Exponential`]: `(d / ref) ^ -rolloff`
//!
//! with `d` first clamped to `[min_distance, max_distance]`. The default is
//! the physical free-field law: inverse, 1 m reference, rolloff 1 (gain `1/d`,
//! i.e. 6.02 dB per doubling of distance) and distance clamped at 1 m, so the
//! gain never exceeds 1 (0 dB) however close the source gets.

/// Shape of the distance rolloff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceCurve {
    /// `ref / (ref + rolloff * (d - ref))`; the free-field `1/d` law for rolloff 1.
    Inverse,
    /// Straight line from 1 at the reference distance to `1 - rolloff` at the maximum.
    Linear,
    /// `(d / ref) ^ -rolloff`.
    Exponential,
}

/// Distance attenuation parameters (linear amplitude gain as a function of distance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceModel {
    /// Rolloff curve.
    pub curve: DistanceCurve,
    /// Distance (metres) at which the gain is 1.0 (default 1.0).
    pub reference_distance: f32,
    /// Rolloff steepness (default 1.0 = physical for `Inverse`).
    pub rolloff_factor: f32,
    /// Distances below this are treated as this distance (near-field clamp, default 1.0).
    pub min_distance: f32,
    /// Distances beyond this are treated as this distance (default 10 000 m).
    pub max_distance: f32,
}

impl Default for DistanceModel {
    /// Inverse distance, 1 m reference, rolloff 1 (6.02 dB per doubling),
    /// clamped at a 1 m minimum distance.
    fn default() -> Self {
        Self {
            curve: DistanceCurve::Inverse,
            reference_distance: 1.0,
            rolloff_factor: 1.0,
            min_distance: 1.0,
            max_distance: 10_000.0,
        }
    }
}

impl DistanceModel {
    /// Default model (see [`Default`]) with a different reference / minimum distance.
    pub fn with_reference(reference_distance: f32) -> Self {
        Self {
            reference_distance,
            min_distance: reference_distance,
            ..Self::default()
        }
    }

    /// Linear amplitude gain at `distance` metres. Always finite and `>= 0`;
    /// NaN distances read as the minimum distance. No allocation, no panics.
    pub fn gain(&self, distance: f32) -> f32 {
        let r = if self.reference_distance.is_finite() { self.reference_distance.max(1e-6) } else { 1.0 };
        let rolloff = if self.rolloff_factor.is_finite() { self.rolloff_factor.max(0.0) } else { 1.0 };
        let min_d = if self.min_distance.is_finite() { self.min_distance.max(0.0) } else { 0.0 };
        let max_d = if self.max_distance.is_nan() { f32::MAX } else { self.max_distance.max(min_d) };

        let d = if distance.is_nan() { min_d } else { distance.max(min_d).min(max_d) };

        let g = match self.curve {
            DistanceCurve::Inverse => r / (r + rolloff * (d - r)).max(1e-6 * r),
            DistanceCurve::Linear => {
                let span = max_d - r;
                if span.is_finite() && span > 0.0 {
                    1.0 - (rolloff * (d - r) / span).min(1.0)
                } else {
                    1.0
                }
            }
            DistanceCurve::Exponential => (d / r).max(1e-6).powf(-rolloff),
        };
        if g.is_finite() { g.max(0.0) } else { 0.0 }
    }
}
