//! Emitter model (#156): spherical and directional speakers, and the physical size of a source.
//!
//! Three things describe how an emitter radiates:
//!
//! * [`EmitterPattern`]: the radiation pattern, as a per-band amplitude gain versus the angle
//!   between the emitter's forward axis and the direction a sound leaves in.
//!   - `Omni` is the spherical speaker: gain 1 in every direction.
//!   - `CardioidFamily` is the original one-parameter family of
//!     [`crate::source_directivity`] (kept bit-identical).
//!   - `Supercardioid` / `Hypercardioid` are first-order patterns with a rear lobe.
//!   - `SoundCone` is the game-audio cone: full level inside the inner angle, `outer_gain_db`
//!     outside the outer angle, linear in dB between (frequency independent).
//!   - `Horn` is a constant-directivity horn with its own horizontal and vertical coverage
//!     (the -6 dB full angles) and a rear floor. Coverage widens below ~1 kHz (a real horn
//!     loses pattern control below its cut-off) and the rear floor is shallower there.
//! * [`EmitterShape`]: the physical aperture, used to place the soft-occlusion probe points
//!   (a point, a sphere, a disc or a line array instead of one fixed 0.35 m disc).
//! * [`EmitterModel`]: pattern + shape (+ the cached diffuse-field gain) as stored on a
//!   [`crate::scene_output::SceneOutputConfig`].
//!
//! The engine evaluates the pattern per (listener, emitter) pair: on the direct path, on every
//! early-reflection tap (each leaves the emitter in its own direction) and, as the diffuse-field
//! average, on the reverb send. The backends receive an [`EmitterTrace`] so they can rank and
//! prune image-source paths by the pattern and sample occlusion over the real aperture; they
//! never apply the pattern to the returned gains (the engine does, once).
//!
//! The horn, cone and first-order patterns are engineering approximations (smooth, monotone,
//! documented numbers), not measurements; measured balloons are a separate feature (#148).

use crate::bands::Band8;
use crate::source_directivity::{
    diffuse_field_power as cardioid_diffuse_power, pattern_band_gains, BAND_EXPONENTS,
    MIN_DIRECTIVITY_GAIN,
};

/// Radiation pattern of an emitter. See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub enum EmitterPattern {
    /// Spherical speaker: gain 1 in every direction (also what an emitter without an
    /// orientation is).
    Omni,
    /// `directivity` in `0..=1`: omni .. cardioid (see [`crate::source_directivity`]).
    CardioidFamily { directivity: f32 },
    /// `|0.37 + 0.63 cos(theta)|`: narrower than a cardioid, rear lobe -11.7 dB, nulls at 125 deg.
    Supercardioid,
    /// `|0.25 + 0.75 cos(theta)|`: narrowest first-order pattern, rear lobe -6 dB, nulls at 109 deg.
    Hypercardioid,
    /// Game-style sound cone. Full angles in degrees (`inner <= outer`, clamped to `0..=360`);
    /// gain 0 dB inside `inner/2`, `outer_gain_db` (`<= 0`) outside `outer/2`, linear in dB
    /// (and in angle) between. Frequency independent.
    SoundCone { inner_deg: f32, outer_deg: f32, outer_gain_db: f32 },
    /// Constant-directivity horn. `h_deg` / `v_deg` are the FULL horizontal / vertical -6 dB
    /// coverage angles at and above ~1 kHz; `rear_db` (`<= 0`) is the floor of the pattern;
    /// `up` fixes which way "vertical" is (any non-parallel vector, usually +Y).
    Horn { h_deg: f32, v_deg: f32, rear_db: f32, up: [f32; 3] },
}

/// Coverage multiplier per octave band (62.5 Hz .. 8 kHz) of the horn: wider at low frequencies.
pub const HORN_COVERAGE_WIDENING: [f32; 8] = [2.5, 2.0, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0];
/// Fraction of the horn's rear attenuation reached per band: shallower at low frequencies.
pub const HORN_REAR_DEPTH: [f32; 8] = [0.4, 0.5, 0.65, 0.8, 1.0, 1.0, 1.0, 1.0];

impl EmitterPattern {
    /// A horn with the given -6 dB full coverage angles, a -30 dB rear floor and +Y up.
    pub fn horn(h_deg: f32, v_deg: f32) -> Self {
        EmitterPattern::Horn { h_deg, v_deg, rear_db: -30.0, up: [0.0, 1.0, 0.0] }
    }

    /// `true` if the pattern is exactly 1 in every direction (the callers skip all work).
    pub fn is_omni(&self) -> bool {
        match self {
            EmitterPattern::Omni => true,
            EmitterPattern::CardioidFamily { directivity } => !(directivity.is_finite() && *directivity > 0.0),
            _ => false,
        }
    }

    /// Per-band linear amplitude gain for a sound leaving in direction `dir` from an emitter
    /// whose forward axis is `forward` (neither needs to be normalised). Degenerate input reads
    /// as on-axis (gain 1). Never below [`MIN_DIRECTIVITY_GAIN`] (-60 dB), never above 1.
    pub fn band_gains(&self, forward: [f32; 3], dir: [f32; 3]) -> Band8 {
        self.band_gains_up(forward, None, dir)
    }

    fn band_gains_up(&self, forward: [f32; 3], up_override: Option<[f32; 3]>, dir: [f32; 3]) -> Band8 {
        if self.is_omni() {
            return Band8::splat(1.0);
        }
        let (Some(f), Some(d)) = (normalize(forward), normalize(dir)) else {
            return Band8::splat(1.0);
        };
        let cos = (f[0] * d[0] + f[1] * d[1] + f[2] * d[2]).clamp(-1.0, 1.0);
        match self {
            EmitterPattern::Omni => Band8::splat(1.0),
            EmitterPattern::CardioidFamily { directivity } => pattern_band_gains(*directivity, cos),
            EmitterPattern::Supercardioid => first_order(0.37, cos),
            EmitterPattern::Hypercardioid => first_order(0.25, cos),
            EmitterPattern::SoundCone { inner_deg, outer_deg, outer_gain_db } => {
                let inner = inner_deg.clamp(0.0, 360.0) * 0.5;
                let outer = outer_deg.clamp(0.0, 360.0).max(inner_deg.clamp(0.0, 360.0)) * 0.5;
                let outer_db = outer_gain_db.clamp(-120.0, 0.0);
                let theta = cos.acos().to_degrees();
                let db = if theta <= inner {
                    0.0
                } else if theta >= outer || outer <= inner {
                    outer_db
                } else {
                    outer_db * (theta - inner) / (outer - inner)
                };
                Band8::splat(db_to_gain(db))
            }
            EmitterPattern::Horn { h_deg, v_deg, rear_db, up } => {
                let up = up_override.unwrap_or(*up);
                let Some((r, u)) = horn_frame(f, up) else {
                    return Band8::splat(1.0);
                };
                let x = d[0] * r[0] + d[1] * r[1] + d[2] * r[2];
                let y = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
                let theta = cos.acos();
                let phi = y.atan2(x);
                let half_h = (h_deg.clamp(1.0, 359.0) * 0.5).to_radians();
                let half_v = (v_deg.clamp(1.0, 359.0) * 0.5).to_radians();
                let (cp, sp) = (phi.cos(), phi.sin());
                // Half-angle of the elliptical cone in the direction phi.
                let cov = 1.0 / ((cp * cp) / (half_h * half_h) + (sp * sp) / (half_v * half_v)).sqrt();
                let rear = rear_db.clamp(-120.0, 0.0);
                let mut g = [0.0_f32; 8];
                for (b, out) in g.iter_mut().enumerate() {
                    let a = theta / (cov * HORN_COVERAGE_WIDENING[b]);
                    let db = (-6.0 * a * a).max(rear * HORN_REAR_DEPTH[b]);
                    *out = db_to_gain(db);
                }
                Band8::new(g)
            }
        }
    }

    /// Per-band mean-square of the pattern over the whole sphere ("diffuse-field power",
    /// relative to an omnidirectional emitter with the same on-axis level).
    pub fn diffuse_field_power(&self) -> Band8 {
        match self {
            _ if self.is_omni() => Band8::splat(1.0),
            EmitterPattern::CardioidFamily { directivity } => cardioid_diffuse_power(*directivity),
            _ => {
                // Equal-area Fibonacci sphere around a fixed frame (forward +Z, up +Y); the mean
                // square does not depend on the frame.
                const N: usize = 1024;
                let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
                let mut acc = [0.0_f64; 8];
                for i in 0..N {
                    let z = 1.0 - 2.0 * (i as f32 + 0.5) / N as f32;
                    let rho = (1.0 - z * z).max(0.0).sqrt();
                    let phi = golden * i as f32;
                    let dir = [rho * phi.cos(), rho * phi.sin(), z];
                    let g = self.band_gains_up([0.0, 0.0, 1.0], Some([0.0, 1.0, 0.0]), dir);
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
        }
    }

    /// Scalar reverb-send gain: the late field follows the emitter's total radiated power, so
    /// `sqrt` of the band mean of [`Self::diffuse_field_power`]. Exactly 1.0 for omni.
    pub fn diffuse_send_gain(&self) -> f32 {
        if self.is_omni() {
            return 1.0;
        }
        let p = self.diffuse_field_power();
        (p.0.iter().sum::<f32>() / 8.0).sqrt()
    }
}

/// Physical aperture of an emitter, used to place the soft-occlusion probe points.
#[derive(Clone, Debug, PartialEq)]
pub enum EmitterShape {
    /// The historic behaviour: a 0.35 m disc facing the listener. Bit-identical to before.
    Default,
    /// A point: every probe ray targets the emitter position (occlusion is all or nothing).
    Point,
    /// A ball of `radius` metres: probes fill its volume (depth matters for thick occluders).
    Sphere { radius: f32 },
    /// A disc of `radius` metres facing the listener.
    Disc { radius: f32 },
    /// A line array: probes along `axis` over `+-half_length` metres.
    Line { half_length: f32, axis: [f32; 3] },
}

const GOLDEN_ANGLE: f32 = 2.399_963_2;

impl EmitterShape {
    /// Offset from the emitter position of probe `i` of `n` (`0 <= i < n`; the centre probe is
    /// separate and not counted here). `u`, `w` span the plane facing the listener and `view`
    /// points from the emitter toward the listener (an orthonormal frame). Deterministic;
    /// `Default` reproduces the legacy 0.35 m golden-angle disc exactly (callers keep their own
    /// legacy code for it, this returns the same numbers).
    pub fn probe_offset(&self, i: usize, n: usize, u: [f32; 3], w: [f32; 3], view: [f32; 3]) -> [f32; 3] {
        let n = n.max(1) as f32;
        let fi = i as f32;
        match self {
            EmitterShape::Point => [0.0; 3],
            EmitterShape::Default => disc_offset(LEGACY_RADIUS, fi, n, u, w),
            EmitterShape::Disc { radius } => disc_offset(radius.max(0.0), fi, n, u, w),
            EmitterShape::Sphere { radius } => {
                let r = radius.max(0.0);
                let rho = r * ((fi + 0.5) / n).cbrt();
                let z = 1.0 - 2.0 * (fi + 0.5) / n;
                let s = (1.0 - z * z).max(0.0).sqrt();
                let phi = fi * GOLDEN_ANGLE;
                let (cx, cy) = (s * phi.cos(), s * phi.sin());
                [
                    rho * (u[0] * cx + w[0] * cy + view[0] * z),
                    rho * (u[1] * cx + w[1] * cy + view[1] * z),
                    rho * (u[2] * cx + w[2] * cy + view[2] * z),
                ]
            }
            EmitterShape::Line { half_length, axis } => {
                let Some(a) = normalize(*axis) else { return [0.0; 3] };
                let t = (-1.0 + 2.0 * (fi + 0.5) / n) * half_length.max(0.0);
                [a[0] * t, a[1] * t, a[2] * t]
            }
        }
    }
}

/// Legacy occlusion source radius (metres): `cpu_simd::OCCLUSION_SOURCE_RADIUS`.
pub const LEGACY_RADIUS: f32 = 0.35;

fn disc_offset(r: f32, fi: f32, n: f32, u: [f32; 3], w: [f32; 3]) -> [f32; 3] {
    let rr = r * ((fi + 0.5) / n).sqrt();
    let (st, ct) = (fi * GOLDEN_ANGLE).sin_cos();
    [
        (u[0] * ct + w[0] * st) * rr,
        (u[1] * ct + w[1] * st) * rr,
        (u[2] * ct + w[2] * st) * rr,
    ]
}

/// Pattern + shape of an emitter as stored on a scene output. Build with [`EmitterModel::new`]
/// (it caches the diffuse-field gain, which costs a sphere integration).
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterModel {
    /// Explicit pattern. `None` = legacy: derived from `SceneOutputConfig::directivity`
    /// (cardioid family) when an orientation is set, omni otherwise.
    pub pattern: Option<EmitterPattern>,
    /// Physical aperture for the occlusion probes.
    pub shape: EmitterShape,
    /// Cached `pattern.diffuse_send_gain()` (1.0 when `pattern` is `None`).
    pub diffuse_gain: f32,
}

impl Default for EmitterModel {
    fn default() -> Self {
        Self { pattern: None, shape: EmitterShape::Default, diffuse_gain: 1.0 }
    }
}

impl EmitterModel {
    pub fn new(pattern: Option<EmitterPattern>, shape: EmitterShape) -> Self {
        let diffuse_gain = pattern.as_ref().map_or(1.0, |p| p.diffuse_send_gain());
        Self { pattern, shape, diffuse_gain }
    }
}

/// What a backend needs to know about an emitter: the pattern in its resolved form, the forward
/// axis and the shape. `pattern` is `None` for an omnidirectional emitter.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterTrace {
    pub pattern: Option<EmitterPattern>,
    pub forward: [f32; 3],
    pub shape: EmitterShape,
}

impl EmitterTrace {
    /// Per-band pattern gain toward `dir` (all ones when omnidirectional).
    pub fn band_gains(&self, dir: [f32; 3]) -> Band8 {
        match &self.pattern {
            Some(p) => p.band_gains(self.forward, dir),
            None => Band8::splat(1.0),
        }
    }
}

fn first_order(a: f32, cos: f32) -> Band8 {
    let p = (a + (1.0 - a) * cos).abs().clamp(MIN_DIRECTIVITY_GAIN, 1.0);
    let mut g = [0.0_f32; 8];
    for (b, out) in g.iter_mut().enumerate() {
        *out = p.powf(BAND_EXPONENTS[b]);
    }
    Band8::new(g)
}

fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0).clamp(MIN_DIRECTIVITY_GAIN, 1.0)
}

/// Orthonormal (right, up) completing `forward` (unit) for the horn's coverage axes.
fn horn_frame(f: [f32; 3], up: [f32; 3]) -> Option<([f32; 3], [f32; 3])> {
    let mut u0 = normalize(up).unwrap_or([0.0, 1.0, 0.0]);
    let mut r = cross(f, u0);
    if dot(r, r) < 1e-8 {
        // `up` parallel to forward: pick any perpendicular axis.
        u0 = if f[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        r = cross(f, u0);
    }
    let r = normalize(r)?;
    let u = normalize(cross(r, f))?;
    Some((r, u))
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 && l.is_finite() {
        Some([v[0] / l, v[1] / l, v[2] / l])
    } else {
        None
    }
}
