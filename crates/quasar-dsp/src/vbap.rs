//! Vector-Base Amplitude Panning (Pulkki) with constant-power normalisation.
//!
//! [`VbapPanner::new`] runs on the API thread (it allocates): it normalises the
//! speaker directions, picks speaker pairs (planar layouts) or hull triplets
//! (layouts that span elevation) and precomputes their matrix inverses.
//! [`VbapPanner::gains`] is zero-allocation and audio-thread safe.
//!
//! Conventions (shared with the rest of Quasar): azimuth 0 = straight ahead
//! (-Z), +azimuth toward +X (right); elevation +up (+Y).
//! `dir = [sin(az)cos(el), sin(el), -cos(az)cos(el)]`.
//!
//! Guarantees of [`VbapPanner::gains`]:
//! * `sum(g^2) == 1` for every direction (constant power; not amplitude-sum),
//!   as long as at least one non-LFE speaker exists.
//! * A source exactly at a speaker gives gain 1 there and 0 elsewhere.
//! * Gains are continuous in `(az, el)`, including at pair/triplet borders
//!   and across the +-pi wrap. LFE slots always receive exactly 0.
//!
//! Coverage:
//! * Planar layouts (all speakers within [`PLANAR_MAX_ELEVATION`] of the
//!   horizon): pairs are chosen on azimuth only. Elevation does not move the
//!   image but widens it: beyond the planar band a power fraction
//!   `w = sin^2(pi/2 * t)` (`t` = elevation excess normalised to the pole) of a
//!   diffuse image (all speakers equal, `1/M` power each) is mixed in, so
//!   `g_i = sqrt((1 - w) g_pair_i^2 + w / M)`. Power stays 1, gains are
//!   continuous in az and el, and at the poles `w = 1` so every azimuth gives
//!   the same diffuse gains (an overhead source is "everywhere on the ring").
//!   Where two azimuth-adjacent speakers are >= 180 degrees apart (e.g. the
//!   rear of a stereo pair) VBAP is undefined, so that arc is covered by a
//!   constant-power sine/cosine crossfade between the two bounding speakers
//!   whose position `s(t)` across the gap (`t` = fraction of the gap, 0..1) is
//!   shaped as `s = t^4 / (t^4 + (1-t)^4)`: odd-symmetric about the gap centre,
//!   smooth, and almost 0 / 1 near each speaker, so a source at +-90 degrees on
//!   stereo goes (almost) wholly to the nearer speaker (far-speaker gain
//!   < 0.01, issue #126) while the rear (180 degrees) still plays with both
//!   speakers at 0.707, with no hard cut.
//! * 3D layouts: triplets come from the convex hull of the speaker directions.
//!   Imaginary speakers are added at +-Y (unless a real speaker sits there) so
//!   the hull closes over the whole sphere; energy panned to an imaginary
//!   speaker is spread equally over its hull neighbours and renormalised.
//!   Layouts that do not surround the listener at all (front-only, half domes)
//!   leave part of the sphere outside that hull. Those are closed with further
//!   imaginary speakers (axis / cube-diagonal directions, farthest-first, until
//!   the origin is strictly inside the hull), so the triplets tile the whole
//!   sphere and the gains are continuous everywhere by construction: a rear
//!   source in a front-only layout is carried by the real speakers bordering
//!   the imaginary ones that cover it.

use std::f64::consts::PI as PI64;

/// Speakers within this elevation (radians, ~5 degrees) of the horizon count as planar.
pub const PLANAR_MAX_ELEVATION: f32 = 0.0873;

/// Pairs whose angular gap is at least this (just under pi) use the crossfade law.
const MAX_VBAP_GAP: f32 = std::f32::consts::PI - 1e-3;
/// Tolerance for "inside the triplet" tests.
const INSIDE_EPS: f32 = 1e-6;
/// Upper bound on imaginary speakers: the number of distinct closing candidates
/// (6 axes + 8 cube diagonals), the +-Y poles included.
const MAX_IMAGINARY: usize = 14;

/// Candidate directions for imaginary speakers: the 6 axes and 8 cube diagonals.
fn imaginary_candidates() -> impl Iterator<Item = [f64; 3]> {
    let h = 1.0_f64 / 3.0_f64.sqrt();
    let axes = [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let diag = [
        [h, h, h], [-h, h, h], [h, h, -h], [-h, h, -h],
        [h, -h, h], [-h, -h, h], [h, -h, -h], [-h, -h, -h],
    ];
    axes.into_iter().chain(diag)
}

/// Power fraction `w` of the diffuse (all-speakers-equal) image mixed into a
/// planar layout's pair panning for a source at elevation `el`.
///
/// Zero inside the planar band (`|el| <= PLANAR_MAX_ELEVATION`, so speakers that
/// sit a few degrees off the horizon still get exact gain 1), then
/// `sin^2(pi/2 * t)` of the normalised excess elevation: C1 at both ends and
/// exactly 1 at the poles, where azimuth is undefined and every azimuth must
/// give the same gains.
fn planar_spread(el: f32) -> f32 {
    let lo = PLANAR_MAX_ELEVATION;
    let t = ((el.abs() - lo) / (std::f32::consts::FRAC_PI_2 - lo)).clamp(0.0, 1.0);
    let s = (t * std::f32::consts::FRAC_PI_2).sin();
    s * s
}

#[derive(Clone, Debug)]
struct PlanarPair {
    /// Output slots of the bounding speakers.
    a: usize,
    b: usize,
    /// Azimuth of `a` (radians) and angular gap from `a` to `b` going positive (0..2pi].
    start: f32,
    gap: f32,
    /// Row-major inverse of `[l_a l_b]` (2D, x/z plane); `None` => crossfade law.
    inv: Option<[f32; 4]>,
}

#[derive(Clone, Debug)]
struct Triplet {
    /// Point indices (real points first, then imaginary).
    idx: [usize; 3],
    /// Rows of the inverse of `[l_a l_b l_c]`.
    inv: [f32; 9],
}

#[derive(Clone, Debug)]
enum Kind {
    Silent,
    Mono(usize),
    Planar {
        pairs: Vec<PlanarPair>,
        /// Output slots of every active speaker (diffuse spread for elevated sources).
        slots: Vec<usize>,
    },
    Hull {
        /// Output slot of every real point.
        pt_out: Vec<usize>,
        /// Unit direction of every real point (nearest-speaker fallback).
        pt_dir: Vec<[f32; 3]>,
        tris: Vec<Triplet>,
        /// Real output slots standing in for imaginary speaker `k` (point `pt_out.len() + k`).
        imag_nbrs: Vec<Vec<usize>>,
    },
}

/// Constant-power VBAP panner for a fixed speaker layout.
#[derive(Clone, Debug)]
pub struct VbapPanner {
    num_outputs: usize,
    kind: Kind,
}

fn unit(p: [f32; 3]) -> Option<[f64; 3]> {
    let v = [p[0] as f64, p[1] as f64, p[2] as f64];
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l <= 1e-6 || !l.is_finite() {
        None
    } else {
        Some([v[0] / l, v[1] / l, v[2] / l])
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Source direction vector for (azimuth, elevation).
#[inline]
fn dir_vec(az: f32, el: f32) -> [f32; 3] {
    let (sa, ca) = az.sin_cos();
    let (se, ce) = el.sin_cos();
    [sa * ce, se, -ca * ce]
}

impl VbapPanner {
    /// Build a panner. `positions` are speaker positions (treated as directions
    /// from the listener origin; need not be unit length). `lfe` lists output
    /// slots that never receive panned signal. API thread only (allocates).
    pub fn new(positions: &[[f32; 3]], lfe: &[usize]) -> Self {
        let num_outputs = positions.len();
        // Active (non-LFE, non-degenerate) speakers: (out slot, unit dir).
        let mut act: Vec<(usize, [f64; 3])> = Vec::new();
        for (i, p) in positions.iter().enumerate() {
            if lfe.contains(&i) {
                continue;
            }
            if let Some(u) = unit(*p) {
                act.push((i, u));
            }
        }

        let kind = if act.is_empty() {
            Kind::Silent
        } else if act.len() == 1 {
            Kind::Mono(act[0].0)
        } else if act.iter().all(|(_, u)| u[1].abs() <= (PLANAR_MAX_ELEVATION as f64).sin()) {
            Self::build_planar(&act)
        } else {
            Self::build_hull(&act)
        };
        Self { num_outputs, kind }
    }

    /// Number of output slots (`positions.len()` given to [`VbapPanner::new`]).
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn build_planar(act: &[(usize, [f64; 3])]) -> Kind {
        // (azimuth, slot) sorted ascending.
        let mut sp: Vec<(f64, usize)> = act
            .iter()
            .map(|(i, u)| (u[0].atan2(-u[2]), *i))
            .collect();
        sp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let m = sp.len();
        let mut pairs = Vec::with_capacity(m);
        for i in 0..m {
            let (az_a, a) = sp[i];
            let (az_b, b) = sp[(i + 1) % m];
            let gap = (az_b - az_a).rem_euclid(2.0 * PI64);
            let gap = if m == 2 && i == 1 && gap < 1e-9 { 2.0 * PI64 } else { gap };
            if gap < 1e-6 {
                continue; // coincident speakers: skip
            }
            let inv = if (gap as f32) < MAX_VBAP_GAP {
                let (ax, az) = (az_a.sin(), -az_a.cos());
                let (bx, bz) = (az_b.sin(), -az_b.cos());
                let det = ax * bz - bx * az;
                if det.abs() < 1e-9 {
                    None
                } else {
                    Some([
                        (bz / det) as f32,
                        (-bx / det) as f32,
                        (-az / det) as f32,
                        (ax / det) as f32,
                    ])
                }
            } else {
                None
            };
            pairs.push(PlanarPair { a, b, start: az_a as f32, gap: gap as f32, inv });
        }
        Kind::Planar { pairs, slots: act.iter().map(|(i, _)| *i).collect() }
    }

    /// Hull triplets of `pts` that can span a source cone (origin on the inner
    /// side), plus whether the origin lies strictly inside the hull (i.e. the
    /// triplets tile the whole sphere of directions).
    fn hull_triplets(pts: &[[f64; 3]]) -> (Vec<Triplet>, bool) {
        let n = pts.len();
        const EPS: f64 = 1e-7;
        let mut enclosed = n >= 4;
        let mut tris: Vec<Triplet> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let (a, b, c) = (pts[i], pts[j], pts[k]);
                    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let mut nrm = cross(ab, ac);
                    let nl = dot(nrm, nrm).sqrt();
                    if nl < 1e-9 {
                        continue;
                    }
                    nrm = [nrm[0] / nl, nrm[1] / nl, nrm[2] / nl];
                    let mut d = dot(nrm, a);
                    // Orient the normal outward: every other point on the inner side.
                    let (mut all_below, mut all_above) = (true, true);
                    for (p, pt) in pts.iter().enumerate() {
                        if p == i || p == j || p == k {
                            continue;
                        }
                        let v = dot(nrm, *pt) - d;
                        if v > EPS {
                            all_below = false;
                        }
                        if v < -EPS {
                            all_above = false;
                        }
                    }
                    if !all_below && !all_above {
                        continue; // not a hull face
                    }
                    if !all_below {
                        nrm = [-nrm[0], -nrm[1], -nrm[2]];
                        d = -d;
                    }
                    // A hull face whose plane passes through (or behind) the
                    // origin cannot span a source cone, and means the hull does
                    // not surround the listener.
                    if d < 1e-4 {
                        enclosed = false;
                        continue;
                    }
                    let mut coplanar: Vec<usize> = Vec::new();
                    for (p, pt) in pts.iter().enumerate() {
                        if p == i || p == j || p == k {
                            continue;
                        }
                        if dot(nrm, *pt) - d >= -EPS {
                            coplanar.push(p);
                        }
                    }
                    if !coplanar.is_empty() {
                        // Cocircular set: canonical fan triangulation from its lowest index.
                        let mut set = vec![i, j, k];
                        set.extend_from_slice(&coplanar);
                        let m = *set.iter().min().unwrap_or(&i);
                        if m != i {
                            continue; // i is the smallest of the triple; m must be in the triple
                        }
                        // Basis in the plane.
                        let helper = if nrm[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                        let u = cross(nrm, helper);
                        let ul = dot(u, u).sqrt();
                        let u = [u[0] / ul, u[1] / ul, u[2] / ul];
                        let v = cross(nrm, u);
                        let ang = |p: usize| dot(pts[p], v).atan2(dot(pts[p], u));
                        let am = ang(m);
                        let mut others: Vec<(f64, usize)> = set
                            .iter()
                            .filter(|&&p| p != m)
                            .map(|&p| ((ang(p) - am).rem_euclid(2.0 * PI64), p))
                            .collect();
                        others.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
                        let pos_of = |p: usize| others.iter().position(|o| o.1 == p);
                        match (pos_of(j), pos_of(k)) {
                            (Some(pj), Some(pk)) if pk == pj + 1 || pj == pk + 1 => {}
                            _ => continue,
                        }
                    }
                    let det = dot(a, cross(b, c));
                    if det.abs() < 1e-9 {
                        continue;
                    }
                    let r0 = cross(b, c);
                    let r1 = cross(c, a);
                    let r2 = cross(a, b);
                    let inv = [
                        (r0[0] / det) as f32, (r0[1] / det) as f32, (r0[2] / det) as f32,
                        (r1[0] / det) as f32, (r1[1] / det) as f32, (r1[2] / det) as f32,
                        (r2[0] / det) as f32, (r2[1] / det) as f32, (r2[2] / det) as f32,
                    ];
                    tris.push(Triplet { idx: [i, j, k], inv });
                }
            }
        }
        let tiled = enclosed && !tris.is_empty();
        (tris, tiled)
    }

    fn build_hull(act: &[(usize, [f64; 3])]) -> Kind {
        let mut pts: Vec<[f64; 3]> = act.iter().map(|(_, u)| *u).collect();
        let pt_out: Vec<usize> = act.iter().map(|(i, _)| *i).collect();
        let nreal = pts.len();
        // Imaginary speakers at the poles unless a real one is (almost) there.
        for y in [1.0_f64, -1.0] {
            if !pts.iter().any(|p| p[1] * y >= 1.0 - 1e-6) {
                pts.push([0.0, y, 0.0]);
            }
        }
        // Layouts that do not surround the listener (front-only, half domes, ...)
        // leave part of the sphere outside the hull. Close it with further
        // imaginary speakers (greedy: the candidate farthest from every existing
        // point first) until the origin is strictly inside the hull. The hull of
        // real + imaginary points then tiles the whole sphere, so the gains are
        // continuous everywhere by construction.
        let (mut tris, mut enclosed) = Self::hull_triplets(&pts);
        while !enclosed && pts.len() - nreal < MAX_IMAGINARY {
            let mut best: Option<([f64; 3], f64)> = None;
            for c in imaginary_candidates() {
                let nearest = pts.iter().map(|p| dot(*p, c)).fold(f64::NEG_INFINITY, f64::max);
                if nearest >= 1.0 - 1e-6 {
                    continue; // already a point there
                }
                if best.map_or(true, |(_, b)| nearest < b - 1e-9) {
                    best = Some((c, nearest));
                }
            }
            let Some((c, _)) = best else { break };
            pts.push(c);
            let r = Self::hull_triplets(&pts);
            tris = r.0;
            enclosed = r.1;
        }

        // Real output slots that stand in for each imaginary speaker: the real
        // points sharing a triplet with it (widening through neighbouring
        // imaginary points if it has none).
        let n_imag = pts.len() - nreal;
        let mut imag_nbrs: Vec<Vec<usize>> = Vec::with_capacity(n_imag);
        for k in 0..n_imag {
            let mut visited = vec![false; pts.len()];
            visited[nreal + k] = true;
            let mut nbrs: Vec<usize> = Vec::new();
            loop {
                let mut grew = false;
                for t in &tris {
                    if !t.idx.iter().any(|&p| visited[p]) {
                        continue;
                    }
                    for &p in &t.idx {
                        if p < nreal {
                            if !nbrs.contains(&pt_out[p]) {
                                nbrs.push(pt_out[p]);
                            }
                        } else if !visited[p] && nbrs.is_empty() {
                            visited[p] = true;
                            grew = true;
                        }
                    }
                }
                if !nbrs.is_empty() || !grew {
                    break;
                }
            }
            imag_nbrs.push(nbrs);
        }

        // With no triplets at all (e.g. every speaker in one vertical plane
        // through the origin) `gains` falls back to the nearest speaker.
        let pt_dir = act
            .iter()
            .map(|(_, u)| [u[0] as f32, u[1] as f32, u[2] as f32])
            .collect();
        Kind::Hull { pt_out, pt_dir, tris, imag_nbrs }
    }

    /// Fill `out` with per-speaker gains for a source at (`az`, `el`) radians.
    ///
    /// Zero-allocation. Slots beyond `num_outputs()` (or beyond `out.len()`)
    /// are left untouched; all other slots are overwritten. LFE slots get 0.
    pub fn gains(&self, az: f32, el: f32, out: &mut [f32]) {
        let n = self.num_outputs.min(out.len());
        for g in out.iter_mut().take(n) {
            *g = 0.0;
        }
        let az = if az.is_finite() { az } else { 0.0 };
        let el = if el.is_finite() { el } else { 0.0 };

        match &self.kind {
            Kind::Silent => {}
            Kind::Mono(i) => {
                if *i < n {
                    out[*i] = 1.0;
                }
            }
            Kind::Planar { pairs, slots } => {
                if pairs.is_empty() {
                    return;
                }
                let two_pi = std::f32::consts::TAU;
                let mut chosen = &pairs[pairs.len() - 1];
                let mut chosen_rel = (az - chosen.start).rem_euclid(two_pi);
                for p in pairs.iter() {
                    let rel = (az - p.start).rem_euclid(two_pi);
                    if rel <= p.gap + INSIDE_EPS {
                        chosen = p;
                        chosen_rel = rel;
                        break;
                    }
                }
                let (ga, gb) = match chosen.inv {
                    Some(inv) => {
                        let (px, pz) = (az.sin(), -az.cos());
                        (
                            (inv[0] * px + inv[1] * pz).max(0.0),
                            (inv[2] * px + inv[3] * pz).max(0.0),
                        )
                    }
                    None => {
                        let t = (chosen_rel / chosen.gap).clamp(0.0, 1.0);
                        // Shaped position: lateral sources stay with the near speaker.
                        let (t4, u4) = (t.powi(4), (1.0 - t).powi(4));
                        let t = t4 / (t4 + u4);
                        let a = t * std::f32::consts::FRAC_PI_2;
                        (a.cos().max(0.0), a.sin().max(0.0))
                    }
                };
                let norm = (ga * ga + gb * gb).sqrt();
                if norm > 1e-12 {
                    if chosen.a < n {
                        out[chosen.a] = ga / norm;
                    }
                    if chosen.b < n {
                        out[chosen.b] += gb / norm;
                    }
                    // Elevation: widen toward a diffuse (all speakers equal) image as
                    // |el| grows, in the power domain so sum(g^2) stays 1.
                    let w = planar_spread(el);
                    if w > 0.0 && !slots.is_empty() {
                        let d2 = w / slots.len() as f32;
                        for &s in slots.iter() {
                            if s < n {
                                out[s] = ((1.0 - w) * out[s] * out[s] + d2).sqrt();
                            }
                        }
                    }
                }
            }
            Kind::Hull { pt_out, pt_dir, tris, imag_nbrs } => {
                let s = dir_vec(az, el);
                let nreal = pt_out.len();
                // Pick the triplet with the largest minimum gain (inside => >= 0).
                let mut best: Option<(&Triplet, [f32; 3])> = None;
                let mut best_min = f32::NEG_INFINITY;
                for t in tris.iter() {
                    let m = &t.inv;
                    let g = [
                        m[0] * s[0] + m[1] * s[1] + m[2] * s[2],
                        m[3] * s[0] + m[4] * s[1] + m[5] * s[2],
                        m[6] * s[0] + m[7] * s[1] + m[8] * s[2],
                    ];
                    let mn = g[0].min(g[1]).min(g[2]);
                    if mn > best_min {
                        best_min = mn;
                        best = Some((t, g));
                    }
                }
                let Some((t, g)) = best else {
                    // No triplets: nearest speaker.
                    let mut bi = 0;
                    let mut bd = f32::NEG_INFINITY;
                    for (i, d) in pt_dir.iter().enumerate() {
                        let v = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
                        if v > bd {
                            bd = v;
                            bi = i;
                        }
                    }
                    if let Some(&o) = pt_out.get(bi) {
                        if o < n {
                            out[o] = 1.0;
                        }
                    }
                    return;
                };
                let mut imag = [0.0_f32; MAX_IMAGINARY];
                for v in 0..3 {
                    let gv = g[v].max(0.0);
                    let p = t.idx[v];
                    if p < nreal {
                        let o = pt_out[p];
                        if o < n {
                            out[o] += gv;
                        }
                    } else {
                        imag[(p - nreal).min(MAX_IMAGINARY - 1)] += gv;
                    }
                }
                for k in 0..imag_nbrs.len().min(MAX_IMAGINARY) {
                    if imag[k] > 0.0 && !imag_nbrs[k].is_empty() {
                        let w = imag[k] / (imag_nbrs[k].len() as f32).sqrt();
                        for &o in imag_nbrs[k].iter() {
                            if o < n {
                                out[o] += w;
                            }
                        }
                    }
                }
                let pow: f32 = out.iter().take(n).map(|g| g * g).sum();
                if pow > 1e-12 {
                    let inv = 1.0 / pow.sqrt();
                    for g in out.iter_mut().take(n) {
                        *g *= inv;
                    }
                }
            }
        }
    }
}

/// `dst[i] += src[i] * (g0 + (g1 - g0) * (i + 1) / n)` for `n = min(dst, src)` samples: the gain
/// ramps linearly from `g0` (end of the previous block) to `g1` (end of this block). This is the
/// per-speaker decode loop of the panner; `g0 == g1` takes the plain-gain path.
/// Allocation-free.
#[inline]
pub fn ramp_add(dst: &mut [f32], src: &[f32], g0: f32, g1: f32) {
    let n = dst.len().min(src.len());
    if g0 == g1 {
        for i in 0..n {
            dst[i] += src[i] * g1;
        }
    } else {
        let step = (g1 - g0) / n.max(1) as f32;
        for i in 0..n {
            dst[i] += src[i] * (g0 + step * (i + 1) as f32);
        }
    }
}
