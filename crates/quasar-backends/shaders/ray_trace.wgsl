// Quasar WGPU compute shader: direct-path occlusion probes and image-source
// early-reflection validation. One workgroup (64 threads) per source/listener
// query, one dispatch for the whole batch.
//
// What runs where (see wgpu_compute.rs): this shader does the GEOMETRY (ray /
// triangle work). It records raw geometric facts (which materials a probe ray
// crosses and at what incidence cosine, the shortest detour length, validated
// reflection paths with their bounce points). The MATERIAL-dependent arithmetic
// (per-band transmission / absorption through the `MaterialProvider` trait object),
// distance law, air absorption, diffraction blend, ranking and the statistical late
// reverb are evaluated on the host with the same code the CPU backend uses.
//
// Geometry uses a host-built flattened median BVH uploaded alongside the triangle
// list. Leaves contain at most four original triangle indices; traversal uses a
// bounded 64-entry private stack (tree depth is <= 32 for supported buffer sizes).
//
// ----------------------------------------------------------------------------
// MEMORY LAYOUT. Every struct below is built only from 16-byte `vec4` fields and
// scalar arrays in the `storage` address space, so there is NO implicit padding;
// the Rust mirrors in wgpu_compute.rs are `repr(C)` and their sizes are asserted in
// a unit test against the byte sizes written here:
//   Params   416   Query 32   Tri 80   Plane 64   Edge 32
//   Crossing 32    RayOut 288 (16 + 16 + 8*32 -> 16 header + 16 first + 256)
//   Head 3776 (16 + 16 + 13*288)   Cand 224 (16 + 16 + 32 + 32 + 128)
// ----------------------------------------------------------------------------

const N_RAYS: u32 = 13u;        // = OCCLUSION_RAYS in cpu_simd.rs
const MAX_CROSSINGS: u32 = 8u;  // = OCCLUSION_MAX_CROSSINGS
const N_DETOUR: u32 = 8u;       // = OCCLUSION_DETOUR_DIRS
const MAX_ORDER: u32 = 8u;      // = MAX_IMAGE_ORDER
const NONE: u32 = 0xffffffffu;

struct Params {
    // x: queries in this dispatch, y: triangles, z: mirror planes, w: max reflection order
    counts0: vec4<u32>,
    // x: max candidate paths per query, y: bisection steps, z: max image nodes per thread, w: BVH node count
    counts1: vec4<u32>,
    // x: occlusion eps, y: detour min offset, z: detour max offset, w: detour margin
    limits0: vec4<f32>,
    // x: max reflection distance, y: edge fade width, z: plane side eps, w: plane box pad
    limits1: vec4<f32>,
    // x: barycentric slack of the triangle test
    limits2: vec4<f32>,
    // (r cos t, r sin t) offsets of the 13 occlusion targets in the (u, w) plane
    disc: array<vec4<f32>, 13>,
    // (cos phi, sin phi) of the 8 detour directions
    detour: array<vec4<f32>, 8>,
}

struct Query {
    source: vec4<f32>,   // w = 1 valid, 0 = non-finite input (skipped)
    listener: vec4<f32>,
}

struct Tri {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    n: vec4<f32>,        // unit normal
    mat: vec4<u32>,      // x = material handle
}

struct BvhNode {
    bmin: vec4<f32>,
    bmax: vec4<f32>,
    info: vec4<u32>, // internal: left/right; leaf: count and flag (w = 1)
    tri: vec4<u32>,  // original triangle indices for a leaf
}

struct Plane {
    no: vec4<f32>,       // xyz = canonical unit normal, w = offset (n . p)
    bmin: vec4<f32>,     // w = spread (m) of a merged facet group
    bmax: vec4<f32>,     // w = 1 for a merged (fitted) group, else 0
    ranges: vec4<u32>,   // x: first plane_tris entry, y: count, z: first edge, w: count
}

struct Edge {
    a: vec4<f32>,
    b: vec4<f32>,
}

struct Crossing {
    n_cos: vec4<f32>,    // xyz = triangle normal, w = dot(-dir, normal)
    mat: vec4<u32>,      // x = material handle
}

struct RayOut {
    count: vec4<u32>,    // x = surfaces crossed (MAX_CROSSINGS + 1 = more than that: opaque)
    first: vec4<f32>,    // first hit point
    cr: array<Crossing, 8>,
}

struct Head {
    info: vec4<u32>,     // x = candidate paths found (may exceed the capacity)
    delta: vec4<f32>,    // x = shortest detour extra path, < 0 = no detour / none needed
    rays: array<RayOut, 13>,
}

struct Cand {
    head: vec4<f32>,     // x = edge window, y = total path length
    info: vec4<u32>,     // x = order (bounces)
    seq: array<u32, 8>,  // plane of each bounce, source side first
    tri: array<u32, 8>,  // triangle of each bounce
    pts: array<vec4<f32>, 8>, // bounce points
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> tris: array<Tri>;
@group(0) @binding(3) var<storage, read> planes: array<Plane>;
@group(0) @binding(4) var<storage, read> plane_tris: array<u32>;
@group(0) @binding(5) var<storage, read> edges: array<Edge>;
@group(0) @binding(6) var<storage, read_write> heads: array<Head>;
@group(0) @binding(7) var<storage, read_write> cands: array<Cand>;
@group(0) @binding(8) var<storage, read> bvh: array<BvhNode>;

var<workgroup> n_cand: atomic<u32>;
var<workgroup> g_ref_valid: u32;
var<workgroup> g_ref_tgt: vec3<f32>;
var<workgroup> g_ref_hit: vec3<f32>;
var<workgroup> g_det: array<f32, 8>;

// Per-thread scratch of the image-source search.
var<private> p_seq: array<u32, 8>;
var<private> p_img: array<vec3<f32>, 9>;
var<private> p_pts: array<vec3<f32>, 8>;
var<private> p_tri: array<u32, 8>;
var<private> p_next: array<u32, 8>;

// ---------------------------------------------------------------- ray / triangle

// Moller-Trumbore over [tmin, tmax], edge tolerant; mirrors Triangle::intersect_max.
// Returns the hit distance or -1.
fn tri_hit(i: u32, o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32) -> f32 {
    let t = tris[i];
    let a = t.a.xyz;
    let e1 = t.b.xyz - a;
    let e2 = t.c.xyz - a;
    let h = cross(d, e2);
    let det = dot(e1, h);
    let scale = sqrt(dot(e1, e1) * dot(e2, e2));
    if !(abs(det) > 1e-9 * scale) {
        return -1.0;
    }
    let inv_det = 1.0 / det;
    let s = o - a;
    let u = dot(s, h) * inv_det;
    let eps = params.limits2.x;
    if u < -eps || u > 1.0 + eps {
        return -1.0;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv_det;
    if v < -eps || u + v > 1.0 + eps {
        return -1.0;
    }
    let hit_t = dot(e2, q) * inv_det;
    if !(hit_t >= tmin && hit_t <= tmax) {
        return -1.0;
    }
    return hit_t;
}

fn box_hit(node: BvhNode, o: vec3<f32>, d: vec3<f32>, tmax: f32) -> bool {
    var near_t = 0.0;
    var far_t = tmax;
    for (var axis = 0u; axis < 3u; axis++) {
        if abs(d[axis]) < 1e-20 {
            if o[axis] < node.bmin[axis] || o[axis] > node.bmax[axis] { return false; }
        } else {
            let inv = 1.0 / d[axis];
            let a = (node.bmin[axis] - o[axis]) * inv;
            let b = (node.bmax[axis] - o[axis]) * inv;
            near_t = max(near_t, min(a, b));
            far_t = min(far_t, max(a, b));
            if far_t < near_t { return false; }
        }
    }
    return far_t >= max(near_t, 0.0);
}

// Closest hit: x = distance, y = bitcast triangle index (NONE = miss).
struct Hit {
    t: f32,
    idx: u32,
}

fn closest_hit(o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32) -> Hit {
    var best = tmax;
    var idx = NONE;
    if params.counts1.w == 0u { return Hit(best, idx); }
    var stack: array<u32, 64>;
    var size = 1u;
    stack[0] = 0u;
    loop {
        if size == 0u { break; }
        size -= 1u;
        let node = bvh[stack[size]];
        if !box_hit(node, o, d, best) { continue; }
        if node.info.w == 1u {
            for (var j = 0u; j < node.info.z; j++) {
                let t = tri_hit(node.tri[j], o, d, tmin, best);
                // Preserve the CPU's input-order tie break despite spatial BVH ordering.
                if t >= 0.0 && (idx == NONE || t < best || (t == best && node.tri[j] < idx)) {
                    best = t;
                    idx = node.tri[j];
                }
            }
        } else {
            stack[size] = node.info.y;
            size += 1u;
            stack[size] = node.info.x;
            size += 1u;
        }
    }
    return Hit(best, idx);
}

fn any_hit(o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32) -> bool {
    if params.counts1.w == 0u { return false; }
    var stack: array<u32, 64>;
    var size = 1u;
    stack[0] = 0u;
    loop {
        if size == 0u { break; }
        size -= 1u;
        let node = bvh[stack[size]];
        if !box_hit(node, o, d, tmax) { continue; }
        if node.info.w == 1u {
            for (var j = 0u; j < node.info.z; j++) {
                if tri_hit(node.tri[j], o, d, tmin, tmax) >= 0.0 { return true; }
            }
        } else {
            stack[size] = node.info.y;
            size += 1u;
            stack[size] = node.info.x;
            size += 1u;
        }
    }
    return false;
}

// True if nothing lies between a and b; the ray starts `before` m before a and ends
// `after` m past b. Mirrors CpuSimdComputeBackend::segment_clear_overshoot.
fn segment_clear_overshoot(a: vec3<f32>, b: vec3<f32>, after: f32, before: f32) -> bool {
    let eps = params.limits0.x;
    let len = distance(a, b);
    if len < 2.0 * eps {
        return true;
    }
    let dir = normalize(b - a);
    let origin = a - dir * before;
    return !any_hit(origin, dir, eps, before + len + after - eps);
}

fn basis_u(axis: vec3<f32>) -> vec3<f32> {
    if axis.z < -0.9999999 {
        return vec3<f32>(0.0, -1.0, 0.0);
    }
    let a = 1.0 / (1.0 + axis.z);
    let b = -axis.x * axis.y * a;
    return vec3<f32>(1.0 - axis.x * axis.x * a, b, -axis.x);
}

// ------------------------------------------------------------ direct-path probes

fn occlusion_target(k: u32, src: vec3<f32>, u: vec3<f32>, w: vec3<f32>) -> vec3<f32> {
    let off = params.disc[k];
    return src + u * off.x + w * off.y;
}

// Walk the segment from -> to and record every surface it crosses
// (CpuSimdComputeBackend::segment_transmission).
fn trace_probe(ray_index: u32, org: vec3<f32>, dst: vec3<f32>) {
    let eps = params.limits0.x;
    let total = distance(org, dst);
    var count = 0u;
    var first = vec3<f32>(0.0);
    if total >= 2.0 * eps {
        let dir = normalize(dst - org);
        var origin = org;
        var remaining = total;
        loop {
            if !(remaining > 2.0 * eps) {
                break;
            }
            let h = closest_hit(origin, dir, eps, remaining - eps);
            if h.idx == NONE {
                break;
            }
            let point = origin + dir * h.t;
            if count == 0u {
                first = point;
            }
            count += 1u;
            if count > MAX_CROSSINGS {
                break;
            }
            let tri = tris[h.idx];
            let c = clamp(dot(-dir, tri.n.xyz), -1.0, 1.0);
            heads[ray_index / N_RAYS].rays[ray_index % N_RAYS].cr[count - 1u] =
                Crossing(vec4<f32>(tri.n.xyz, c), vec4<u32>(tri.mat.x, 0u, 0u, 0u));
            origin = point;
            remaining -= h.t;
        }
    }
    heads[ray_index / N_RAYS].rays[ray_index % N_RAYS].count = vec4<u32>(count, 0u, 0u, 0u);
    heads[ray_index / N_RAYS].rays[ray_index % N_RAYS].first = vec4<f32>(first, 0.0);
}

// Extra path |L-P| + |P-S| - |L-S| of the shortest one-point detour around the first
// hit `h` in lateral direction `j` (exponential search then bisection), or -1.
// Mirrors CpuSimdComputeBackend::detour_extra_path for one direction.
fn detour_dir(j: u32, lis: vec3<f32>, tgt: vec3<f32>, h: vec3<f32>, u: vec3<f32>, w: vec3<f32>) -> f32 {
    let margin = params.limits0.w;
    let dcs = params.detour[j];
    let dir = u * dcs.x + w * dcs.y;
    let direct = distance(lis, tgt);
    var lo = 0.0;
    var hi = params.limits0.y;
    var found = false;
    loop {
        if !(hi <= params.limits0.z) {
            break;
        }
        let p = h + dir * hi;
        if segment_clear_overshoot(lis, p, margin, 0.0) && segment_clear_overshoot(p, tgt, 0.0, margin) {
            found = true;
            break;
        }
        lo = hi;
        hi = hi * 2.0;
    }
    if !found {
        return -1.0;
    }
    for (var i = 0u; i < params.counts1.y; i++) {
        let mid = 0.5 * (lo + hi);
        let p = h + dir * mid;
        if segment_clear_overshoot(lis, p, margin, 0.0) && segment_clear_overshoot(p, tgt, 0.0, margin) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let p = h + dir * hi;
    return max(distance(lis, p) + distance(p, tgt) - direct, 0.0);
}

// ----------------------------------------------------- image-source reflections

fn plane_sd(pi: u32, p: vec3<f32>) -> f32 {
    let no = planes[pi].no;
    return dot(no.xyz, p) - no.w;
}

fn plane_mirror(pi: u32, p: vec3<f32>) -> vec3<f32> {
    let d = 2.0 * plane_sd(pi, p);
    return p - d * planes[pi].no.xyz;
}

fn tri_contains(ti: u32, p: vec3<f32>) -> bool {
    let t = tris[ti];
    let v0 = t.c.xyz - t.a.xyz;
    let v1 = t.b.xyz - t.a.xyz;
    let v2 = p - t.a.xyz;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if !(abs(denom) > 1e-20) {
        return false;
    }
    let u = (d11 * d20 - d01 * d21) / denom;
    let v = (d00 * d21 - d01 * d20) / denom;
    let eps = params.limits2.x;
    return u >= -eps && v >= -eps && u + v <= 1.0 + eps;
}

fn point_segment_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ab = b - a;
    let len2 = dot(ab, ab);
    var t = 0.0;
    if len2 > 1e-20 {
        t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    }
    return distance(p, a + ab * t);
}

// Triangle of plane `pi` containing the in-plane point p and the edge window; the
// triangle is NONE when p is outside the surface. Mirrors locate_on_plane.
struct Located {
    tri: u32,
    w: f32,
    p: vec3<f32>, // bounce point (snapped onto the member triangle for merged groups)
}

// Two-sided line / triangle test within |t| <= reach (same edge tolerance as tri_hit):
// x = 1 on a hit, y = signed distance along d.
fn tri_line_hit(i: u32, o: vec3<f32>, d: vec3<f32>, reach: f32) -> vec2<f32> {
    let t = tris[i];
    let a = t.a.xyz;
    let e1 = t.b.xyz - a;
    let e2 = t.c.xyz - a;
    let h = cross(d, e2);
    let det = dot(e1, h);
    let scale = sqrt(dot(e1, e1) * dot(e2, e2));
    if !(abs(det) > 1e-9 * scale) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let s = o - a;
    let u = dot(s, h) * inv_det;
    let eps = params.limits2.x;
    if u < -eps || u > 1.0 + eps {
        return vec2<f32>(0.0, 0.0);
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv_det;
    if v < -eps || u + v > 1.0 + eps {
        return vec2<f32>(0.0, 0.0);
    }
    let hit_t = dot(e2, q) * inv_det;
    if !(hit_t >= -reach && hit_t <= reach) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(1.0, hit_t);
}

fn locate_on_plane(pi: u32, p0: vec3<f32>) -> Located {
    let pl = planes[pi];
    let pad = params.limits1.w;
    if any(p0 < pl.bmin.xyz - vec3<f32>(pad)) || any(p0 > pl.bmax.xyz + vec3<f32>(pad)) {
        return Located(NONE, 0.0, p0);
    }
    var found = NONE;
    var p = p0;
    if pl.bmax.w > 0.5 {
        // Merged facet group: the bounce line along the group normal must hit a MEMBER
        // triangle (the hit closest to the fitted plane wins); the bounce point is that hit.
        let reach = pl.bmin.w + 2.0 * pad;
        var best_t = 3.0e38;
        for (var i = 0u; i < pl.ranges.y; i++) {
            let ti = plane_tris[pl.ranges.x + i];
            let r = tri_line_hit(ti, p0, pl.no.xyz, reach);
            if r.x > 0.5 && abs(r.y) < abs(best_t) {
                best_t = r.y;
                found = ti;
            }
        }
        if found != NONE {
            p = p0 + pl.no.xyz * best_t;
        }
    } else {
        for (var i = 0u; i < pl.ranges.y; i++) {
            let ti = plane_tris[pl.ranges.x + i];
            if tri_contains(ti, p0) {
                found = ti;
                break;
            }
        }
    }
    if found == NONE {
        return Located(NONE, 0.0, p0);
    }
    let fade = params.limits1.y;
    if !(fade > 0.0) {
        return Located(found, 1.0, p);
    }
    var dist = 3.0e38;
    for (var e = 0u; e < pl.ranges.w; e++) {
        let ed = edges[pl.ranges.z + e];
        dist = min(dist, point_segment_distance(p, ed.a.xyz, ed.b.xyz));
    }
    let s = clamp(dist / fade, 0.0, 1.0);
    return Located(found, s * s * (3.0 - 2.0 * s), p);
}

// Validate the plane sequence p_seq[0..n] (images p_img[0..=n]); append the path to
// the candidate list when it is real. Mirrors validate_image_path up to the gain.
fn validate_path(q: u32, n: u32, src: vec3<f32>, lis: vec3<f32>) {
    let side_eps = params.limits1.z;
    var prev = lis;
    var edge_w = 1.0;
    var k = n;
    loop {
        if k == 0u {
            break;
        }
        let pi = p_seq[k - 1u];
        let tgt = p_img[k];
        let da = plane_sd(pi, prev);
        let db = plane_sd(pi, tgt);
        if !(da * db < 0.0) || abs(da) < side_eps || abs(db) < side_eps {
            return;
        }
        let t = da / (da - db);
        let b = prev + (tgt - prev) * t;
        let loc = locate_on_plane(pi, b);
        if loc.tri == NONE {
            return;
        }
        p_pts[k - 1u] = loc.p;
        p_tri[k - 1u] = loc.tri;
        edge_w *= loc.w;
        prev = loc.p;
        k -= 1u;
    }

    // Visibility of every segment L -> B_n -> .. -> B_1 -> S.
    var total = 0.0;
    var seg_a = lis;
    k = n;
    loop {
        if k == 0u {
            break;
        }
        let seg_b = p_pts[k - 1u];
        // Material-dependent reflected-path visibility is evaluated by the host
        // using the CPU backend's nine-ray transmission bundle. Do not reject
        // geometrically valid image paths here when a blocker is transmissive.
        total += distance(seg_a, seg_b);
        seg_a = seg_b;
        k -= 1u;
    }
    // See above: the host applies per-band blocker transmission after download.
    total += distance(seg_a, src);
    if !(total > 0.0 && total <= params.limits1.x) {
        return;
    }

    let slot = atomicAdd(&n_cand, 1u);
    if slot < params.counts1.x {
        let ci = q * params.counts1.x + slot;
        cands[ci].head = vec4<f32>(edge_w, total, 0.0, 0.0);
        cands[ci].info = vec4<u32>(n, 0u, 0u, 0u);
        for (var j = 0u; j < n; j++) {
            cands[ci].seq[j] = p_seq[j];
            cands[ci].tri[j] = p_tri[j];
            cands[ci].pts[j] = vec4<f32>(p_pts[j], 0.0);
        }
    }
}

// Depth-first image-tree search of the subtree whose first bounce is `first_plane`
// (iterative; same node order and pruning as expand_images).
fn search_subtree(q: u32, first_plane: u32, src: vec3<f32>, lis: vec3<f32>) {
    let n_planes = params.counts0.z;
    let max_order = min(params.counts0.w, MAX_ORDER);
    let node_cap = params.counts1.z;
    p_img[0] = src;
    var depth = 0u;
    p_next[0] = first_plane;
    var nodes = 0u;
    loop {
        var end = n_planes;
        if depth == 0u {
            end = first_plane + 1u;
        }
        let pi = p_next[depth];
        if pi >= end {
            if depth == 0u {
                break;
            }
            depth -= 1u;
            continue;
        }
        p_next[depth] = pi + 1u;
        if nodes >= node_cap {
            break;
        }
        if depth > 0u && p_seq[depth - 1u] == pi {
            continue; // a plane cannot reflect twice in a row
        }
        if depth == 0u && abs(plane_sd(pi, src)) < params.limits1.z {
            continue; // source lies on the surface
        }
        nodes += 1u;
        p_seq[depth] = pi;
        let img = plane_mirror(pi, p_img[depth]);
        p_img[depth + 1u] = img;
        let n = depth + 1u;
        if distance(lis, img) <= params.limits1.x {
            validate_path(q, n, src, lis);
        }
        if n < max_order {
            depth = n;
            p_next[depth] = 0u;
        }
    }
}

// ------------------------------------------------------------------------ main

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let q = wg.x;
    let query = queries[q];
    let src = query.source.xyz;
    let lis = query.listener.xyz;
    let valid = query.source.w > 0.5;

    let axis = normalize(src - lis);
    let u = basis_u(axis);
    let w = cross(axis, u);

    // Phase 1: one occlusion probe per thread.
    if valid && li < N_RAYS {
        trace_probe(q * N_RAYS + li, lis, occlusion_target(li, src, u, w));
    }
    storageBarrier();
    workgroupBarrier();

    // Phase 2: the reference ray (first blocked probe in pattern order).
    if li == 0u {
        g_ref_valid = 0u;
        if valid {
            for (var k = 0u; k < N_RAYS; k++) {
                if heads[q].rays[k].count.x > 0u {
                    g_ref_valid = 1u;
                    g_ref_tgt = occlusion_target(k, src, u, w);
                    g_ref_hit = heads[q].rays[k].first.xyz;
                    break;
                }
            }
        }
    }
    workgroupBarrier();

    // Phase 3: detour search, one lateral direction per thread.
    if li < N_DETOUR {
        g_det[li] = -1.0;
        if g_ref_valid == 1u {
            g_det[li] = detour_dir(li, lis, g_ref_tgt, g_ref_hit, u, w);
        }
    }
    workgroupBarrier();
    if li == 0u {
        var best = -1.0;
        for (var j = 0u; j < N_DETOUR; j++) {
            let d = g_det[j];
            if d >= 0.0 && (best < 0.0 || d < best) {
                best = d;
            }
        }
        heads[q].delta = vec4<f32>(best, 0.0, 0.0, 0.0);
    }

    // Phase 4: early reflections, one first-bounce plane per thread.
    if valid && li < params.counts0.z {
        search_subtree(q, li, src, lis);
    }
    workgroupBarrier();
    if li == 0u {
        heads[q].info = vec4<u32>(atomicLoad(&n_cand), 0u, 0u, 0u);
    }
}
