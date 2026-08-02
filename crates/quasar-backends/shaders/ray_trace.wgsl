// Real-time spatial audio ray tracing shader.
// Each workgroup evaluates one source-listener pair.
// Workgroup threads trace stochastic rays for acoustic parameter estimation.

struct Params {
    listener_pos: vec3<f32>,
    source_pos: vec3<f32>,
    _pad0: f32,
    n_rays: u32,
    max_bounces: u32,
    num_meshes: u32,
    num_indices: u32,
    speed_of_sound: f32,
    max_duration: f32,
    _pad1: f32,
    air_abs: array<f32, 8>,
    sample_rate: f32,
    _pad2: vec3<f32>,
    seed: u32,
}

struct Vertex {
    pos: vec3<f32>,
    _p: f32,
    normal: vec3<f32>,
    _p2: f32,
}

struct Mesh {
    idx_off: u32,
    idx_cnt: u32,
    vert_off: u32,
    mat_idx: u32,
    xform: mat4x4<f32>,
}

struct Material {
    absorption: array<f32, 8>,
    scattering: array<f32, 8>,
    transmission: array<f32, 8>,
}

struct RayHitResult {
    distance: f32,
    hit: u32,
    _pad0: f32,
    _pad1: f32,
    material_idx: u32,
    hit_point: vec3<f32>,
    hit_normal: vec3<f32>,
}

struct SpatialOutput {
    direct_distance: f32,
    direct_occluded: u32,
    _pad0: f32,
    _pad1: f32,
    direct_attenuation: array<f32, 8>,
    late_t60: array<f32, 8>,
    late_energy: f32,
    _pad2: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> verts: array<Vertex>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(4) var<storage, read> mats: array<Material>;
@group(0) @binding(5) var<storage, read_write> output: array<SpatialOutput>;
@group(0) @binding(6) var<storage, read> ray_hits: array<RayHitResult>;

var<private> rng_state: u32;

fn pcg() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rf() -> f32 {
    return f32(pcg()) / 4294967296.0;
}

fn uniform_sphere() -> vec3<f32> {
    let theta = 6.2831853 * rf();
    let phi = acos(2.0 * rf() - 1.0);
    return vec3<f32>(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

struct Hit {
    t: f32,
    normal: vec3<f32>,
    mat_idx: u32,
    hit: bool,
}

fn scene_intersect(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var best_t: f32 = 1e10;
    var best_normal: vec3<f32> = vec3<f32>(0.0);
    var best_mat: u32 = 0u;
    var did_hit: bool = false;

    for (var mi: u32 = 0u; mi < params.num_meshes; mi = mi + 1u) {
        let mesh = meshes[mi];
        let idx_end = mesh.idx_off + mesh.idx_cnt;
        var vi: u32 = mesh.idx_off;
        while (vi + 2u < idx_end) {
            let i0 = indices[vi];
            let i1 = indices[vi + 1u];
            let i2 = indices[vi + 2u];
            let v0 = verts[mesh.vert_off + i0];
            let v1 = verts[mesh.vert_off + i1];
            let v2 = verts[mesh.vert_off + i2];

            // Transform to world space via mesh.xform
            let a = (mesh.xform * vec4<f32>(v0.pos, 1.0)).xyz;
            let b = (mesh.xform * vec4<f32>(v1.pos, 1.0)).xyz;
            let c = (mesh.xform * vec4<f32>(v2.pos, 1.0)).xyz;

            // Möller-Trumbore
            let e1 = b - a;
            let e2 = c - a;
            let h = cross(rd, e2);
            let det = dot(e1, h);
            if (abs(det) < 1e-12) {
                vi = vi + 3u;
                continue;
            }
            let inv_det = 1.0 / det;
            let s = ro - a;
            let u = dot(s, h) * inv_det;
            if (u < 0.0 || u > 1.0) {
                vi = vi + 3u;
                continue;
            }
            let q = cross(s, e1);
            let v = dot(rd, q) * inv_det;
            if (v < 0.0 || u + v > 1.0) {
                vi = vi + 3u;
                continue;
            }
            let t = dot(e2, q) * inv_det;
            if (t > 0.001 && t < best_t) {
                best_t = t;
                best_normal = normalize(cross(e1, e2));
                best_mat = mesh.mat_idx;
                did_hit = true;
            }
            vi = vi + 3u;
        }
    }

    return Hit(best_t, best_normal, best_mat, did_hit);
}

fn hemispherical_diffuse(n: vec3<f32>) -> vec3<f32> {
    let u = rf();
    let v = rf();
    let theta = 2.0 * 3.14159265 * u;
    let phi = acos(sqrt(v));
    let local_dir = vec3<f32>(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));

    // Build tangent frame
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let tangent = normalize(cross(up, n));
    if (length(tangent) < 0.001) {
        let right = vec3<f32>(1.0, 0.0, 0.0);
        let tangent2 = normalize(cross(right, n));
        let bitangent2 = cross(n, tangent2);
        return local_dir.x * tangent2 + local_dir.y * n + local_dir.z * bitangent2;
    }
    let bitangent = cross(n, tangent);
    return local_dir.x * tangent + local_dir.y * n + local_dir.z * bitangent;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    let num_workgroups = gid.y + 1u;

    // Seed RNG per thread
    rng_state = params.seed + thread_idx * 6364136223846793005u;

    let listener = params.listener_pos;
    let source = params.source_pos;

    // Direct path check
    let to_source = source - listener;
    let dist = length(to_source);
    let dir = normalize(to_source);

    var direct_occluded: u32 = 0u;
    let hit = scene_intersect(listener + dir * 0.01, dir);

    var direct_dist = dist;
    if (hit.hit && hit.t < dist) {
        direct_occluded = 1u;
    }

    // Accumulate late reverb energy via stochastic rays
    var total_energy: f32 = 0.0;
    var hit_count: u32 = 0u;
    var total_absorption: array<f32, 8>;
    for (var b = 0u; b < 8u; b = b + 1u) {
        total_absorption[b] = 0.0;
    }

    for (var i: u32 = 0u; i < params.n_rays; i = i + 1u) {
        let rdir = uniform_sphere();
        let hit2 = scene_intersect(listener + rdir * 0.01, rdir);
        if (hit2.hit) {
            hit_count = hit_count + 1u;
            let refl_coeff = 1.0 - mats[hit2.mat_idx].absorption[0];
            total_energy = total_energy + refl_coeff / (1.0 + hit2.t);

            for (var b = 0u; b < 8u; b = b + 1u) {
                total_absorption[b] = total_absorption[b] + mats[hit2.mat_idx].absorption[b];
            }
        }
    }

    // Write result for this thread
    var out: SpatialOutput;
    out.direct_distance = direct_dist;
    out.direct_occluded = direct_occluded;
    out._pad0 = 0.0;
    out._pad1 = 0.0;

    let air_atten = 1.0 / (1.0 + dist * 0.01);
    for (var b = 0u; b < 8u; b = b + 1u) {
        out.direct_attenuation[b] = air_atten;
    }

    var avg_abs: f32 = 0.0;
    if (hit_count > 0u) {
        for (var b = 0u; b < 8u; b = b + 1u) {
            let avg = total_absorption[b] / f32(hit_count);
            avg_abs = avg_abs + avg;
            let t60 = 0.161 * 1000.0 / (100.0 * max(avg, 0.01));
            out.late_t60[b] = clamp(t60, 0.1, 10.0);
        }
        avg_abs = avg_abs / 8.0;
    } else {
        for (var b = 0u; b < 8u; b = b + 1u) {
            out.late_t60[b] = 0.5;
        }
    }

    out.late_energy = total_energy / (f32(params.n_rays) + 1.0);
    out._pad2 = vec3<f32>(0.0);

    output[thread_idx] = out;
}
