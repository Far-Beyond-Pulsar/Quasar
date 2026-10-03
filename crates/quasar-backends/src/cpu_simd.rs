use quasar_core::backend::{
    DirectPathResult, EarlyReflection, IAcousticComputeBackend, LateReverbEstimate,
    MaterialProvider, SpatialQuery, SpatialQueryResult,
};
use quasar_core::bands::Band8;
use quasar_core::distance::DistanceModel;
use quasar_core::error::SpatialAudioError;
use quasar_core::rays::{Ray, RayHit, RayInteractionContext};
use quasar_core::scene::AcousticScene;

/// CPU-based spatial compute backend using BVH-accelerated ray tracing.
///
/// Features:
/// - BVH acceleration structure (SAH builder)
/// - Multithreaded ray execution via rayon
/// - Möller-Trumbore triangle intersection
/// - Image-source specular early reflections (up to configurable order, see
///   [`CpuSimdComputeBackend::trace_early_reflections`])
/// - Sabine/Eyring statistical late reverberation estimation
/// - Dynamic scene update support (rebuilds BVH)
pub struct CpuSimdComputeBackend {
    scene: AcousticScene,
    bvh: Option<FlatBvh>,
    /// Flat world-space triangle list used by reflection-plane construction.
    triangles: Vec<Triangle>,
    /// Deduplicated candidate mirror planes (largest first, capped), see [`ReflectPlane`].
    planes: Vec<ReflectPlane>,
    /// Volume / area statistics for the late-reverb estimate.
    room: RoomStats,
    /// Number of "room is not closed" warnings emitted (at most 1 per backend).
    room_warnings: u32,
    config: CpuSimdConfig,
    distance_model: DistanceModel,
    debug_capture: std::sync::Arc<crate::debug_capture::AcousticDebugCapture>,
}

/// Configuration for the CPU SIMD backend.
#[derive(Clone, Debug)]
pub struct CpuSimdConfig {
    /// Maximum specular bounce order for early reflections (default: 3; at most 8).
    pub max_reflection_order: u32,
    /// Maximum number of early-reflection paths returned per query, strongest
    /// first (default: 16; at most 64, the engine's crossfader capacity).
    pub max_reflections: usize,
    /// Maximum number of distinct mirror planes considered by the image-source
    /// tracer, largest area first (default: 32). Query cost grows as
    /// `P (P-1)^(order-1)`, see `trace_early_reflections`.
    pub max_reflection_planes: usize,
    /// Width (m) of the fade at the border of a reflecting surface: a specular
    /// bounce closer than this to the outer edge of its surface is attenuated
    /// smoothly to zero at the edge, so paths appear / disappear continuously
    /// as the listener moves (default: 0.1; 0 = hard edge).
    pub reflection_edge_fade: f32,
    /// Stochastic rays for late reverb estimation (default: 64).
    pub diffuse_rays_per_query: u32,
    /// Max distance for reflection tracing in world units (default: 50.0).
    pub max_reflection_distance: f32,
    /// Speed of sound in meters per second (default: 343.0).
    pub speed_of_sound: f32,
    /// Air temperature in Celsius (default: 20.0).
    pub temperature_celsius: f32,
    /// Relative humidity percentage (default: 50.0).
    pub humidity_percent: f32,
    /// Audio device sample rate in Hz (default: 48000). Every `delay_samples` the
    /// backend returns is `path_length * sample_rate / speed_of_sound`. The engine
    /// overwrites it with its own rate through
    /// [`IAcousticComputeBackend::set_sample_rate`] when the backend is installed.
    pub sample_rate: f32,
}

impl Default for CpuSimdConfig {
    fn default() -> Self {
        Self {
            max_reflection_order: 3,
            max_reflections: 16,
            max_reflection_planes: 32,
            reflection_edge_fade: 0.1,
            diffuse_rays_per_query: 64,
            max_reflection_distance: 50.0,
            speed_of_sound: 343.0,
            temperature_celsius: 20.0,
            humidity_percent: 50.0,
            sample_rate: 48_000.0,
        }
    }
}

// ── Direct-path occlusion parameters ─────────────────────────────────

/// Rays per occlusion query: the source centre plus 12 points of a golden-angle
/// sunflower on a disc around the source (deterministic, no RNG).
pub(crate) const OCCLUSION_RAYS: usize = 13;
/// Parallel rays used to estimate reflected-path visibility through a 12 cm tube.
const REFLECTION_VISIBILITY_RAYS: usize = 9;
/// Radius of the reflected-path visibility tube, in metres.
const REFLECTION_VISIBILITY_RADIUS: f32 = 0.06;
/// Radius (m) of the disc of target points around the source (its apparent size).
pub(crate) const OCCLUSION_SOURCE_RADIUS: f32 = 0.35;
/// Golden angle (rad) between consecutive sunflower points.
pub(crate) const GOLDEN_ANGLE: f32 = 2.399_963_2;
/// Surfaces crossed by one ray beyond this count as opaque.
pub(crate) const OCCLUSION_MAX_CROSSINGS: usize = 8;
/// Epsilon (m) kept clear at both ends of every occlusion segment and after each crossing.
pub(crate) const OCCLUSION_EPS: f32 = 1e-3;
/// Lateral directions probed when looking for the shortest detour around an occluder.
pub(crate) const OCCLUSION_DETOUR_DIRS: usize = 8;
/// First / last lateral offset (m) of the exponential detour search.
pub(crate) const OCCLUSION_DETOUR_MIN_OFFSET: f32 = 0.05;
pub(crate) const OCCLUSION_DETOUR_MAX_OFFSET: f32 = 26.0;
/// Bisection refinements of the detour offset (resolution about offset / 32).
pub(crate) const OCCLUSION_BISECT_STEPS: usize = 5;
/// How far (m) a detour leg is extended past the detour point when checking it.
pub(crate) const OCCLUSION_DETOUR_MARGIN: f32 = 0.02;
/// Cap of the single-edge diffraction attenuation (dB).
const OCCLUSION_MAX_DIFFRACTION_DB: f32 = 30.0;
/// Lowest per-band occlusion amplitude (-80 dB): a fully blocked path is attenuated, not NaN/zero.
const OCCLUSION_FLOOR: f32 = 1e-4;

/// Per-band occlusion of the direct path.
struct OcclusionResult {
    /// Linear amplitude per band (1 = clear).
    bands: Band8,
    /// Any occluder between listener and source (some probe ray is blocked).
    occluded: bool,
}
/// Barycentric slack of the ray-triangle test (watertightness across shared edges).
pub(crate) const BARY_EPS: f32 = 1e-5;
/// Relative padding of BVH boxes (a few ULPs), so flat boxes and grazing rays are kept.
const AABB_PAD: f32 = 1e-6;

// ── AABB ─────────────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Aabb {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
}

impl Aabb {
    fn new_empty() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }

    fn from_points(points: &[[f32; 3]]) -> Self {
        let mut b = Self::new_empty();
        for p in points {
            for i in 0..3 {
                b.min[i] = b.min[i].min(p[i]);
                b.max[i] = b.max[i].max(p[i]);
            }
        }
        b
    }

    fn union(&self, other: &Aabb) -> Aabb {
        let mut b = *self;
        for i in 0..3 {
            b.min[i] = b.min[i].min(other.min[i]);
            b.max[i] = b.max[i].max(other.max[i]);
        }
        b
    }

    /// Slab test against the ray interval `[ray.min_distance, t_max]`.
    ///
    /// Returns the entry distance (clamped to `min_distance`) or `None` on a miss.
    /// A zero direction component is handled explicitly (no `0 * inf = NaN`): the
    /// ray then either lies inside the slab for its whole length or misses. The box
    /// is padded by a few ULPs of its coordinates so flat (zero-thickness) boxes of
    /// axis-aligned quads, and rays grazing a face, are not lost to rounding.
    fn intersect_t(&self, ray: &Ray, t_max: f32) -> Option<f32> {
        let mut tmin = ray.min_distance;
        let mut tmax = t_max;
        for i in 0..3 {
            let pad = AABB_PAD * (1.0 + self.min[i].abs().max(self.max[i].abs()));
            let lo = self.min[i] - pad;
            let hi = self.max[i] + pad;
            let o = ray.origin[i];
            let d = ray.direction[i];
            if d.abs() < 1e-20 {
                if o < lo || o > hi {
                    return None;
                }
                continue;
            }
            let inv_d = 1.0 / d;
            let t1 = (lo - o) * inv_d;
            let t2 = (hi - o) * inv_d;
            let ta = t1.min(t2);
            let tb = t1.max(t2);
            tmin = tmin.max(ta);
            tmax = tmax.min(tb);
            if tmin > tmax {
                return None;
            }
        }
        Some(tmin)
    }

    fn surface_area(&self) -> f32 {
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        2.0 * (dx * dy + dx * dz + dy * dz)
    }

}

// ── Triangle ──────────────────────────────────────────────────────────

/// A single triangle for intersection testing.
#[derive(Clone, Debug)]
pub(crate) struct Triangle {
    pub(crate) a: [f32; 3],
    pub(crate) b: [f32; 3],
    pub(crate) c: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) material_handle: u32,
    #[allow(dead_code)]
    mesh_id: u64,
}

impl Triangle {
    fn new(
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        material_handle: u32,
        mesh_id: u64,
    ) -> Self {
        let e1 = sub3(b, a);
        let e2 = sub3(c, a);
        let n = cross3(e1, e2);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let normal = if len > 1e-12 {
            [n[0] / len, n[1] / len, n[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };
        Self {
            a,
            b,
            c,
            normal,
            material_handle,
            mesh_id,
        }
    }

    /// True for zero-area triangles (they have no meaningful normal and cannot be hit).
    fn is_degenerate(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> bool {
        let n = cross3(sub3(b, a), sub3(c, a));
        let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        !(area2 > 1e-12)
    }

    /// Möller-Trumbore ray-triangle intersection over `[ray.min_distance, t_max]`.
    ///
    /// Edge-tolerant: the barycentric coordinates may fall [`BARY_EPS`] outside
    /// the triangle, so a ray through a shared edge or vertex of a tessellated
    /// surface hits at least one of its triangles (no seams). Near-parallel rays
    /// are rejected relative to the triangle size (`|det| <= 1e-9 |e1| |e2|`),
    /// not by an absolute f32 threshold, so glancing hits on large triangles
    /// survive.
    pub(crate) fn intersect_max(&self, ray: &Ray, t_max: f32) -> Option<f32> {
        let edge1 = sub3(self.b, self.a);
        let edge2 = sub3(self.c, self.a);
        let h = cross3(ray.direction, edge2);
        let det = dot3(edge1, h);
        let scale = (dot3(edge1, edge1) * dot3(edge2, edge2)).sqrt();
        if !(det.abs() > 1e-9 * scale) {
            return None;
        }
        let inv_det = 1.0 / det;
        let s = sub3(ray.origin, self.a);
        let u = dot3(s, h) * inv_det;
        if u < -BARY_EPS || u > 1.0 + BARY_EPS {
            return None;
        }
        let q = cross3(s, edge1);
        let v = dot3(ray.direction, q) * inv_det;
        if v < -BARY_EPS || u + v > 1.0 + BARY_EPS {
            return None;
        }
        let t = dot3(edge2, q) * inv_det;
        if !(t >= ray.min_distance && t <= t_max) {
            return None;
        }
        Some(t)
    }

    /// True if `p` (assumed in the triangle's plane) lies inside it, with the
    /// same [`BARY_EPS`] barycentric slack as the ray test.
    fn contains_point(&self, p: [f32; 3]) -> bool {
        let v0 = sub3(self.c, self.a);
        let v1 = sub3(self.b, self.a);
        let v2 = sub3(p, self.a);
        let d00 = dot3(v0, v0);
        let d01 = dot3(v0, v1);
        let d11 = dot3(v1, v1);
        let d20 = dot3(v2, v0);
        let d21 = dot3(v2, v1);
        let denom = d00 * d11 - d01 * d01;
        if !(denom.abs() > 1e-20) {
            return false;
        }
        let u = (d11 * d20 - d01 * d21) / denom;
        let v = (d00 * d21 - d01 * d20) / denom;
        u >= -BARY_EPS && v >= -BARY_EPS && u + v <= 1.0 + BARY_EPS
    }

    fn centroid(&self) -> [f32; 3] {
        [
            (self.a[0] + self.b[0] + self.c[0]) / 3.0,
            (self.a[1] + self.b[1] + self.c[1]) / 3.0,
            (self.a[2] + self.b[2] + self.c[2]) / 3.0,
        ]
    }

    pub(crate) fn aabb(&self) -> Aabb {
        Aabb::from_points(&[self.a, self.b, self.c])
    }
}

// ── BVH Node ──────────────────────────────────────────────────────────

/// BVH node (either internal or leaf).
enum BvhNode {
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
        #[allow(dead_code)]
        split_axis: u8,
    },
    Leaf {
        aabb: Aabb,
        triangles: Vec<Triangle>,
    },
}

#[allow(dead_code)] // The pointer tree is used only as a temporary SAH build representation.
impl BvhNode {
    /// Build a BVH from triangles using the Surface Area Heuristic.
    fn build(triangles: &mut [Triangle]) -> Self {
        Self::build_sah(triangles, 0)
    }

    fn build_sah(triangles: &mut [Triangle], depth: usize) -> Self {
        let leaf_threshold = 4;
        let max_depth = 32;

        if triangles.len() <= leaf_threshold || depth >= max_depth {
            let aabb = triangles
                .iter()
                .fold(Aabb::new_empty(), |acc, t| acc.union(&t.aabb()));
            return BvhNode::Leaf {
                aabb,
                triangles: triangles.to_vec(),
            };
        }

        let centroid_aabb = triangles
            .iter()
            .fold(Aabb::new_empty(), |acc, t| acc.union(&Aabb::from_points(&[t.centroid()])));

        let mut best_cost = f32::MAX;
        let mut best_split: Option<(u8, usize)> = None;
        let n = triangles.len();

        for axis in 0..3u8 {
            let a = axis as usize;
            let span = centroid_aabb.max[a] - centroid_aabb.min[a];
            if span < 1e-8 {
                continue;
            }

            triangles.sort_by(|t1, t2| {
                t1.centroid()[a].partial_cmp(&t2.centroid()[a]).unwrap()
            });

            let mut prefix_aabb = vec![Aabb::new_empty(); n];
            let mut suffix_aabb = vec![Aabb::new_empty(); n];

            let mut acc = Aabb::new_empty();
            for i in 0..n {
                acc = acc.union(&triangles[i].aabb());
                prefix_aabb[i] = acc;
            }

            let mut acc = Aabb::new_empty();
            for i in (0..n).rev() {
                acc = acc.union(&triangles[i].aabb());
                suffix_aabb[i] = acc;
            }

            for i in 1..n {
                let left_area = prefix_aabb[i - 1].surface_area();
                let right_area = suffix_aabb[i].surface_area();
                let cost = 1.0 + (left_area * i as f32 + right_area * (n - i) as f32) / (n as f32);
                if cost < best_cost {
                    best_cost = cost;
                    best_split = Some((axis, i));
                }
            }
        }

        if let Some((axis, split_idx)) = best_split {
            let a = axis as usize;
            triangles.sort_by(|t1, t2| {
                t1.centroid()[a].partial_cmp(&t2.centroid()[a]).unwrap()
            });

            let (left_tri, right_tri) = triangles.split_at_mut(split_idx);
            let left = Box::new(BvhNode::build_sah(left_tri, depth + 1));
            let right = Box::new(BvhNode::build_sah(right_tri, depth + 1));

            let aabb = left.aabb().union(right.aabb());
            BvhNode::Internal {
                aabb,
                left,
                right,
                split_axis: axis,
            }
        } else {
            let aabb = triangles
                .iter()
                .fold(Aabb::new_empty(), |acc, t| acc.union(&t.aabb()));
            BvhNode::Leaf {
                aabb,
                triangles: triangles.to_vec(),
            }
        }
    }

    fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Internal { aabb, .. } => aabb,
            BvhNode::Leaf { aabb, .. } => aabb,
        }
    }

    /// Traverse the BVH and find the closest intersection.
    fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        self.intersect_closest(ray, ray.max_distance)
    }

    /// Front-to-back traversal: the child whose box is entered first is visited
    /// first, and `best` (the closest hit distance found so far, initially the ray
    /// limit) prunes every box and triangle that cannot beat it.
    fn intersect_closest(&self, ray: &Ray, best: f32) -> Option<RayHit> {
        match self {
            BvhNode::Internal { left, right, .. } => {
                let tl = left.aabb().intersect_t(ray, best);
                let tr = right.aabb().intersect_t(ray, best);
                match (tl, tr) {
                    (None, None) => None,
                    (Some(_), None) => left.intersect_closest(ray, best),
                    (None, Some(_)) => right.intersect_closest(ray, best),
                    (Some(a), Some(b)) => {
                        let (first, second, t_second) =
                            if a <= b { (left, right, b) } else { (right, left, a) };
                        let h1 = first.intersect_closest(ray, best);
                        let best2 = h1.as_ref().map_or(best, |h| h.distance.min(best));
                        // The far child can only matter if its box starts before the best hit.
                        if t_second > best2 {
                            return h1;
                        }
                        match (h1, second.intersect_closest(ray, best2)) {
                            (Some(x), Some(y)) => Some(if y.distance < x.distance { y } else { x }),
                            (Some(x), None) => Some(x),
                            (None, y) => y,
                        }
                    }
                }
            }
            BvhNode::Leaf { aabb, triangles } => {
                aabb.intersect_t(ray, best)?;
                let mut best_t = best;
                let mut closest: Option<RayHit> = None;
                for tri in triangles {
                    if let Some(t) = tri.intersect_max(ray, best_t) {
                        if t < best_t || closest.is_none() {
                            best_t = t;
                            closest = Some(RayHit {
                                distance: t,
                                point: ray.point_at(t),
                                normal: tri.normal,
                                material_handle: tri.material_handle,
                                hit: true,
                            });
                        }
                    }
                }
                closest
            }
        }
    }
}

/// Cache-friendly BVH storage: nodes and leaf triangles each occupy contiguous arrays.
/// Child and triangle ranges are 32-bit offsets; no heap pointers are followed while tracing.
struct FlatNode { aabb: Aabb, left: u32, right: u32, start: u32, len: u32 }
struct FlatBvh { nodes: Vec<FlatNode>, triangles: Vec<Triangle> }

impl FlatBvh {
    fn build(root: &BvhNode) -> Self {
        fn append(node: &BvhNode, out: &mut FlatBvh) -> u32 {
            let index = out.nodes.len() as u32;
            out.nodes.push(FlatNode { aabb: node.aabb().clone(), left: 0, right: 0, start: 0, len: 0 });
            match node {
                BvhNode::Internal { left, right, .. } => {
                    let l = append(left, out); let r = append(right, out);
                    out.nodes[index as usize].left = l;
                    out.nodes[index as usize].right = r;
                }
                BvhNode::Leaf { triangles, .. } => {
                    let start = out.triangles.len() as u32;
                    out.triangles.extend(triangles.iter().cloned());
                    out.nodes[index as usize].start = start;
                    out.nodes[index as usize].len = triangles.len() as u32;
                }
            }
            index
        }
        let mut out = Self { nodes: Vec::new(), triangles: Vec::new() };
        append(root, &mut out);
        out
    }

    fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        let mut stack = vec![0u32];
        let mut best = ray.max_distance;
        let mut closest = None;
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i as usize];
            if node.aabb.intersect_t(ray, best).is_none() { continue; }
            if node.len > 0 {
                for tri in &self.triangles[node.start as usize..(node.start + node.len) as usize] {
                    if let Some(t) = tri.intersect_max(ray, best) {
                        if t < best || closest.is_none() {
                            best = t;
                            closest = Some(RayHit { distance: t, point: ray.point_at(t), normal: tri.normal, material_handle: tri.material_handle, hit: true });
                        }
                    }
                }
            } else {
                let l = node.left; let r = node.right;
                let tl = self.nodes[l as usize].aabb.intersect_t(ray, best);
                let tr = self.nodes[r as usize].aabb.intersect_t(ray, best);
                match (tl, tr) {
                    (Some(a), Some(b)) if a <= b => { stack.push(r); stack.push(l); }
                    (Some(_), Some(_)) => { stack.push(l); stack.push(r); }
                    (Some(_), None) => stack.push(l),
                    (None, Some(_)) => stack.push(r),
                    (None, None) => {}
                }
            }
        }
        closest
    }
}

/// Final stage of the direct-path occlusion, shared with the GPU backend: from the
/// number of `visible` / `blocked` probe rays (of [`OCCLUSION_RAYS`]), the summed
/// squared per-band transmission `t2_sum` of the blocked rays and the shortest
/// detour path difference `delta` (`None` = no detour found), compute the
/// Kurze-Anderson diffraction amplitude and the dB blend with the visibility
/// (see `compute_occlusion`). `blocked` must be non-zero.
pub(crate) fn combine_occlusion(
    visible: usize,
    blocked: usize,
    t2_sum: &[f32; 8],
    delta: Option<f32>,
    speed_of_sound: f32,
) -> Band8 {
    let mut diffraction = [0.0_f32; 8];
    if let Some(delta) = delta {
        for b in 0..8 {
            let f = quasar_core::bands::FREQ_BAND_CENTRES[b];
            let n = 2.0 * delta * f / speed_of_sound;
            let x = (2.0 * std::f32::consts::PI * n).sqrt();
            let ratio = if x < 1e-3 { 1.0 } else { x / x.tanh() };
            let a_db = (5.0 + 20.0 * ratio.log10()).min(OCCLUSION_MAX_DIFFRACTION_DB);
            diffraction[b] = 10.0_f32.powf(-a_db / 20.0);
        }
    }
    let v = visible as f32 / OCCLUSION_RAYS as f32;
    let mut bands = Band8::splat(1.0);
    for b in 0..8 {
        let t2 = t2_sum[b] / blocked as f32;
        let shadow = (t2 + diffraction[b] * diffraction[b]).sqrt().clamp(OCCLUSION_FLOOR, 1.0);
        bands.0[b] = shadow.powf(1.0 - v);
    }
    bands
}

/// Per-band two-edge extension of `combine_occlusion`. This intentionally uses
/// two cascaded Kurze-Anderson terms as a deterministic engineering approximation;
/// it is not a uniform theory of diffraction (UTD) solution.
fn combine_occlusion_two_edges(
    visible: usize,
    blocked: usize,
    t2_sum: &[f32; 8],
    deltas: Option<(f32, f32)>,
    speed_of_sound: f32,
) -> Band8 {
    let mut diffraction = [0.0_f32; 8];
    if let Some((delta_a, delta_b)) = deltas {
        for b in 0..8 {
            let attenuation = |delta: f32| {
                let n = 2.0 * delta.max(0.0) * quasar_core::bands::FREQ_BAND_CENTRES[b] / speed_of_sound;
                let x = (2.0 * std::f32::consts::PI * n).sqrt();
                let ratio = if x < 1e-3 { 1.0 } else { x / x.tanh() };
                10.0_f32.powf(-((5.0 + 20.0 * ratio.log10()).min(OCCLUSION_MAX_DIFFRACTION_DB)) / 20.0)
            };
            diffraction[b] = attenuation(delta_a) * attenuation(delta_b);
        }
    }
    let v = visible as f32 / OCCLUSION_RAYS as f32;
    let mut bands = Band8::splat(1.0);
    for b in 0..8 {
        let t2 = t2_sum[b] / blocked as f32;
        bands.0[b] = (t2 + diffraction[b] * diffraction[b])
            .sqrt()
            .clamp(OCCLUSION_FLOOR, 1.0)
            .powf(1.0 - v);
    }
    bands
}

// ── Mirror planes (image-source early reflections) ────────────────────

/// Maximum specular order the image-source tracer supports (fixed-size stack arrays).
pub(crate) const MAX_IMAGE_ORDER: usize = 8;
/// Largest number of image-tree nodes expanded per query; beyond it the search
/// stops (deterministically, in depth-first order) and the strongest paths found
/// so far are returned.
pub(crate) const MAX_IMAGE_NODES: usize = 250_000;
/// Triangles whose unit normals have a dot product above this (about 0.36 deg) and
/// whose plane offsets differ by less than [`PLANE_OFFSET_TOL`] share one mirror plane.
const PLANE_NORMAL_COS: f32 = 0.99998;
const PLANE_OFFSET_TOL: f32 = 2e-3;
/// A point closer than this (m) to a mirror plane counts as lying on it (no reflection).
pub(crate) const PLANE_SIDE_EPS: f32 = 1e-4;
/// Padding (m) of a plane's bounding box when locating a bounce point.
pub(crate) const PLANE_BOX_PAD: f32 = 2e-3;
/// Hard cap on the number of image paths returned by one query.
pub(crate) const MAX_REFLECTION_PATHS: usize = 64;

/// A candidate mirror plane: every triangle (of any mesh) lying in one plane.
///
/// The normal is canonical (its largest component is positive), so triangles of
/// opposite winding fall in the same plane.
pub(crate) struct ReflectPlane {
    pub(crate) normal: [f32; 3],
    /// `normal . p` for any point `p` of the plane.
    pub(crate) offset: f32,
    pub(crate) area: f32,
    pub(crate) aabb: Aabb,
    /// Indices into the backend's flat triangle list.
    pub(crate) tris: Vec<usize>,
    /// Edges used by exactly one triangle of the plane: its outer border (and
    /// the border of any hole), the places the reflecting surface ends.
    pub(crate) boundary: Vec<([f32; 3], [f32; 3])>,
}

impl ReflectPlane {
    #[inline]
    fn signed_distance(&self, p: [f32; 3]) -> f32 {
        dot3(self.normal, p) - self.offset
    }

    #[inline]
    fn mirror(&self, p: [f32; 3]) -> [f32; 3] {
        let d = 2.0 * self.signed_distance(p);
        [p[0] - d * self.normal[0], p[1] - d * self.normal[1], p[2] - d * self.normal[2]]
    }

    fn contains_in_box(&self, p: [f32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.aabb.min[i] - PLANE_BOX_PAD && p[i] <= self.aabb.max[i] + PLANE_BOX_PAD)
    }
}

/// Mutable state of one image-source search (one query).
struct ImageSearch {
    source: [f32; 3],
    listener: [f32; 3],
    order: usize,
    /// Plane index of each bounce, source side first.
    seq: [usize; MAX_IMAGE_ORDER],
    /// `images[k]` = source mirrored across `seq[0..k]`.
    images: [[f32; 3]; MAX_IMAGE_ORDER + 1],
    nodes: usize,
    found: Vec<PathCandidate>,
    debug_paths: Vec<crate::debug_capture::DebugReflectionPath>,
}

/// A validated path with the energy used to rank it.
pub(crate) struct PathCandidate {
    pub(crate) energy: f32,
    pub(crate) refl: EarlyReflection,
}

/// Distance from `p` to the segment `a b`.
fn point_segment_distance(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = sub3(b, a);
    let len2 = dot3(ab, ab);
    let t = if len2 > 1e-20 { (dot3(sub3(p, a), ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    distance3(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t])
}

/// Group `triangles` into deduplicated mirror planes, keep the `max_planes`
/// largest (by total area; ties keep the lower index) and compute their borders.
/// Deterministic; runs when the scene is (re)built, not per query.
pub(crate) fn build_planes(triangles: &[Triangle], max_planes: usize) -> Vec<ReflectPlane> {
    use std::collections::HashMap;

    let mut planes: Vec<ReflectPlane> = Vec::new();
    // Grid of canonical normals (cell = 1/50) -> plane indices, so a triangle only
    // compares against planes with a similar orientation (3^3 neighbouring cells).
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    let cell = |n: [f32; 3]| ((n[0] * 50.0).round() as i32, (n[1] * 50.0).round() as i32, (n[2] * 50.0).round() as i32);

    for (ti, t) in triangles.iter().enumerate() {
        let mut n = t.normal;
        let mut d = dot3(n, t.a);
        let mut axis = 0;
        for i in 1..3 {
            if n[i].abs() > n[axis].abs() {
                axis = i;
            }
        }
        if n[axis] < 0.0 {
            n = [-n[0], -n[1], -n[2]];
            d = -d;
        }
        let (cx, cy, cz) = cell(n);
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &pi in list {
                            let p = &planes[pi];
                            if dot3(p.normal, n) > PLANE_NORMAL_COS && (p.offset - d).abs() < PLANE_OFFSET_TOL {
                                found = Some(pi);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        let area = 0.5 * dot3(cross3(sub3(t.b, t.a), sub3(t.c, t.a)), cross3(sub3(t.b, t.a), sub3(t.c, t.a))).sqrt();
        match found {
            Some(pi) => {
                let p = &mut planes[pi];
                p.area += area;
                p.aabb = p.aabb.union(&t.aabb());
                p.tris.push(ti);
            }
            None => {
                grid.entry((cx, cy, cz)).or_default().push(planes.len());
                planes.push(ReflectPlane {
                    normal: n,
                    offset: d,
                    area,
                    aabb: t.aabb(),
                    tris: vec![ti],
                    boundary: Vec::new(),
                });
            }
        }
    }

    // Largest first; the stable sort keeps scene order among equal areas.
    planes.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap_or(std::cmp::Ordering::Equal));
    planes.truncate(max_planes);

    // Borders: edges referenced by exactly one triangle of the plane.
    let key = |p: [f32; 3]| [(p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64];
    for plane in planes.iter_mut() {
        let mut edges: HashMap<([i64; 3], [i64; 3]), (u32, [f32; 3], [f32; 3])> = HashMap::new();
        for &ti in &plane.tris {
            let t = &triangles[ti];
            for (p, q) in [(t.a, t.b), (t.b, t.c), (t.c, t.a)] {
                let (kp, kq) = (key(p), key(q));
                let k = if kp <= kq { (kp, kq) } else { (kq, kp) };
                edges.entry(k).and_modify(|e| e.0 += 1).or_insert((1, p, q));
            }
        }
        plane.boundary = edges.values().filter(|e| e.0 == 1).map(|e| (e.1, e.2)).collect();
    }
    planes
}

/// Gain stage of one validated image path, shared with the GPU backend (which finds
/// and validates the geometry on the device and calls this on the host with the
/// bounce data): the reflection coefficient per band (product over the bounces of
/// `sqrt(1 - alpha)` at each bounce's incidence angle), the shared distance law and
/// air absorption over the TOTAL path length, and the edge window `edge_w`.
///
/// `seq[k]` is the plane index of bounce `k` (source side first), `tri_of[k]` the
/// triangle it lies on, `pts[k]` the bounce point; `total` the full path length.
/// `None` when the path carries no energy.
pub(crate) fn path_candidate(
    cfg: &CpuSimdConfig,
    distance_model: &DistanceModel,
    planes: &[ReflectPlane],
    triangles: &[Triangle],
    seq: &[usize],
    tri_of: &[usize],
    pts: &[[f32; 3]],
    source: [f32; 3],
    listener: [f32; 3],
    total: f32,
    edge_w: f32,
    materials: &dyn MaterialProvider,
) -> Option<PathCandidate> {
    let n = seq.len();
    // Reflection coefficient per band: product over the bounces.
    let mut refl = Band8::splat(1.0);
    let mut before = source;
    for k in 0..n {
        let plane = &planes[seq[k]];
        let tri = &triangles[tri_of[k]];
        let dir = normalize3(sub3(pts[k], before));
        let cos_i = dot3(dir, plane.normal).abs().clamp(0.0, 1.0);
        let ctx = RayInteractionContext {
            surface_normal: plane.normal,
            ray_direction: dir,
            incident_angle_rad: cos_i.acos(),
            temperature_celsius: cfg.temperature_celsius,
            humidity_percent: cfg.humidity_percent,
        };
        let absorption = materials.evaluate_material(tri.material_handle, &ctx);
        for b in 0..8 {
            let a = absorption.0[b];
            let a = if a.is_finite() { a.clamp(0.0, 1.0) } else { 1.0 };
            refl.0[b] *= (1.0 - a).sqrt();
        }
        before = pts[k];
    }

    // Full path attenuation (distance law x air absorption over the TOTAL length).
    let path = Band8::splat(distance_model.gain(total)).mul(&quasar_core::air::air_absorption_gain(
        total,
        cfg.temperature_celsius,
        cfg.humidity_percent,
    ));
    let gain = refl.mul(&path).scale(edge_w);
    let energy: f32 = gain.0.iter().map(|g| g * g).sum();
    if !(energy > 1e-14) {
        return None;
    }
    Some(PathCandidate {
        energy,
        refl: EarlyReflection {
            direction: normalize3(sub3(pts[n - 1], listener)),
            delay_samples: total * cfg.sample_rate / cfg.speed_of_sound,
            gain,
            order: n as u32,
        },
    })
}

/// Rank validated paths and keep the strongest `max_reflections`: strongest first
/// (energy, then order, then delay: a total order, deterministic), near-identical
/// paths merged (same order, length and direction; the strongest is kept).
pub(crate) fn rank_reflections(mut found: Vec<PathCandidate>, cfg: &CpuSimdConfig) -> Vec<EarlyReflection> {
    found.sort_by(|a, b| {
        b.energy
            .partial_cmp(&a.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.refl.order.cmp(&b.refl.order))
            .then(a.refl.delay_samples.partial_cmp(&b.refl.delay_samples).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut out: Vec<EarlyReflection> = Vec::new();
    for c in found {
        let dup = out.iter().any(|o| {
            o.order == c.refl.order
                && (o.delay_samples - c.refl.delay_samples).abs() * cfg.speed_of_sound / cfg.sample_rate < 1e-3
                && dot3(o.direction, c.refl.direction) > 1.0 - 1e-6
        });
        if !dup {
            out.push(c.refl);
        }
    }
    out.truncate(cfg.max_reflections.min(MAX_REFLECTION_PATHS));
    out
}

// ── Room statistics (late-reverb estimate) ────────────────────────────

/// Midpoint angles of the random-incidence (Paris) integration.
const PARIS_ANGLES: usize = 16;

/// Scene statistics for the statistical late-field estimate, computed once when the
/// scene is built.
pub(crate) struct RoomStats {
    /// Room volume (m^3).
    pub(crate) volume: f32,
    /// Total surface area (m^2) and its split per material handle (sorted by handle).
    pub(crate) area: f32,
    pub(crate) by_material: Vec<(u32, f32)>,
    /// Bounding box of the scene.
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
    /// The triangle soup is a closed, consistently wound surface (the volume is exact);
    /// `false` means the bounding-box volume fallback was used (or the scene is empty).
    pub(crate) closed: bool,
}

impl RoomStats {
    pub(crate) fn empty() -> Self {
        Self { volume: 0.0, area: 0.0, by_material: Vec::new(), min: [0.0; 3], max: [0.0; 3], closed: false }
    }

    /// Inside the bounding box padded by 0.5 m.
    fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] - 0.5 && p[i] <= self.max[i] + 0.5)
    }

    /// Volume from the signed tetrahedra `a . (b x c) / 6` when the triangle soup is a
    /// closed, consistently oriented surface (every directed edge occurs once and so does
    /// its reverse); the net `|sum|` is then the cavity minus any closed solids (columns
    /// wound outward, the room shell wound inward). Otherwise the bounding-box VOLUME
    /// (`closed` is false: the figure is only right for box-like rooms; the backend
    /// warns once, see `CpuSimdComputeBackend::room_is_closed`).
    pub(crate) fn build(triangles: &[Triangle]) -> Self {
        use std::collections::{BTreeMap, HashMap};
        if triangles.is_empty() {
            return Self::empty();
        }
        let mut aabb = Aabb::new_empty();
        let mut per: BTreeMap<u32, f32> = BTreeMap::new();
        let mut area = 0.0_f32;
        let mut signed = 0.0_f64;
        let mut edges: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
        let key = |p: [f32; 3]| [(p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64];
        for t in triangles {
            aabb = aabb.union(&t.aabb());
            let n = cross3(sub3(t.b, t.a), sub3(t.c, t.a));
            let a = 0.5 * dot3(n, n).sqrt();
            area += a;
            *per.entry(t.material_handle).or_insert(0.0) += a;
            let bxc = cross3(t.b, t.c);
            signed += (dot3(t.a, bxc) as f64) / 6.0;
            for (p, q) in [(t.a, t.b), (t.b, t.c), (t.c, t.a)] {
                *edges.entry((key(p), key(q))).or_insert(0) += 1;
            }
        }
        let closed = edges.iter().all(|(&(p, q), &c)| c == 1 && edges.get(&(q, p)) == Some(&1));
        let box_volume = (0..3).map(|i| (aabb.max[i] - aabb.min[i]).max(0.0)).product::<f32>();
        let volume_closed = closed && signed.abs() > 1e-6;
        let volume = if volume_closed {
            signed.abs() as f32
        } else {
            box_volume
        };
        Self { volume, area, by_material: per.into_iter().collect(), min: aabb.min, max: aabb.max, closed: volume_closed }
    }
}

/// Random-incidence absorption per band of one material:
/// `2 int alpha(theta) sin theta cos theta dtheta` over `0..pi/2`, midpoint rule.
fn random_incidence_absorption(materials: &dyn MaterialProvider, handle: u32, cfg: &CpuSimdConfig) -> Band8 {
    let mut acc = [0.0_f32; 8];
    let d = std::f32::consts::FRAC_PI_2 / PARIS_ANGLES as f32;
    for k in 0..PARIS_ANGLES {
        let theta = (k as f32 + 0.5) * d;
        let ctx = RayInteractionContext {
            surface_normal: [0.0, 1.0, 0.0],
            ray_direction: [theta.sin(), -theta.cos(), 0.0],
            incident_angle_rad: theta,
            temperature_celsius: cfg.temperature_celsius,
            humidity_percent: cfg.humidity_percent,
        };
        let a = materials.evaluate_material(handle, &ctx);
        let w = (2.0 * theta).sin() * d;
        for b in 0..8 {
            let v = a.0[b];
            acc[b] += if v.is_finite() { v.clamp(0.0, 1.0) } else { 1.0 } * w;
        }
    }
    Band8::new(acc)
}

// ── Vec3 helpers ──────────────────────────────────────────────────────

#[inline]
pub(crate) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Smooth deterministic tangent frame for the source probe disc. The previous
/// helper-axis switch rotated the finite probe pattern abruptly at |axis.y|=.9.
/// This frame is continuous except at its unavoidable south-pole singularity.
pub(crate) fn probe_basis(axis: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let u = if z < -0.999_999_9 {
        [0.0, -1.0, 0.0]
    } else {
        let a = 1.0 / (1.0 + z);
        let b = -x * y * a;
        [1.0 - x * x * a, b, -x]
    };
    (u, cross3(axis, u))
}

#[inline]
pub(crate) fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn transform_point4x4(p: [f32; 3], m: &[f32; 16]) -> [f32; 3] {
    let x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    let y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    let z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    [x, y, z]
}

// ── Main backend impl ─────────────────────────────────────────────────

impl CpuSimdComputeBackend {
    /// Create a new `CpuSimdComputeBackend` with the given scene and configuration.
    pub fn new(scene: AcousticScene, config: CpuSimdConfig) -> Self {
        let mut backend = Self {
            scene,
            bvh: None,
            triangles: Vec::new(),
            planes: Vec::new(),
            room: RoomStats::empty(),
            room_warnings: 0,
            config,
            distance_model: DistanceModel::default(),
            debug_capture: Default::default(),
        };
        backend.build_bvh();
        backend
    }

    /// Build the BVH acceleration structure (and the mirror-plane table of the
    /// early-reflection tracer) from the scene geometry.
    pub fn build_bvh(&mut self) {
        self.triangles = Self::triangles_from_scene(&self.scene);
        self.planes = build_planes(&self.triangles, self.config.max_reflection_planes);
        self.room = RoomStats::build(&self.triangles);
        if !self.triangles.is_empty() && !self.room.closed && self.room_warnings == 0 {
            self.room_warnings = 1;
            eprintln!(
                "quasar-backends: acoustic scene is not a closed, consistently wound surface; \
                 using the bounding-box volume ({:.1} m^3) for the late-reverb estimate (warned once)",
                self.room.volume
            );
        }
        if self.triangles.is_empty() {
            self.bvh = None;
            return;
        }
        let mut triangles = self.triangles.clone();
        let tree = BvhNode::build(&mut triangles);
        self.bvh = Some(FlatBvh::build(&tree));
    }

    /// Whether the current scene is a closed, consistently wound surface (exact room
    /// volume). `false` for open / inconsistent / empty scenes, which use the
    /// bounding-box volume for the late-reverb estimate; the warning for that is
    /// printed at most once per backend (see [`Self::room_warning_count`]).
    pub fn room_is_closed(&self) -> bool {
        self.room.closed
    }

    /// How many "room is not closed" warnings this backend has printed (0 or 1).
    pub fn room_warning_count(&self) -> u32 {
        self.room_warnings
    }

    /// Number of distinct mirror planes the early-reflection tracer considers
    /// (coplanar triangles merged, capped at `max_reflection_planes`).
    pub fn reflection_plane_count(&self) -> usize {
        self.planes.len()
    }

    pub(crate) fn triangles_from_scene(scene: &AcousticScene) -> Vec<Triangle> {
        let mut tris = Vec::new();
        for mesh in &scene.meshes {
            for chunk in mesh.indices.chunks_exact(3) {
                let i0 = chunk[0] as usize;
                let i1 = chunk[1] as usize;
                let i2 = chunk[2] as usize;
                if i0 >= mesh.positions.len()
                    || i1 >= mesh.positions.len()
                    || i2 >= mesh.positions.len()
                {
                    continue;
                }
                let a = transform_point4x4(mesh.positions[i0], &mesh.transform);
                let b = transform_point4x4(mesh.positions[i1], &mesh.transform);
                let c = transform_point4x4(mesh.positions[i2], &mesh.transform);
                if Triangle::is_degenerate(a, b, c) {
                    continue; // zero area: cannot be hit, has no normal
                }
                tris.push(Triangle::new(a, b, c, mesh.material_handle, mesh.id));
            }
        }
        tris
    }

    /// Trace a single ray through the BVH.
    pub fn debug_capture(&self) -> std::sync::Arc<crate::debug_capture::AcousticDebugCapture> {
        self.debug_capture.clone()
    }

    fn trace_single_ray(&self, ray: &Ray) -> Option<RayHit> {
        let hit = self.bvh.as_ref().and_then(|bvh| bvh.intersect(ray));
        if self.debug_capture.is_enabled() {
            self.debug_capture.record_ray(ray.clone(), hit.clone());
        }
        hit
    }

    /// Compute the direct path between source and listener.
    ///
    /// Gain = distance law x air absorption x per-band occlusion (see
    /// [`Self::compute_occlusion`]).
    fn compute_direct_path(
        &self,
        source: &[f32; 3],
        listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> DirectPathResult {
        let dist = distance3(*source, *listener);

        let occ = self.compute_occlusion(source, listener, materials);

        let atten = Band8::splat(self.distance_model.gain(dist));
        let air = quasar_core::air::air_absorption_gain(
            dist,
            self.config.temperature_celsius,
            self.config.humidity_percent,
        );
        let total_atten = atten.mul(&air).mul(&occ.bands);

        DirectPathResult {
            attenuation: total_atten,
            delay_samples: dist * self.config.sample_rate / self.config.speed_of_sound,
            distance: dist,
            occluded: occ.occluded,
            occlusion_factor: occ.bands.mean(),
            occlusion: occ.bands,
        }
    }

    /// Material interaction context for a ray hit (used for transmission lookups).
    fn hit_context(&self, dir: [f32; 3], hit: &RayHit) -> RayInteractionContext {
        RayInteractionContext {
            surface_normal: hit.normal,
            ray_direction: dir,
            incident_angle_rad: dot3(normalize3([-dir[0], -dir[1], -dir[2]]), hit.normal)
                .clamp(-1.0, 1.0)
                .acos(),
            temperature_celsius: self.config.temperature_celsius,
            humidity_percent: self.config.humidity_percent,
        }
    }

    /// Walk the segment `from -> to` and multiply the per-band amplitude
    /// transmission of EVERY surface it crosses (not just the first).
    ///
    /// Returns `(product, crossings, first_hit_point)`. More than
    /// [`OCCLUSION_MAX_CROSSINGS`] crossings count as opaque.
    fn segment_transmission(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        materials: &dyn MaterialProvider,
    ) -> (Band8, usize, Option<[f32; 3]>) {
        let total = distance3(from, to);
        if total < 2.0 * OCCLUSION_EPS {
            return (Band8::splat(1.0), 0, None);
        }
        let dir = normalize3(sub3(to, from));
        let mut origin = from;
        let mut remaining = total;
        let mut product = Band8::splat(1.0);
        let mut crossings = 0usize;
        let mut first = None;
        while remaining > 2.0 * OCCLUSION_EPS {
            let ray = Ray {
                origin,
                direction: dir,
                min_distance: OCCLUSION_EPS,
                max_distance: remaining - OCCLUSION_EPS,
            };
            let hit = match self.trace_single_ray(&ray) {
                Some(h) if h.hit => h,
                _ => break,
            };
            if first.is_none() {
                first = Some(hit.point);
            }
            crossings += 1;
            if crossings > OCCLUSION_MAX_CROSSINGS {
                return (Band8::zeros(), crossings, first);
            }
            let ctx = self.hit_context(dir, &hit);
            let t = materials.evaluate_transmission(hit.material_handle, &ctx);
            for b in 0..8 {
                product.0[b] *= t.0[b].clamp(0.0, 1.0);
            }
            origin = hit.point;
            remaining -= hit.distance;
        }
        (product, crossings, first)
    }

    /// Estimate reflected-path visibility by averaging transmission over a small,
    /// deterministic bundle of parallel rays. The centre ray plus eight rays on a
    /// 6 cm radius ring make blocker edges fade over the bundle diameter instead of
    /// removing the entire specular path at one exact intersection.
    pub(crate) fn segment_soft_transmission(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        materials: &dyn MaterialProvider,
    ) -> Band8 {
        let axis = normalize3(sub3(to, from));
        let (u, w) = probe_basis(axis);
        let mut sum = [0.0_f32; 8];
        for ray_index in 0..REFLECTION_VISIBILITY_RAYS {
            let offset = if ray_index == 0 {
                [0.0; 3]
            } else {
                let angle = (ray_index - 1) as f32 * (2.0 * std::f32::consts::PI / 8.0);
                let (s, c) = angle.sin_cos();
                [
                    REFLECTION_VISIBILITY_RADIUS * (u[0] * c + w[0] * s),
                    REFLECTION_VISIBILITY_RADIUS * (u[1] * c + w[1] * s),
                    REFLECTION_VISIBILITY_RADIUS * (u[2] * c + w[2] * s),
                ]
            };
            // Keep the exact endpoints on the specular path so the visibility
            // samples do not accidentally treat the reflecting surface at a bounce
            // as an intervening blocker. The small two-leg kink represents a ray
            // passing through the sampled point in the segment's cross-section.
            let mid = [
                0.5 * (from[0] + to[0]) + offset[0],
                0.5 * (from[1] + to[1]) + offset[1],
                0.5 * (from[2] + to[2]) + offset[2],
            ];
            let (gain_a, _, _) = self.segment_transmission(from, mid, materials);
            let (gain_b, _, _) = self.segment_transmission(mid, to, materials);
            let gain = gain_a.mul(&gain_b);
            for band in 0..8 {
                sum[band] += gain.0[band] / REFLECTION_VISIBILITY_RAYS as f32;
            }
        }
        Band8(sum)
    }

    /// True if nothing lies between `a` and `b`; the ray starts `before` metres before `a` and
    /// ends `after` metres past `b`.
    fn segment_clear_overshoot(&self, a: [f32; 3], b: [f32; 3], after: f32, before: f32) -> bool {
        let len = distance3(a, b);
        if len < 2.0 * OCCLUSION_EPS {
            return true;
        }
        let dir = normalize3(sub3(b, a));
        let origin = [a[0] - dir[0] * before, a[1] - dir[1] * before, a[2] - dir[2] * before];
        let ray = Ray {
            origin,
            direction: dir,
            min_distance: OCCLUSION_EPS,
            max_distance: before + len + after - OCCLUSION_EPS,
        };
        !matches!(self.trace_single_ray(&ray), Some(h) if h.hit)
    }

    /// Extra path length `|L-P| + |P-S| - |L-S|` of the shortest one-point detour
    /// around whatever blocks `listener -> target`, found by probing around the
    /// first hit `h`: for each of [`OCCLUSION_DETOUR_DIRS`] lateral directions the
    /// smallest offset (exponential search, then bisection, so the result varies
    /// continuously with the geometry) whose two legs are both clear. `None` if
    /// no such detour exists within [`OCCLUSION_DETOUR_MAX_OFFSET`].
    fn detour_extra_path(
        &self,
        listener: [f32; 3],
        target: [f32; 3],
        h: [f32; 3],
        u: [f32; 3],
        w: [f32; 3],
    ) -> Option<f32> {
        let direct = distance3(listener, target);
        let mut best: Option<f32> = None;
        for j in 0..OCCLUSION_DETOUR_DIRS {
            let phi = j as f32 * (2.0 * std::f32::consts::PI / OCCLUSION_DETOUR_DIRS as f32);
            let (s_phi, c_phi) = phi.sin_cos();
            let dir = [
                u[0] * c_phi + w[0] * s_phi,
                u[1] * c_phi + w[1] * s_phi,
                u[2] * c_phi + w[2] * s_phi,
            ];
            let at = |s: f32| [h[0] + dir[0] * s, h[1] + dir[1] * s, h[2] + dir[2] * s];
            let clear = |s: f32| {
                let p = at(s);
                // Each leg is checked a little beyond the detour point so a point that
                // merely lies ON an occluding surface (e.g. in the plane of a wall) does
                // not count as a way around it.
                self.segment_clear_overshoot(listener, p, OCCLUSION_DETOUR_MARGIN, 0.0)
                    && self.segment_clear_overshoot(p, target, 0.0, OCCLUSION_DETOUR_MARGIN)
            };

            // Exponential search for the first clear offset.
            let mut lo = 0.0_f32;
            let mut hi = OCCLUSION_DETOUR_MIN_OFFSET;
            let mut found = false;
            while hi <= OCCLUSION_DETOUR_MAX_OFFSET {
                if clear(hi) {
                    found = true;
                    break;
                }
                lo = hi;
                hi *= 2.0;
            }
            if !found {
                continue;
            }
            for _ in 0..OCCLUSION_BISECT_STEPS {
                let mid = 0.5 * (lo + hi);
                if clear(mid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let p = at(hi);
            let extra = (distance3(listener, p) + distance3(p, target) - direct).max(0.0);
            best = Some(best.map_or(extra, |b| b.min(extra)));
        }
        best
    }

    /// Find a deterministic two-corner route around sequential blockers. The
    /// first waypoint is probed around the direct ray's first hit; the first hit
    /// on that waypoint-to-target leg seeds a second probe. Returned excess
    /// lengths are local to each edge and feed the cascaded diffraction model.
    fn double_detour_excesses(
        &self,
        listener: [f32; 3],
        target: [f32; 3],
        first_hit: [f32; 3],
        u: [f32; 3],
        w: [f32; 3],
    ) -> Option<(f32, f32)> {
        let mut best: Option<(f32, f32, f32)> = None;
        for a in 0..OCCLUSION_DETOUR_DIRS {
            let phi_a = a as f32 * (2.0 * std::f32::consts::PI / OCCLUSION_DETOUR_DIRS as f32);
            let (sa, ca) = phi_a.sin_cos();
            let dir_a = [u[0] * ca + w[0] * sa, u[1] * ca + w[1] * sa, u[2] * ca + w[2] * sa];
            let mut offset_a = OCCLUSION_DETOUR_MIN_OFFSET;
            while offset_a <= OCCLUSION_DETOUR_MAX_OFFSET {
                let p1 = [first_hit[0] + dir_a[0] * offset_a, first_hit[1] + dir_a[1] * offset_a, first_hit[2] + dir_a[2] * offset_a];
                if self.segment_clear_overshoot(listener, p1, OCCLUSION_DETOUR_MARGIN, 0.0) {
                    let leg = distance3(p1, target);
                    let ray = Ray { origin: p1, direction: normalize3(sub3(target, p1)), min_distance: OCCLUSION_EPS, max_distance: leg - OCCLUSION_EPS };
                    let Some(hit) = self.trace_single_ray(&ray).filter(|hit| hit.hit) else {
                        offset_a *= 2.0;
                        continue;
                    };
                    let h2 = hit.point;
                    for b in 0..OCCLUSION_DETOUR_DIRS {
                        let phi_b = b as f32 * (2.0 * std::f32::consts::PI / OCCLUSION_DETOUR_DIRS as f32);
                        let (sb, cb) = phi_b.sin_cos();
                        let dir_b = [u[0] * cb + w[0] * sb, u[1] * cb + w[1] * sb, u[2] * cb + w[2] * sb];
                        let mut offset_b = OCCLUSION_DETOUR_MIN_OFFSET;
                        while offset_b <= OCCLUSION_DETOUR_MAX_OFFSET {
                            let p2 = [h2[0] + dir_b[0] * offset_b, h2[1] + dir_b[1] * offset_b, h2[2] + dir_b[2] * offset_b];
                            if self.segment_clear_overshoot(p1, p2, OCCLUSION_DETOUR_MARGIN, 0.0)
                                && self.segment_clear_overshoot(p2, target, 0.0, OCCLUSION_DETOUR_MARGIN)
                            {
                                let extra_a = (distance3(listener, p1) + distance3(p1, h2) - distance3(listener, h2)).max(0.0);
                                let extra_b = (distance3(h2, p2) + distance3(p2, target) - distance3(h2, target)).max(0.0);
                                let total = extra_a + extra_b;
                                if best.map_or(true, |x| total < x.0) { best = Some((total, extra_a, extra_b)); }
                                break;
                            }
                            offset_b *= 2.0;
                        }
                    }
                }
                offset_a *= 2.0;
            }
        }
        best.map(|(_, a, b)| (a, b))
    }

    /// Per-band direct-path occlusion (amplitude, 1 = clear, 0 = blocked).
    ///
    /// * **Transmission.** [`OCCLUSION_RAYS`] rays go from the listener to a
    ///   deterministic pattern of points on a disc of radius
    ///   [`OCCLUSION_SOURCE_RADIUS`] around the source (the centre plus a golden-angle
    ///   sunflower, perpendicular to the line of sight, no RNG). A ray that crosses
    ///   surfaces is attenuated by the per-band amplitude `transmission` of EVERY
    ///   surface it crosses (two walls attenuate more than one; transmission 0
    ///   blocks).
    /// * **Soft visibility.** `v` = fraction of unobstructed rays, so the result
    ///   varies smoothly while the source moves across an edge (penumbra of about
    ///   one disc diameter).
    /// * **Diffraction.** For blocked rays the shortest one-point detour around
    ///   the occluder gives the path-length difference `delta`. When that route
    ///   cannot clear the geometry, a deterministic two-waypoint search traces
    ///   around a second blocker and cascades two Kurze-Anderson edge terms. This
    ///   two-edge approximation is validated against synthetic geometry and the
    ///   analytic cascade to 1e-6 amplitude; it has not been calibrated against
    ///   BEM/UTD or measured doorway/corridor data. Per band the Fresnel number
    ///   `N = 2 delta f / c` feeds the Kurze-Anderson single-edge attenuation
    ///   `A = 5 + 20 log10(sqrt(2 pi N) / tanh sqrt(2 pi N))` dB, so high bands are
    ///   shadowed more than low ones. No detour within reach means no diffraction.
    ///   This remains an engineering approximation, not a validated two-edge
    ///   acoustics model: scene triangles have no diffraction-edge identity, wedge
    ///   angle, finite edge extent, or complex phase. Published multiple-edge UTD
    ///   formulations require those inputs and coherent fields. For example,
    ///   Rodríguez et al. (JASA 2017, doi:10.1121/1.4997942) validates a special
    ///   equal-height/equal-spacing obstacle array, not a staggered doorway. A
    ///   doorway/corridor tolerance needs a geometry-matched BEM or measured
    ///   reference and cannot be inferred from the synthetic route test below.
    /// * **Combination.** Shadow amplitude `S = min(1, sqrt(mean T^2 + D^2))` (through-wall
    ///   and around-edge energy add), blended in dB with the visibility:
    ///   `O = S^(1 - v)`. A fully clear path returns exactly 1 in every band.
    ///
    /// Step bound: one ray flipping between blocked and clear changes the result by
    /// `|20 log10 S| / OCCLUSION_RAYS` dB per band, i.e. at most 1.5 dB per ray for a
    /// 20 dB shadow; a source moving less than `2 * radius / OCCLUSION_RAYS` per
    /// compute tick flips about one ray per tick.
    fn compute_occlusion(
        &self,
        source: &[f32; 3],
        listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> OcclusionResult {
        let clear = OcclusionResult { bands: Band8::splat(1.0), occluded: false };
        let dist = distance3(*source, *listener);
        if dist < 2.0 * OCCLUSION_EPS || self.bvh.is_none() {
            return clear;
        }

        // Basis perpendicular to the line of sight.
        let axis = normalize3(sub3(*source, *listener));
        let (u, w) = probe_basis(axis);

        let mut visible = 0usize;
        let mut blocked = 0usize;
        let mut t2_sum = [0.0_f32; 8];
        // Blocked ray used for the detour: the first in pattern order, i.e. the one
        // nearest the centre (index 0 = centre, then growing radius).
        let mut reference: Option<([f32; 3], [f32; 3])> = None;

        for k in 0..OCCLUSION_RAYS {
            let target = if k == 0 {
                *source
            } else {
                let i = (k - 1) as f32;
                let n = (OCCLUSION_RAYS - 1) as f32;
                let r = OCCLUSION_SOURCE_RADIUS * ((i + 0.5) / n).sqrt();
                let theta = i * GOLDEN_ANGLE;
                let (st, ct) = theta.sin_cos();
                [
                    source[0] + (u[0] * ct + w[0] * st) * r,
                    source[1] + (u[1] * ct + w[1] * st) * r,
                    source[2] + (u[2] * ct + w[2] * st) * r,
                ]
            };
            let (t, crossings, first) = self.segment_transmission(*listener, target, materials);
            if crossings == 0 {
                visible += 1;
            } else {
                blocked += 1;
                for b in 0..8 {
                    t2_sum[b] += t.0[b] * t.0[b];
                }
                if reference.is_none() {
                    if let Some(h) = first {
                        reference = Some((target, h));
                    }
                }
            }
        }

        if blocked == 0 {
            return clear;
        }

        // Diffraction amplitude per band for the blocked rays.
        let delta = reference.and_then(|(target, h)| self.detour_extra_path(*listener, target, h, u, w));
        // Preserve the calibrated single-edge path whenever it exists. The second
        // edge search is a fallback for corner/doorway shadows the one-point route
        // cannot clear, avoiding mode switching at ordinary single edges.
        let two_edges = if delta.is_none() {
            reference.and_then(|(target, h)| self.double_detour_excesses(*listener, target, h, u, w))
        } else { None };
        let bands = if let Some(deltas) = two_edges {
            combine_occlusion_two_edges(visible, blocked, &t2_sum, Some(deltas), self.config.speed_of_sound)
        } else {
            combine_occlusion(visible, blocked, &t2_sum, delta, self.config.speed_of_sound)
        };
        OcclusionResult { bands, occluded: true }
    }

    /// Image-source specular early reflections, orders `1..=max_reflection_order`.
    ///
    /// **Algorithm.** The source is mirrored across candidate planes (see
    /// [`ReflectPlane`]) to build the image tree: `I_k` is `I_{k-1}` mirrored across
    /// the `k`-th plane of the sequence. A sequence `p_1 .. p_N` yields a path iff,
    /// walking back from the listener, the segment `L -> I_N` crosses plane `p_N` at
    /// a point `B_N` inside the surface (a triangle of that plane), the segment
    /// `B_N -> I_{N-1}` crosses `p_{N-1}` inside its surface, and so on down to
    /// `B_1`; the specular law then holds at every bounce by construction. Every
    /// segment `L B_N, B_N B_{N-1}, .., B_1 S` is cast against the BVH and must be
    /// traced against the BVH (the surface being bounced from is excluded by an
    /// epsilon at the segment ends). Per-band transmission gains multiply along
    /// the segments, so fully transmissive blockers preserve the reflection.
    /// Scattering is not modelled (purely specular).
    ///
    /// **Per path** (an [`EarlyReflection`]):
    /// * `direction`: unit vector from the listener toward the **last** reflection
    ///   point `B_N`, world space.
    /// * `delay_samples`: total path length (sum of the segments) `* fs / c`.
    /// * `gain` per band: `prod_k sqrt(1 - alpha_b(theta_k))` over the bounces
    ///   (`alpha` = absorption of the surface the bounce point lies on, evaluated
    ///   at that bounce's incidence angle; `sqrt` because absorption is an energy
    ///   fraction and the gain an amplitude) times the shared distance model and
    ///   ISO 9613-1 air absorption evaluated at the TOTAL path length, times an
    ///   edge window (1 in the interior of a surface, smoothstep to 0 over
    ///   `reflection_edge_fade` metres at its border).
    /// * `order`: number of bounces.
    ///
    /// The strongest `max_reflections` paths (energy `sum_b gain_b^2`, ties by
    /// order then delay) are returned, strongest first; near-identical paths (same
    /// order, length and direction, e.g. from two almost-coplanar surfaces) are
    /// merged. The result is deterministic.
    ///
    /// **Continuity.** A path fades to zero gain when its bounce point approaches
    /// the border of its surface (edge window above), so it appears / disappears
    /// continuously as the listener moves across that boundary. Every path segment
    /// averages transmission over a deterministic 9-ray tube of 12 cm diameter.
    /// One ray changing from clear to fully blocked changes that segment gain by at
    /// most 1/9 of its unobstructed value per update. Fully transmissive blockers
    /// preserve the reflection.
    ///
    /// **Cost** per query: the image tree has `sum_{d=1..N} P (P-1)^(d-1)` nodes
    /// (`P` = `max_reflection_planes`, `N` = order; 32 planes, order 3: 31.8 k), each
    /// costing a mirror plus (at a leaf of the validation) up to `d` cheap
    /// segment-plane-polygon tests; BVH ray casts (`order + 1` per path) are only
    /// done for paths that passed the geometric tests (typically a few dozen). At
    /// most [`MAX_IMAGE_NODES`] nodes are expanded. A 6-plane room at order 3 is
    /// 156 nodes, microseconds; see `image_source_tests::cost_report`.
    fn trace_early_reflections(
        &self,
        source: &[f32; 3],
        listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> Vec<EarlyReflection> {
        let order = (self.config.max_reflection_order as usize).min(MAX_IMAGE_ORDER);
        if order == 0 || self.planes.is_empty() || self.bvh.is_none() {
            return Vec::new();
        }
        if source.iter().chain(listener.iter()).any(|v| !v.is_finite()) {
            return Vec::new();
        }
        let mut st = ImageSearch {
            source: *source,
            listener: *listener,
            order,
            seq: [0; MAX_IMAGE_ORDER],
            images: [[0.0; 3]; MAX_IMAGE_ORDER + 1],
            nodes: 0,
            found: Vec::new(),
            debug_paths: Vec::new(),
        };
        st.images[0] = *source;
        self.expand_images(&mut st, 0, materials);
        let reflections = rank_reflections(st.found, &self.config);
        for mut path in st.debug_paths {
            path.selected = reflections.iter().any(|r| {
                r.order == path.reflection.order && r.direction == path.reflection.direction
                    && r.delay_samples == path.reflection.delay_samples && r.gain.0 == path.reflection.gain.0
            });
            self.debug_capture.record_path(path);
        }
        reflections
    }

    /// Depth-first image-tree expansion: node at `depth` holds `images[0..=depth]`
    /// and `seq[..depth]`; each child mirrors `images[depth]` across one more plane.
    fn expand_images(&self, st: &mut ImageSearch, depth: usize, materials: &dyn MaterialProvider) {
        for pi in 0..self.planes.len() {
            if st.nodes >= MAX_IMAGE_NODES {
                return;
            }
            if depth > 0 && st.seq[depth - 1] == pi {
                continue; // a plane cannot reflect twice in a row
            }
            let plane = &self.planes[pi];
            if depth == 0 && plane.signed_distance(st.source).abs() < PLANE_SIDE_EPS {
                continue; // source lies on the surface: nothing to mirror
            }
            st.nodes += 1;
            st.seq[depth] = pi;
            let img = plane.mirror(st.images[depth]);
            st.images[depth + 1] = img;
            let n = depth + 1;

            // Total path length of a valid path equals |L - I_n|.
            if distance3(st.listener, img) <= self.config.max_reflection_distance {
                if let Some(c) = self.validate_image_path(st, n, materials) {
                    st.found.push(c);
                }
            }
            if n < st.order {
                self.expand_images(st, n, materials);
            }
        }
    }

    /// Geometric + visibility validation of the sequence `st.seq[..n]` (images
    /// `st.images[..=n]`); builds the [`EarlyReflection`] when it is a real path.
    fn validate_image_path(
        &self,
        st: &mut ImageSearch,
        n: usize,
        materials: &dyn MaterialProvider,
    ) -> Option<PathCandidate> {
        // Bounce points B_n .. B_1 (pts[k - 1] = B_k) and the triangle each lies on.
        let mut pts = [[0.0_f32; 3]; MAX_IMAGE_ORDER];
        let mut tri_of = [0usize; MAX_IMAGE_ORDER];
        let mut edge_w = 1.0_f32;
        let mut prev = st.listener;
        for k in (1..=n).rev() {
            let plane = &self.planes[st.seq[k - 1]];
            let target = st.images[k];
            let da = plane.signed_distance(prev);
            let db = plane.signed_distance(target);
            // The segment must really cross the plane (strictly, away from it).
            if !(da * db < 0.0) || da.abs() < PLANE_SIDE_EPS || db.abs() < PLANE_SIDE_EPS {
                return None;
            }
            let t = da / (da - db);
            let b = [
                prev[0] + (target[0] - prev[0]) * t,
                prev[1] + (target[1] - prev[1]) * t,
                prev[2] + (target[2] - prev[2]) * t,
            ];
            let (tri, w) = self.locate_on_plane(plane, b)?;
            pts[k - 1] = b;
            tri_of[k - 1] = tri;
            edge_w *= w;
            prev = b;
        }

        // Visibility of every segment L -> B_n -> .. -> B_1 -> S.
        let mut total = 0.0_f32;
        let mut blocker_gain = Band8::splat(1.0);
        let mut from = st.listener;
        for k in (0..n).rev() {
            let to = pts[k];
            let segment_gain = self.segment_soft_transmission(from, to, materials);
            blocker_gain = blocker_gain.mul(&segment_gain);
            total += distance3(from, to);
            from = to;
        }
        let segment_gain = self.segment_soft_transmission(from, st.source, materials);
        blocker_gain = blocker_gain.mul(&segment_gain);
        total += distance3(from, st.source);
        if !(total.is_finite() && total > 0.0) || total > self.config.max_reflection_distance {
            return None;
        }
        let seq: [usize; MAX_IMAGE_ORDER] = st.seq;
        let mut candidate = path_candidate(
            &self.config,
            &self.distance_model,
            &self.planes,
            &self.triangles,
            &seq[..n],
            &tri_of[..n],
            &pts[..n],
            st.source,
            st.listener,
            total,
            edge_w,
            materials,
        )?;
        candidate.refl.gain = candidate.refl.gain.mul(&blocker_gain);
        candidate.energy = candidate.refl.gain.0.iter().map(|g| g * g).sum();
        if !(candidate.energy > 1e-14) { return None; }
        if self.debug_capture.is_enabled() {
            st.debug_paths.push(crate::debug_capture::DebugReflectionPath {
                source: st.source,
                listener: st.listener,
                bounces: pts[..n].to_vec(),
                normals: tri_of[..n].iter().map(|&i| self.triangles[i].normal).collect(),
                material_handles: tri_of[..n].iter().map(|&i| self.triangles[i].material_handle).collect(),
                reflection: candidate.refl.clone(),
                selected: false,
            });
        }
        Some(candidate)
    }

    /// Triangle of `plane` containing the in-plane point `p` plus the edge window
    /// (1 inside, smoothstep to 0 at the surface border over `reflection_edge_fade`).
    fn locate_on_plane(&self, plane: &ReflectPlane, p: [f32; 3]) -> Option<(usize, f32)> {
        if !plane.contains_in_box(p) {
            return None;
        }
        let tri = plane.tris.iter().copied().find(|&ti| self.triangles[ti].contains_point(p))?;
        let fade = self.config.reflection_edge_fade;
        if !(fade > 0.0) {
            return Some((tri, 1.0));
        }
        let mut dist = f32::INFINITY;
        for (a, b) in &plane.boundary {
            dist = dist.min(point_segment_distance(p, *a, *b));
        }
        let s = (dist / fade).clamp(0.0, 1.0);
        Some((tri, s * s * (3.0 - 2.0 * s)))
    }

    /// Statistical late-field estimate (diffuse-field theory), from the room statistics
    /// computed when the scene was built (see [`RoomStats`]).
    ///
    /// * **Absorption.** Per band, the random-incidence absorption of every surface
    ///   material is `a = 2 int_0^{pi/2} alpha(theta) sin(theta) cos(theta) dtheta`
    ///   (Paris formula), integrated numerically with [`PARIS_ANGLES`] midpoint angles
    ///   through the angle-dependent material model, and averaged over the surface area
    ///   `S` (`a_bar = sum A_m a_m / S`).
    /// * **T60.** Eyring with air absorption, `T = 0.161 V / (-S ln(1 - a_bar) + 4 m V)`
    ///   (`m` = ISO 9613-1 energy attenuation in Np/m = dB/m / 4.343). Eyring is the
    ///   consistent choice at any absorption (it tends to Sabine, `0.161 V / (S a_bar)`, for
    ///   small `a_bar`; Sabine over-predicts T60 in absorbent rooms), so it is used
    ///   everywhere rather than taking a min/max of the two. Clamped to 0.05 .. 20 s.
    /// * **Level.** The reverberant field relative to the direct sound at 1 m is
    ///   `16 pi / R` with the room constant `R = S a / (1 - a)` (`a` = band mean of
    ///   `a_bar`); `late_loudness_db` is its amplitude in dB (see
    ///   [`quasar_core::reverb_model`]). It is independent of the distance (diffuse
    ///   field), which is why the engine's reverb send does not follow the direct gain; the
    ///   direct-to-reverberant ratio at distance `r` is `R / (16 pi r^2)`, critical distance
    ///   `sqrt(R / 16 pi)`. Source / listener dependence: the late energy that follows the
    ///   direct sound is reduced by `exp(-13.82 r / (c T))` (Barron's revised theory: the
    ///   field has decayed while the direct sound travelled `r`), and a source or listener
    ///   outside the room's bounding box is treated as outside the room (20 dB per outside
    ///   party).
    /// * **Volume.** From the mesh (signed tetrahedra) when the scene is a closed,
    ///   consistently wound surface, else the bounding-box VOLUME with a warning printed
    ///   when the scene is built.
    fn estimate_late_reverb(
        &self,
        source: &[f32; 3],
        listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> LateReverbEstimate {
        late_reverb_from_room(&self.room, &self.config, source, listener, materials)
    }
}

/// The statistical late-field estimate of [`CpuSimdComputeBackend::estimate_late_reverb`]
/// as a free function of the precomputed room statistics, so other backends (the WGPU
/// backend precomputes the same [`RoomStats`] at `update_scene`) produce bit-identical
/// results from identical inputs.
pub(crate) fn late_reverb_from_room(
    room: &RoomStats,
    cfg: &CpuSimdConfig,
    source: &[f32; 3],
    listener: &[f32; 3],
    materials: &dyn MaterialProvider,
) -> LateReverbEstimate {
    if room.area < 1e-6 || room.volume < 1e-6 || room.by_material.is_empty() {
        // No geometry = no room: effectively anechoic.
        return LateReverbEstimate {
            t60: Band8::splat(0.3),
            early_late_split_secs: 0.05,
            late_loudness_db: quasar_core::reverb_model::LATE_DB_MIN,
        };
    }
    let mut abs_area = [0.0_f32; 8];
    for &(handle, area) in &room.by_material {
        let a = random_incidence_absorption(materials, handle, cfg);
        for b in 0..8 {
            abs_area[b] += a.0[b] * area;
        }
    }
    let mut t60 = [0.0_f32; 8];
    let mut a_mean = 0.0_f32;
    for b in 0..8 {
        let a_bar = (abs_area[b] / room.area).clamp(0.001, 0.999);
        a_mean += a_bar / 8.0;
        // Air energy attenuation m (Np/m) = dB/m / 4.343.
        let m = quasar_core::air::air_absorption_db_per_m(
            quasar_core::bands::FREQ_BAND_CENTRES[b],
            cfg.temperature_celsius,
            cfg.humidity_percent,
        ) / 4.343;
        let denom = -room.area * (1.0 - a_bar).ln() + 4.0 * m * room.volume;
        t60[b] = (quasar_core::reverb_model::SABINE_K * room.volume / denom).clamp(0.05, 20.0);
    }
    let t_mean = t60.iter().sum::<f32>() / 8.0;

    let room_constant = room.area * a_mean / (1.0 - a_mean);
    let mut level_db = 10.0 * (16.0 * std::f32::consts::PI / room_constant).log10();
    let r = distance3(*source, *listener);
    level_db -= 60.0 * r / (cfg.speed_of_sound * t_mean); // exp(-13.82 r / (c T)), in dB
    for p in [source, listener] {
        if !room.contains(*p) {
            level_db -= 20.0;
        }
    }

    LateReverbEstimate {
        t60: Band8::new(t60),
        // Late field begins after the mixing time ~ sqrt(V) ms (V in m^3), 20 .. 150 ms.
        early_late_split_secs: (room.volume.sqrt() * 1e-3).clamp(0.02, 0.15),
        late_loudness_db: level_db.clamp(quasar_core::reverb_model::LATE_DB_MIN, quasar_core::reverb_model::LATE_DB_MAX),
    }
}

impl IAcousticComputeBackend for CpuSimdComputeBackend {
    fn query_spatial(
        &self,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Vec<SpatialQueryResult> {
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;

        if queries.is_empty() {
            return Vec::new();
        }

        let results: Vec<SpatialQueryResult> = queries
            .par_iter()
            .map(|q| {
                let direct = self.compute_direct_path(&q.source_position, &q.listener_position, materials);
                let early = self.trace_early_reflections(&q.source_position, &q.listener_position, materials);
                let late = self.estimate_late_reverb(&q.source_position, &q.listener_position, materials);

                SpatialQueryResult {
                    source_id: q.source_id,
                    direct_path: direct,
                    early_reflections: early,
                    late_reverb: late,
                }
            })
            .collect();

        results
    }

    fn set_distance_model(&mut self, model: DistanceModel) {
        self.distance_model = model;
    }

    fn set_atmosphere(&mut self, temperature_celsius: f32, humidity_percent: f32) {
        if temperature_celsius.is_finite() && humidity_percent.is_finite() {
            self.config.temperature_celsius = temperature_celsius;
            self.config.humidity_percent = humidity_percent;
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        if sample_rate.is_finite() && sample_rate > 0.0 {
            self.config.sample_rate = sample_rate;
        }
    }

    fn supports_dynamic_geometry(&self) -> bool {
        true
    }

    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError> {
        self.scene = scene.clone();
        self.build_bvh();
        Ok(())
    }

    fn trace_ray(&self, ray: &Ray) -> Vec<RayHit> {
        self.trace_single_ray(ray).into_iter().collect()
    }
}

#[cfg(test)]
mod two_edge_diffraction_tests {
    use super::*;

    #[test]
    // Synthetic analytical reference only: this does not establish agreement with
    // BEM, UTD, or measured room responses.
    fn cascaded_two_edge_attenuation_matches_analytic_kurze_anderson_within_1e_6() {
        let deltas = (0.15_f32, 0.22_f32);
        let got = combine_occlusion_two_edges(0, OCCLUSION_RAYS, &[0.0; 8], Some(deltas), 343.0);
        for band in 0..8 {
            let edge = |delta: f32| {
                let n = 2.0 * delta * quasar_core::bands::FREQ_BAND_CENTRES[band] / 343.0;
                let x = (2.0 * std::f32::consts::PI * n).sqrt();
                let db = (5.0 + 20.0 * (x / x.tanh()).log10()).min(OCCLUSION_MAX_DIFFRACTION_DB);
                let amplitude = 10.0_f32.powf(-db / 20.0);
                amplitude
            };
            let expected = (edge(deltas.0) * edge(deltas.1)).max(OCCLUSION_FLOOR);
            assert!((got.0[band] - expected).abs() <= 1.0e-6, "band {band}: {} vs {expected}", got.0[band]);
        }
    }
}
