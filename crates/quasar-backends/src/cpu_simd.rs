use quasar_core::backend::{
    DirectPathResult, EarlyReflection, IAcousticComputeBackend, LateReverbEstimate,
    MaterialProvider, SpatialQuery, SpatialQueryResult,
};
use quasar_core::bands::Band8;
use quasar_core::error::SpatialAudioError;
use quasar_core::rays::{Ray, RayHit, RayInteractionContext};
use quasar_core::scene::AcousticScene;

/// CPU-based spatial compute backend using BVH-accelerated ray tracing.
///
/// Features:
/// - BVH acceleration structure (SAH builder)
/// - Multithreaded ray execution via rayon
/// - Möller-Trumbore triangle intersection
/// - Specular path tracing for early reflections (up to configurable order)
/// - Sabine/Eyring statistical late reverberation estimation
/// - Dynamic scene update support (rebuilds BVH)
pub struct CpuSimdComputeBackend {
    scene: AcousticScene,
    bvh: Option<BvhNode>,
    config: CpuSimdConfig,
}

/// Configuration for the CPU SIMD backend.
#[derive(Clone, Debug)]
pub struct CpuSimdConfig {
    /// Maximum specular bounce order for early reflections (default: 3).
    pub max_reflection_order: u32,
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
}

impl Default for CpuSimdConfig {
    fn default() -> Self {
        Self {
            max_reflection_order: 3,
            diffuse_rays_per_query: 64,
            max_reflection_distance: 50.0,
            speed_of_sound: 343.0,
            temperature_celsius: 20.0,
            humidity_percent: 50.0,
        }
    }
}

// ── AABB ─────────────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
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

    fn intersect(&self, ray: &Ray) -> bool {
        let mut tmin = ray.min_distance;
        let mut tmax = ray.max_distance;
        for i in 0..3 {
            let inv_d = 1.0 / ray.direction[i];
            let t1 = (self.min[i] - ray.origin[i]) * inv_d;
            let t2 = (self.max[i] - ray.origin[i]) * inv_d;
            let ta = t1.min(t2);
            let tb = t1.max(t2);
            tmin = tmin.max(ta);
            tmax = tmax.min(tb);
            if tmin > tmax {
                return false;
            }
        }
        true
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
struct Triangle {
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    normal: [f32; 3],
    material_handle: u32,
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

    /// Möller-Trumbore ray-triangle intersection.
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let edge1 = sub3(self.b, self.a);
        let edge2 = sub3(self.c, self.a);
        let h = cross3(ray.direction, edge2);
        let det = dot3(edge1, h);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let s = sub3(ray.origin, self.a);
        let u = dot3(s, h) * inv_det;
        if u < 0.0 || u > 1.0 {
            return None;
        }
        let q = cross3(s, edge1);
        let v = dot3(ray.direction, q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = dot3(edge2, q) * inv_det;
        if t < ray.min_distance || t > ray.max_distance {
            return None;
        }
        Some(t)
    }

    fn centroid(&self) -> [f32; 3] {
        [
            (self.a[0] + self.b[0] + self.c[0]) / 3.0,
            (self.a[1] + self.b[1] + self.c[1]) / 3.0,
            (self.a[2] + self.b[2] + self.c[2]) / 3.0,
        ]
    }

    fn aabb(&self) -> Aabb {
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
        match self {
            BvhNode::Internal { left, right, .. } => {
                if !self.aabb().intersect(ray) {
                    return None;
                }
                let hit_left = left.intersect(ray);
                let hit_right = right.intersect(ray);
                match (hit_left, hit_right) {
                    (Some(l), Some(r)) => {
                        if l.distance <= r.distance {
                            Some(l)
                        } else {
                            Some(r)
                        }
                    }
                    (Some(h), None) | (None, Some(h)) => Some(h),
                    (None, None) => None,
                }
            }
            BvhNode::Leaf { triangles, .. } => {
                let mut closest: Option<RayHit> = None;
                for tri in triangles {
                    if let Some(t) = tri.intersect(ray) {
                        let point = ray.point_at(t);
                        match &closest {
                            Some(ref best) if t >= best.distance => {}
                            _ => {
                                closest = Some(RayHit {
                                    distance: t,
                                    point,
                                    normal: tri.normal,
                                    material_handle: tri.material_handle,
                                    hit: true,
                                });
                            }
                        }
                    }
                }
                closest
            }
        }
    }

}

// ── Vec3 helpers ──────────────────────────────────────────────────────

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[inline]
fn reflect3(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let d = dot3(incident, normal);
    [incident[0] - 2.0 * d * normal[0], incident[1] - 2.0 * d * normal[1], incident[2] - 2.0 * d * normal[2]]
}

#[inline]
fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
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
            config,
        };
        backend.build_bvh();
        backend
    }

    /// Build the BVH acceleration structure from the scene geometry.
    pub fn build_bvh(&mut self) {
        let mut triangles = Self::triangles_from_scene(&self.scene);
        if triangles.is_empty() {
            self.bvh = None;
            return;
        }
        self.bvh = Some(BvhNode::build(&mut triangles));
    }

    fn triangles_from_scene(scene: &AcousticScene) -> Vec<Triangle> {
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
                tris.push(Triangle::new(a, b, c, mesh.material_handle, mesh.id));
            }
        }
        tris
    }

    /// Trace a single ray through the BVH.
    fn trace_single_ray(&self, ray: &Ray) -> Option<RayHit> {
        self.bvh.as_ref().and_then(|bvh| bvh.intersect(ray))
    }

    /// Trace a ray and evaluate the material response at the hit point.
    fn trace_with_material(
        &self,
        ray: &Ray,
        materials: &dyn MaterialProvider,
        _context: &RayInteractionContext,
    ) -> (Option<RayHit>, quasar_materials::evaluator::AcousticResponse8Band) {
        let hit = self.trace_single_ray(ray);
        let response = match &hit {
            Some(h) if h.hit => {
                let ctx = RayInteractionContext {
                    surface_normal: h.normal,
                    ray_direction: ray.direction,
                    incident_angle_rad: dot3(
                        normalize3([-ray.direction[0], -ray.direction[1], -ray.direction[2]]),
                        h.normal,
                    )
                    .acos(),
                    temperature_celsius: self.config.temperature_celsius,
                    humidity_percent: self.config.humidity_percent,
                };
                let absorption = materials.evaluate_material(h.material_handle, &ctx);
                quasar_materials::evaluator::AcousticResponse8Band::new(
                    absorption,
                    Band8::zeros(),
                    Band8::zeros(),
                )
            }
            _ => quasar_materials::evaluator::AcousticResponse8Band::air(),
        };
        (hit, response)
    }

    /// Compute the direct path between source and listener.
    fn compute_direct_path(
        &self,
        source: &[f32; 3],
        listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> DirectPathResult {
        let dist = distance3(*source, *listener);

        let mut occluded = false;
        let mut occlusion_factor = 1.0f32;

        let shadow_ray = Ray {
            origin: *listener,
            // Trace TOWARD the source: a wall between source and listener must
            // be on the listener->source segment. The previous code traced
            // `normalize(listener - source)`, i.e. away from the source, so
            // occlusion_factor was always 1.0 and M1 was dead code in practice.
            direction: normalize3(sub3(*source, *listener)),
            min_distance: 0.01,
            max_distance: dist - 0.01,
        };

        if let Some(hit) = self.trace_single_ray(&shadow_ray) {
            if hit.hit {
                occluded = true;
                let ctx = RayInteractionContext {
                    surface_normal: hit.normal,
                    ray_direction: shadow_ray.direction,
                    incident_angle_rad: dot3(
                        normalize3([
                            -shadow_ray.direction[0],
                            -shadow_ray.direction[1],
                            -shadow_ray.direction[2],
                        ]),
                        hit.normal,
                    )
                    .acos(),
                    temperature_celsius: self.config.temperature_celsius,
                    humidity_percent: self.config.humidity_percent,
                };
                let absorption = materials.evaluate_material(hit.material_handle, &ctx);
                occlusion_factor = 1.0 - absorption.mean().clamp(0.0, 0.95);
            }
        }

        let atten = Self::distance_attenuation(dist);
        let air = Self::air_absorption(
            dist,
            self.config.temperature_celsius,
            self.config.humidity_percent,
        );
        let mut total_atten = atten.mul(&air);
        // Fold the traced occlusion factor into the direct-path attenuation.
        // A clear line of sight keeps occlusion_factor == 1.0 and skips the loop;
        // an occluding surface (factor < 1.0) scales every band down so the
        // direct gain and the per-band occlusion lowpass both respond to geometry.
        if occlusion_factor < 1.0 {
            for band in total_atten.0.iter_mut() {
                *band *= occlusion_factor;
            }
        }

        DirectPathResult {
            attenuation: total_atten,
            delay_samples: dist * 48_000.0 / self.config.speed_of_sound,
            distance: dist,
            occluded,
            occlusion_factor,
        }
    }

    /// Trace specular early reflections up to max_reflection_order.
    fn trace_early_reflections(
        &self,
        source: &[f32; 3],
        listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> Vec<EarlyReflection> {
        let mut reflections = Vec::new();
        let mut path: Vec<[f32; 3]> = vec![*listener];

        self.trace_reflection_bounce(
            *listener,
            *source,
            normalize3(sub3(*listener, *source)),
            1,
            &mut path,
            materials,
            &mut reflections,
        );

        reflections
    }

    fn trace_reflection_bounce(
        &self,
        origin: [f32; 3],
        source: [f32; 3],
        incoming_dir: [f32; 3],
        order: u32,
        _path: &mut Vec<[f32; 3]>,
        materials: &dyn MaterialProvider,
        reflections: &mut Vec<EarlyReflection>,
    ) {
        if order > self.config.max_reflection_order {
            return;
        }

        let ray = Ray {
            origin,
            direction: incoming_dir,
            min_distance: 0.01,
            max_distance: self.config.max_reflection_distance,
        };

        let (hit, response) = self.trace_with_material(&ray, materials, &RayInteractionContext::default());

        if let Some(h) = hit {
            if !h.hit {
                return;
            }

            let reflected_dir = reflect3(incoming_dir, h.normal);

            let to_listener = sub3(h.point, source);
            let refl_dist = (to_listener[0] * to_listener[0]
                + to_listener[1] * to_listener[1]
                + to_listener[2] * to_listener[2])
                .sqrt();

            let total_dist = h.distance + refl_dist;

            let mut gain = Band8::splat(1.0);
            for b in 0..8 {
                let reflect_coeff = 1.0 - response.absorption.0[b];
                gain.0[b] = reflect_coeff / (1.0 + total_dist * 0.1);
            }

            let refl_dir_from_listener = normalize3(sub3(h.point, source));

            reflections.push(EarlyReflection {
                direction: refl_dir_from_listener,
                delay_samples: total_dist * 48_000.0 / self.config.speed_of_sound,
                gain,
                order,
            });

            self.trace_reflection_bounce(
                h.point,
                source,
                reflected_dir,
                order + 1,
                _path,
                materials,
                reflections,
            );
        }
    }

    /// Estimate late reverberation parameters using statistical acoustics.
    fn estimate_late_reverb(
        &self,
        _source: &[f32; 3],
        _listener: &[f32; 3],
        materials: &dyn MaterialProvider,
    ) -> LateReverbEstimate {
        let mut total_surface_area = 0.0f32;
        let mut total_absorption = [0.0f32; 8];
        let mut tri_count = 0;
        let mut scene_aabb = Aabb::new_empty();

        for mesh in &self.scene.meshes {
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
                scene_aabb = scene_aabb.union(&Aabb::from_points(&[a, b, c]));

                let e1 = sub3(b, a);
                let e2 = sub3(c, a);
                let n = cross3(e1, e2);
                let area = 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                total_surface_area += area;

                let ctx = RayInteractionContext::default();
                let abs_val = materials.evaluate_material(mesh.material_handle, &ctx);
                for b in 0..8 {
                    total_absorption[b] += abs_val.0[b] * area;
                }
                tri_count += 1;
            }
        }

        if total_surface_area < 1e-8 || tri_count == 0 {
            return LateReverbEstimate {
                t60: Band8::splat(0.3),
                early_late_split_secs: 0.05,
                late_loudness_db: -20.0,
            };
        }

        let room_volume = scene_aabb.surface_area() * 0.5;

        let mut t60 = [0.0f32; 8];
        for b in 0..8 {
            let avg_absorption = (total_absorption[b] / total_surface_area).clamp(0.01, 0.99);
            let sabine = 0.161 * room_volume / (total_surface_area * avg_absorption);
            let eyring = 0.161 * room_volume / (-total_surface_area * (1.0 - avg_absorption).ln());
            t60[b] = sabine.min(eyring).clamp(0.1, 10.0);
        }

        let late_energy_db = -10.0 - 10.0 * (total_absorption.iter().sum::<f32>() / 8.0).log10();

        LateReverbEstimate {
            t60: Band8::new(t60),
            early_late_split_secs: 0.05,
            late_loudness_db: late_energy_db,
        }
    }

    /// Compute distance-based attenuation per band (inverse distance law).
    fn distance_attenuation(distance: f32) -> Band8 {
        let atten = 1.0 / (1.0 + distance);
        Band8::splat(atten.clamp(0.0, 1.0))
    }

    /// Compute air absorption per band for a given distance (ISO 9613-1 simplified).
    fn air_absorption(distance: f32, temperature: f32, humility: f32) -> Band8 {
        let centres = quasar_core::bands::FREQ_BAND_CENTRES;
        let mut vals = [0.0f32; 8];
        for i in 0..8 {
            let freq = centres[i];
            let alpha = air_absorption_coefficient(freq, temperature, humility);
            vals[i] = (-alpha * distance).exp();
        }
        Band8::new(vals)
    }
}

/// Simplified air absorption coefficient (dB/m) from ISO 9613-1.
fn air_absorption_coefficient(freq_hz: f32, temp_c: f32, h_percent: f32) -> f32 {
    let tk = temp_c + 273.15;
    let pr = 101325.0;
    let psat = 610.94 * ((17.625 * temp_c) / (temp_c + 243.04)).exp();
    let h = h_percent * psat / pr;
    let fr_o = 24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h);
    let fr_n = (tk / 293.15).sqrt() * (9.0 + 350.0 * h * (h / (1.0 - h)).exp());
    let freq = freq_hz;

    let alpha =
        freq * freq
            * (1.84e-11 / (pr * (tk / 293.15).sqrt())
                + (tk / 293.15).powf(-2.5)
                    * (0.01278 * (-2239.1 / tk).exp() / (fr_o + freq * freq / fr_o)
                        + 0.1068 * (-3352.0 / tk).exp() / (fr_n + freq * freq / fr_n)));

    alpha
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

    fn supports_dynamic_geometry(&self) -> bool {
        true
    }

    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError> {
        self.scene = scene.clone();
        self.build_bvh();
        Ok(())
    }

    fn trace_ray(&self, ray: &Ray) -> Vec<RayHit> {
        let mut hits = Vec::new();

        if let Some(bvh) = &self.bvh {
            if let Some(hit) = bvh.intersect(ray) {
                hits.push(hit);
            }
        }

        hits
    }
}
