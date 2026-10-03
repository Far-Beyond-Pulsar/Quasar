//! WGPU compute backend (#78).
//!
//! # What runs where
//!
//! One compute dispatch (a workgroup of 64 threads per source/listener query,
//! all queries of a batch in the same dispatch) runs the **geometric** work on the GPU:
//!
//! * **Direct-path occlusion probes.** The 13 probe rays of the CPU backend
//!   (`OCCLUSION_RAYS`: the source centre plus a golden-angle disc) are walked
//!   through the scene recording EVERY surface crossed (material handle, incidence
//!   cosine, normal), and the shortest one-point detour around the occluder (the
//!   diffraction path difference) is searched on the device.
//! * **Image-source early reflections.** The mirror planes (coplanar triangles
//!   merged, precomputed on the CPU at `update_scene`) are uploaded; the GPU walks
//!   the image tree (one thread per first-bounce plane), validates every path
//!   (plane crossing inside the surface, edge window, visibility of every segment)
//!   and returns the validated paths with their bounce points.
//!
//! The **host** then finishes with the same code the CPU backend uses
//! (`cpu_simd::{combine_occlusion, path_candidate, rank_reflections,
//! late_reverb_from_room}`): per-band transmission / absorption from the
//! [`MaterialProvider`] trait object (which cannot run on a GPU) for exactly the
//! crossings and bounces the GPU found, the shared [`DistanceModel`], ISO 9613-1
//! air absorption, the Kurze-Anderson diffraction blend, ranking / merging of the
//! paths, and the **late reverb** (Eyring T60 / diffuse level from the room
//! statistics precomputed at `update_scene`: it is `O(materials)` per query and
//! needs the material trait object, so it is evaluated on the host; identical to the
//! CPU backend by construction). Results are therefore the same field-for-field as
//! `CpuSimdComputeBackend`'s up to f32 rounding of the geometry (see the parity
//! test `tests/wgpu_parity_tests.rs`).
//!
//! # Limits
//!
//! * **Scaling.** Triangles are tested brute force: every ray costs `O(triangles)`
//!   (no BVH on the GPU). Intended for room-model scenes (up to a few thousand
//!   triangles); a flattened BVH is future work. Cost per query is roughly
//!   `(13 + 8 * (2 * (log2(26 / 0.05) + 5)) + P^order) * T` triangle tests.
//! * **Planes.** At most 64 mirror planes (one GPU thread per first-bounce plane,
//!   workgroup size 64), reflection order at most 8; the image-tree node budget is
//!   per thread (`MAX_IMAGE_NODES / planes`, the CPU applies it per query).
//! * **Candidate capacity.** At most [`WgpuComputeConfig::max_candidates_per_query`]
//!   validated paths per query come back; beyond that paths are dropped in a
//!   non-deterministic order (a message is printed once). Raise the capacity if a
//!   scene is that rich; the CPU backend has no such cap before ranking.
//! * **Precision.** All geometry runs in f32 on the device (transcendental-free; the
//!   disc / detour direction tables are precomputed). A ray grazing a triangle edge
//!   can flip between devices; the parity test documents its tolerances.
//! * **Threading.** `query_spatial` blocks the calling thread (`map_async` +
//!   `device.poll(Wait)`): call it from the compute thread, never the audio thread.
//! * **Errors.** [`WgpuComputeBackend::try_query_spatial`] returns errors; the
//!   trait method (which cannot) prints the error once and returns an EMPTY vector
//!   (never a fabricated result; `HybridProbeSampler` already treats it as an error).
//!   No adapter / insufficient limits / shader failure are explicit
//!   [`SpatialAudioError::Backend`] errors from the constructors.
//! * **Atmosphere.** Temperature / humidity come from [`WgpuComputeConfig`]; there is
//!   updated through `IAcousticComputeBackend::set_atmosphere` (#121).
//! * **Dispatch size.** Batches larger than `max_sources_per_dispatch` (and than the
//!   device's storage-binding limit) are split into several dispatches; every query
//!   gets a result, in order.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use quasar_core::backend::{
    DirectPathResult, IAcousticComputeBackend, MaterialProvider, SpatialQuery, SpatialQueryResult,
};
use quasar_core::bands::Band8;
use quasar_core::distance::DistanceModel;
use quasar_core::error::SpatialAudioError;
use quasar_core::rays::{Ray, RayHit, RayInteractionContext};
use quasar_core::scene::AcousticScene;

use crate::cpu_simd::{
    build_planes, combine_occlusion, cross3, distance3, late_reverb_from_room, normalize3,
    path_candidate, rank_reflections, sub3, CpuSimdComputeBackend, CpuSimdConfig, PathCandidate,
    ReflectPlane, RoomStats, Triangle, BARY_EPS, GOLDEN_ANGLE, MAX_IMAGE_NODES, MAX_IMAGE_ORDER,
    OCCLUSION_BISECT_STEPS, OCCLUSION_DETOUR_DIRS, OCCLUSION_DETOUR_MARGIN,
    OCCLUSION_DETOUR_MAX_OFFSET, OCCLUSION_DETOUR_MIN_OFFSET, OCCLUSION_EPS,
    OCCLUSION_MAX_CROSSINGS, OCCLUSION_RAYS, OCCLUSION_SOURCE_RADIUS, PLANE_BOX_PAD,
    PLANE_SIDE_EPS,
};

// The shader hard-codes these array sizes; keep them in lockstep.
const _: () = assert!(OCCLUSION_RAYS == 13);
const _: () = assert!(OCCLUSION_MAX_CROSSINGS == 8);
const _: () = assert!(OCCLUSION_DETOUR_DIRS == 8);
const _: () = assert!(MAX_IMAGE_ORDER == 8);

/// Threads per workgroup = most mirror planes (one thread per first-bounce plane).
const WORKGROUP_SIZE: usize = 64;
/// Storage buffers the pipeline binds (a limit the device must offer).
const STORAGE_BUFFERS: u32 = 7;

/// Configuration for the WGPU compute backend.
#[derive(Clone, Debug)]
pub struct WgpuComputeConfig {
    /// Maximum specular bounce order for early reflections (default: 3; 0..=8).
    pub max_reflection_order: u32,
    /// Maximum early-reflection paths returned per query, strongest first (default: 16).
    pub max_reflections: usize,
    /// Maximum distinct mirror planes, largest first (default: 32; at most 64).
    pub max_reflection_planes: usize,
    /// Width (m) of the fade at the border of a reflecting surface (default: 0.1).
    pub reflection_edge_fade: f32,
    /// Maximum total path length of a reflection (default: 50 m).
    pub max_reflection_distance: f32,
    /// Validated image paths the GPU can return per query before ranking (default: 128).
    pub max_candidates_per_query: u32,
    /// Speed of sound in m/s (default: 343.0).
    pub speed_of_sound: f32,
    /// Air temperature in Celsius (default: 20.0).
    pub temperature_celsius: f32,
    /// Relative humidity percentage (default: 50.0).
    pub humidity_percent: f32,
    /// Audio device sample rate in Hz (default: 48000), see
    /// [`IAcousticComputeBackend::set_sample_rate`].
    pub sample_rate: f32,
    /// Max queries processed in one dispatch (default: 1024); larger batches are split.
    pub max_sources_per_dispatch: u32,
}

impl Default for WgpuComputeConfig {
    fn default() -> Self {
        Self {
            max_reflection_order: 3,
            max_reflections: 16,
            max_reflection_planes: 32,
            reflection_edge_fade: 0.1,
            max_reflection_distance: 50.0,
            max_candidates_per_query: 128,
            speed_of_sound: 343.0,
            temperature_celsius: 20.0,
            humidity_percent: 50.0,
            sample_rate: 48_000.0,
            max_sources_per_dispatch: 1024,
        }
    }
}

impl WgpuComputeConfig {
    fn validate(&self) -> Result<(), SpatialAudioError> {
        let bad = |what: &str| Err(SpatialAudioError::Backend(format!("WgpuComputeConfig: {what}")));
        if self.max_reflection_order as usize > MAX_IMAGE_ORDER {
            return bad("max_reflection_order must be <= 8");
        }
        if self.max_reflection_planes == 0 || self.max_reflection_planes > WORKGROUP_SIZE {
            return bad("max_reflection_planes must be in 1..=64");
        }
        if self.max_candidates_per_query == 0 || self.max_sources_per_dispatch == 0 {
            return bad("max_candidates_per_query and max_sources_per_dispatch must be >= 1");
        }
        if !(self.speed_of_sound.is_finite() && self.speed_of_sound > 0.0) {
            return bad("speed_of_sound must be finite and positive");
        }
        Ok(())
    }

    /// The equivalent CPU-backend configuration (shared host-side code reads it).
    fn cpu(&self) -> CpuSimdConfig {
        CpuSimdConfig {
            max_reflection_order: self.max_reflection_order,
            max_reflections: self.max_reflections,
            max_reflection_planes: self.max_reflection_planes,
            reflection_edge_fade: self.reflection_edge_fade,
            max_reflection_distance: self.max_reflection_distance,
            speed_of_sound: self.speed_of_sound,
            temperature_celsius: self.temperature_celsius,
            humidity_percent: self.humidity_percent,
            sample_rate: self.sample_rate,
            ..CpuSimdConfig::default()
        }
    }
}

/// Scene data kept on the host: the world-space triangles (same order as the GPU
/// buffer, so GPU triangle indices map back to materials), the mirror planes and
/// the room statistics of the late-reverb estimate.
struct HostScene {
    triangles: Vec<Triangle>,
    planes: Vec<ReflectPlane>,
    room: RoomStats,
}

/// Scene buffers on the device.
struct GpuScene {
    tris: wgpu::Buffer,
    planes: wgpu::Buffer,
    plane_tris: wgpu::Buffer,
    edges: wgpu::Buffer,
    n_tris: u32,
    n_planes: u32,
}

/// WGPU-based compute backend: GPU geometry (occlusion probes, image-source paths),
/// host material / statistical stages. See the [module documentation](self).
pub struct WgpuComputeBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    gpu: GpuScene,
    host: HostScene,
    config: WgpuComputeConfig,
    cpu_config: CpuSimdConfig,
    distance_model: DistanceModel,
    /// Queries per dispatch after the device's buffer limits.
    chunk_len: usize,
    error_logged: AtomicBool,
    overflow_logged: AtomicBool,
}

impl WgpuComputeBackend {
    /// Request a high-performance adapter and a device for the backend, blocking the
    /// calling thread. `Err(Backend("no suitable GPU adapter ..."))` when the machine
    /// has no usable adapter (headless CI), so callers can fall back explicitly.
    pub fn request_headless_device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), SpatialAudioError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = block_on(
            None,
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }),
        )
        .ok_or_else(|| {
            SpatialAudioError::Backend("no suitable GPU adapter found for the wgpu compute backend".into())
        })?;
        let info = adapter.get_info();
        let (device, queue) = block_on(
            None,
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("quasar_wgpu_compute"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            ),
        )
        .map_err(|e| {
            SpatialAudioError::Backend(format!("adapter '{}' ({:?}) refused the device: {e}", info.name, info.backend))
        })?;
        Ok((Arc::new(device), Arc::new(queue)))
    }

    /// Create a backend on an existing device: compiles the shader (a compile or
    /// validation failure is an `Err`), builds the pipeline and uploads `scene`.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        scene: AcousticScene,
        config: WgpuComputeConfig,
    ) -> Result<Self, SpatialAudioError> {
        config.validate()?;
        let limits = device.limits();
        if limits.max_storage_buffers_per_shader_stage < STORAGE_BUFFERS
            || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE as u32
            || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE as u32
        {
            return Err(SpatialAudioError::Backend(format!(
                "device limits too low for the wgpu compute backend (need {STORAGE_BUFFERS} storage buffers \
                 and 64-thread workgroups, have {} / {} / {})",
                limits.max_storage_buffers_per_shader_stage,
                limits.max_compute_workgroup_size_x,
                limits.max_compute_invocations_per_workgroup
            )));
        }
        let per_query = per_query_bytes(config.max_candidates_per_query);
        let max_bind = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        if per_query > max_bind {
            return Err(SpatialAudioError::Backend(format!(
                "max_candidates_per_query = {} needs {per_query} bytes per query, the device binds at most {max_bind}",
                config.max_candidates_per_query
            )));
        }
        let chunk_len = ((max_bind / per_query) as usize)
            .min(config.max_sources_per_dispatch as usize)
            .min(65_535)
            .max(1);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bind_group_layout = Self::create_layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quasar_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quasar_ray_trace_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../shaders/ray_trace.wgsl"))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("quasar_ray_trace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(err) = block_on(Some(&device), device.pop_error_scope()) {
            return Err(SpatialAudioError::Backend(format!("wgpu compute shader / pipeline: {err}")));
        }

        let cpu_config = config.cpu();
        let (host, gpu) = Self::build_scene(&device, &queue, &scene, &config)?;
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            gpu,
            host,
            config,
            cpu_config,
            distance_model: DistanceModel::default(),
            chunk_len,
            error_logged: AtomicBool::new(false),
            overflow_logged: AtomicBool::new(false),
        })
    }

    /// [`Self::request_headless_device`] followed by [`Self::new`].
    pub fn new_headless(scene: AcousticScene, config: WgpuComputeConfig) -> Result<Self, SpatialAudioError> {
        let (device, queue) = Self::request_headless_device()?;
        Self::new(device, queue, scene, config)
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use std::num::NonZeroU64;
        let entry = |binding: u32, ty: wgpu::BufferBindingType, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size as u64),
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quasar_bind_group_layout"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform, std::mem::size_of::<GpuParams>()),
                entry(1, ro, std::mem::size_of::<GpuQuery>()),
                entry(2, ro, std::mem::size_of::<GpuTri>()),
                entry(3, ro, std::mem::size_of::<GpuPlane>()),
                entry(4, ro, std::mem::size_of::<u32>()),
                entry(5, ro, std::mem::size_of::<GpuEdge>()),
                entry(6, rw, std::mem::size_of::<GpuHead>()),
                entry(7, rw, std::mem::size_of::<GpuCand>()),
            ],
        })
    }

    /// Preprocess `scene` on the CPU (world-space triangles, mirror planes, room
    /// statistics) and upload the geometry buffers.
    fn build_scene(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &AcousticScene,
        config: &WgpuComputeConfig,
    ) -> Result<(HostScene, GpuScene), SpatialAudioError> {
        let triangles = CpuSimdComputeBackend::triangles_from_scene(scene);
        let planes = build_planes(&triangles, config.max_reflection_planes);
        let room = RoomStats::build(&triangles);

        let gpu_tris: Vec<GpuTri> = triangles
            .iter()
            .map(|t| GpuTri {
                a: [t.a[0], t.a[1], t.a[2], 0.0],
                b: [t.b[0], t.b[1], t.b[2], 0.0],
                c: [t.c[0], t.c[1], t.c[2], 0.0],
                n: [t.normal[0], t.normal[1], t.normal[2], 0.0],
                mat: [t.material_handle, 0, 0, 0],
            })
            .collect();
        let mut plane_tris: Vec<u32> = Vec::new();
        let mut edges: Vec<GpuEdge> = Vec::new();
        let gpu_planes: Vec<GpuPlane> = planes
            .iter()
            .map(|p| {
                let tri_start = plane_tris.len() as u32;
                plane_tris.extend(p.tris.iter().map(|&t| t as u32));
                let edge_start = edges.len() as u32;
                edges.extend(p.boundary.iter().map(|(a, b)| GpuEdge {
                    a: [a[0], a[1], a[2], 0.0],
                    b: [b[0], b[1], b[2], 0.0],
                }));
                GpuPlane {
                    no: [p.normal[0], p.normal[1], p.normal[2], p.offset],
                    bmin: [p.aabb.min[0], p.aabb.min[1], p.aabb.min[2], 0.0],
                    bmax: [p.aabb.max[0], p.aabb.max[1], p.aabb.max[2], 0.0],
                    ranges: [tri_start, p.tris.len() as u32, edge_start, p.boundary.len() as u32],
                }
            })
            .collect();

        let limits = device.limits();
        let tri_bytes = (gpu_tris.len() * std::mem::size_of::<GpuTri>()) as u64;
        if tri_bytes > (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size) {
            return Err(SpatialAudioError::InvalidScene(format!(
                "{} triangles do not fit one storage binding of this device",
                gpu_tris.len()
            )));
        }

        let upload = |label: &str, data: &[u8], min: usize| {
            // Storage bindings must be non-empty: an empty list is one zeroed element
            // (the shader never reads it: its count is 0).
            let size = data.len().max(min) as u64;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !data.is_empty() {
                queue.write_buffer(&buf, 0, data);
            }
            buf
        };
        let gpu = GpuScene {
            tris: upload("quasar_tris", bytemuck::cast_slice(&gpu_tris), std::mem::size_of::<GpuTri>()),
            planes: upload("quasar_planes", bytemuck::cast_slice(&gpu_planes), std::mem::size_of::<GpuPlane>()),
            plane_tris: upload("quasar_plane_tris", bytemuck::cast_slice(&plane_tris), 4),
            edges: upload("quasar_edges", bytemuck::cast_slice(&edges), std::mem::size_of::<GpuEdge>()),
            n_tris: gpu_tris.len() as u32,
            n_planes: gpu_planes.len() as u32,
        };
        Ok((HostScene { triangles, planes, room }, gpu))
    }

    /// Shader uniform for a dispatch of `n` queries.
    fn params(&self, n: usize) -> GpuParams {
        let mut disc = [[0.0_f32; 4]; 13];
        for k in 1..OCCLUSION_RAYS {
            let i = (k - 1) as f32;
            let m = (OCCLUSION_RAYS - 1) as f32;
            let r = OCCLUSION_SOURCE_RADIUS * ((i + 0.5) / m).sqrt();
            let (st, ct) = (i * GOLDEN_ANGLE).sin_cos();
            disc[k] = [r * ct, r * st, 0.0, 0.0];
        }
        let mut detour = [[0.0_f32; 4]; 8];
        for (j, d) in detour.iter_mut().enumerate() {
            let phi = j as f32 * (2.0 * std::f32::consts::PI / OCCLUSION_DETOUR_DIRS as f32);
            let (s, c) = phi.sin_cos();
            *d = [c, s, 0.0, 0.0];
        }
        GpuParams {
            counts0: [n as u32, self.gpu.n_tris, self.gpu.n_planes, self.config.max_reflection_order],
            counts1: [
                self.config.max_candidates_per_query,
                OCCLUSION_BISECT_STEPS as u32,
                (MAX_IMAGE_NODES / (self.gpu.n_planes.max(1) as usize)).max(1) as u32,
                0,
            ],
            limits0: [
                OCCLUSION_EPS,
                OCCLUSION_DETOUR_MIN_OFFSET,
                OCCLUSION_DETOUR_MAX_OFFSET,
                OCCLUSION_DETOUR_MARGIN,
            ],
            limits1: [
                self.config.max_reflection_distance,
                self.config.reflection_edge_fade,
                PLANE_SIDE_EPS,
                PLANE_BOX_PAD,
            ],
            limits2: [BARY_EPS, 0.0, 0.0, 0.0],
            disc,
            detour,
        }
    }

    /// Run `chunk` (at most `chunk_len` queries) in one dispatch and read the raw
    /// output back: the per-query heads followed by the candidate paths.
    fn dispatch_chunk(&self, chunk: &[SpatialQuery]) -> Result<Vec<u8>, SpatialAudioError> {
        let n = chunk.len();
        let cap = self.config.max_candidates_per_query as usize;
        let heads_bytes = (n * std::mem::size_of::<GpuHead>()) as u64;
        let cands_bytes = (n * cap * std::mem::size_of::<GpuCand>()) as u64;

        let gpu_queries: Vec<GpuQuery> = chunk
            .iter()
            .map(|q| {
                let valid = q.source_position.iter().chain(q.listener_position.iter()).all(|v| v.is_finite());
                let s = if valid { q.source_position } else { [0.0; 3] };
                let l = if valid { q.listener_position } else { [0.0; 3] };
                GpuQuery {
                    source: [s[0], s[1], s[2], if valid { 1.0 } else { 0.0 }],
                    listener: [l[0], l[1], l[2], 0.0],
                }
            })
            .collect();
        let params = self.params(n);

        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let params_buf = mk("quasar_params", std::mem::size_of::<GpuParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let queries_buf = mk("quasar_queries", (n * std::mem::size_of::<GpuQuery>()) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let heads_buf = mk("quasar_heads", heads_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let cands_buf = mk("quasar_cands", cands_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let staging = mk("quasar_staging", heads_bytes + cands_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&queries_buf, 0, bytemuck::cast_slice(&gpu_queries));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quasar_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: queries_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.gpu.tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.gpu.planes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.gpu.plane_tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.gpu.edges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: heads_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cands_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("quasar_dispatch_encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quasar_ray_trace_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1); // one workgroup per query
        }
        encoder.copy_buffer_to_buffer(&heads_buf, 0, &staging, 0, heads_bytes);
        encoder.copy_buffer_to_buffer(&cands_buf, 0, &staging, heads_bytes, cands_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);

        let validation = block_on(Some(&self.device), self.device.pop_error_scope());
        let oom = block_on(Some(&self.device), self.device.pop_error_scope());
        if let Some(e) = validation.or(oom) {
            return Err(SpatialAudioError::Backend(format!("wgpu compute dispatch failed: {e}")));
        }
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(SpatialAudioError::Backend(format!("wgpu readback failed: {e}"))),
            Err(_) => return Err(SpatialAudioError::Backend("wgpu readback callback was dropped".into())),
        }
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(bytes)
    }

    /// Like [`IAcousticComputeBackend::query_spatial`] but reports failures.
    ///
    /// Exactly one result per query, in query order (batches larger than the
    /// per-dispatch limit are split into several dispatches).
    pub fn try_query_spatial(
        &self,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Result<Vec<SpatialQueryResult>, SpatialAudioError> {
        let mut out = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(self.chunk_len) {
            let bytes = self.dispatch_chunk(chunk)?;
            let heads_len = chunk.len() * std::mem::size_of::<GpuHead>();
            let (heads, cands) = bytes.split_at(heads_len);
            for (i, q) in chunk.iter().enumerate() {
                let head: GpuHead = bytemuck::pod_read_unaligned(
                    &heads[i * std::mem::size_of::<GpuHead>()..(i + 1) * std::mem::size_of::<GpuHead>()],
                );
                out.push(self.decode(q, &head, cands, i, materials));
            }
        }
        Ok(out)
    }

    /// Turn the raw GPU output of query `q` into a [`SpatialQueryResult`].
    fn decode(
        &self,
        q: &SpatialQuery,
        head: &GpuHead,
        cands: &[u8],
        index: usize,
        materials: &dyn MaterialProvider,
    ) -> SpatialQueryResult {
        let src = q.source_position;
        let lis = q.listener_position;
        let finite = src.iter().chain(lis.iter()).all(|v| v.is_finite());
        let cfg = &self.cpu_config;
        let dist = distance3(src, lis);

        // ---- direct path ----
        let (occlusion, occluded) = if finite { self.decode_occlusion(src, lis, dist, head, materials) } else { (Band8::splat(1.0), false) };
        let atten = Band8::splat(self.distance_model.gain(dist));
        let air = quasar_core::air::air_absorption_gain(dist, cfg.temperature_celsius, cfg.humidity_percent);
        let direct_path = DirectPathResult {
            attenuation: atten.mul(&air).mul(&occlusion),
            delay_samples: dist * cfg.sample_rate / cfg.speed_of_sound,
            distance: dist,
            occluded,
            occlusion_factor: occlusion.mean(),
            occlusion,
        };

        // ---- early reflections ----
        let early_reflections = if finite { self.decode_reflections(src, lis, head, cands, index, materials) } else { Vec::new() };

        // ---- late reverb (host, shared estimator) ----
        let late_reverb = late_reverb_from_room(&self.host.room, cfg, &src, &lis, materials);

        SpatialQueryResult { source_id: q.source_id, direct_path, early_reflections, late_reverb }
    }

    /// Per-band occlusion from the probe crossings (the CPU backend's
    /// `compute_occlusion`, with the traversal done by the GPU).
    fn decode_occlusion(
        &self,
        src: [f32; 3],
        lis: [f32; 3],
        dist: f32,
        head: &GpuHead,
        materials: &dyn MaterialProvider,
    ) -> (Band8, bool) {
        let clear = (Band8::splat(1.0), false);
        if dist < 2.0 * OCCLUSION_EPS || self.host.triangles.is_empty() {
            return clear;
        }
        // Same basis / probe pattern as the shader (and the CPU backend).
        let axis = normalize3(sub3(src, lis));
        let helper = if axis[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let u = normalize3(cross3(axis, helper));
        let w = cross3(axis, u);
        let params = self.params(0);

        let mut visible = 0usize;
        let mut blocked = 0usize;
        let mut t2_sum = [0.0_f32; 8];
        for k in 0..OCCLUSION_RAYS {
            let ray = &head.rays[k];
            let count = ray.count[0] as usize;
            if count == 0 {
                visible += 1;
                continue;
            }
            blocked += 1;
            if count > OCCLUSION_MAX_CROSSINGS {
                continue; // more surfaces than the model resolves: opaque (T = 0)
            }
            let off = params.disc[k];
            let target = [
                src[0] + u[0] * off[0] + w[0] * off[1],
                src[1] + u[1] * off[0] + w[1] * off[1],
                src[2] + u[2] * off[0] + w[2] * off[1],
            ];
            let dir = normalize3(sub3(target, lis));
            let mut product = Band8::splat(1.0);
            for c in &ray.cr[..count] {
                let ctx = RayInteractionContext {
                    surface_normal: [c.n_cos[0], c.n_cos[1], c.n_cos[2]],
                    ray_direction: dir,
                    incident_angle_rad: c.n_cos[3].clamp(-1.0, 1.0).acos(),
                    temperature_celsius: self.cpu_config.temperature_celsius,
                    humidity_percent: self.cpu_config.humidity_percent,
                };
                let t = materials.evaluate_transmission(c.mat[0], &ctx);
                for b in 0..8 {
                    product.0[b] *= t.0[b].clamp(0.0, 1.0);
                }
            }
            for b in 0..8 {
                t2_sum[b] += product.0[b] * product.0[b];
            }
        }
        if blocked == 0 {
            return clear;
        }
        let delta = if head.delta[0] >= 0.0 { Some(head.delta[0]) } else { None };
        (combine_occlusion(visible, blocked, &t2_sum, delta, self.cpu_config.speed_of_sound), true)
    }

    /// Gains and ranking of the validated image paths the GPU returned.
    fn decode_reflections(
        &self,
        src: [f32; 3],
        lis: [f32; 3],
        head: &GpuHead,
        cands: &[u8],
        index: usize,
        materials: &dyn MaterialProvider,
    ) -> Vec<quasar_core::backend::EarlyReflection> {
        let cap = self.config.max_candidates_per_query as usize;
        let found = head.info[0] as usize;
        if found > cap && !self.overflow_logged.swap(true, Ordering::Relaxed) {
            eprintln!(
                "quasar-backends: wgpu compute found {found} image paths for one query but \
                 max_candidates_per_query = {cap}; the excess is dropped (raise the capacity)"
            );
        }
        let stride = std::mem::size_of::<GpuCand>();
        let base = index * cap * stride;
        let mut raw: Vec<GpuCand> = (0..found.min(cap))
            .map(|i| bytemuck::pod_read_unaligned(&cands[base + i * stride..base + (i + 1) * stride]))
            .collect();
        // The GPU appends in a non-deterministic order; restore the CPU backend's
        // discovery order (depth-first = lexicographic plane sequence, prefixes first)
        // so ties in the ranking resolve identically.
        let key = |c: &GpuCand| c.seq[..(c.info[0] as usize).min(8)].to_vec();
        raw.sort_by(|a, b| key(a).cmp(&key(b)));

        let mut paths: Vec<PathCandidate> = Vec::with_capacity(raw.len());
        for c in &raw {
            let n = (c.info[0] as usize).min(MAX_IMAGE_ORDER);
            if n == 0 {
                continue;
            }
            let seq: Vec<usize> = c.seq[..n].iter().map(|&p| p as usize).collect();
            let tri_of: Vec<usize> = c.tri[..n].iter().map(|&t| t as usize).collect();
            if seq.iter().any(|&p| p >= self.host.planes.len()) || tri_of.iter().any(|&t| t >= self.host.triangles.len()) {
                continue; // corrupt record: never index out of range
            }
            let pts: Vec<[f32; 3]> = c.pts[..n].iter().map(|p| [p[0], p[1], p[2]]).collect();
            if let Some(p) = path_candidate(
                &self.cpu_config,
                &self.distance_model,
                &self.host.planes,
                &self.host.triangles,
                &seq,
                &tri_of,
                &pts,
                src,
                lis,
                c.head[1],
                c.head[0],
                materials,
            ) {
                paths.push(p);
            }
        }
        rank_reflections(paths, &self.cpu_config)
    }

    fn log_error_once(&self, e: &SpatialAudioError) {
        if !self.error_logged.swap(true, Ordering::Relaxed) {
            eprintln!("quasar-backends: wgpu compute query failed, returning no results: {e}");
        }
    }
}

impl IAcousticComputeBackend for WgpuComputeBackend {
    /// Never fabricates a result: on a GPU failure the error is printed once and an
    /// EMPTY vector is returned (use [`WgpuComputeBackend::try_query_spatial`] to get
    /// the error). Blocks until the GPU finishes: compute thread only.
    fn query_spatial(
        &self,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Vec<SpatialQueryResult> {
        match self.try_query_spatial(queries, materials) {
            Ok(r) => r,
            Err(e) => {
                self.log_error_once(&e);
                Vec::new()
            }
        }
    }

    fn set_distance_model(&mut self, model: DistanceModel) {
        self.distance_model = model;
    }

    fn set_atmosphere(&mut self, temperature_celsius: f32, humidity_percent: f32) {
        if temperature_celsius.is_finite() && humidity_percent.is_finite() {
            self.config.temperature_celsius = temperature_celsius;
            self.config.humidity_percent = humidity_percent;
            self.cpu_config.temperature_celsius = temperature_celsius;
            self.cpu_config.humidity_percent = humidity_percent;
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        if sample_rate.is_finite() && sample_rate > 0.0 {
            self.config.sample_rate = sample_rate;
            self.cpu_config.sample_rate = sample_rate;
        }
    }

    fn supports_dynamic_geometry(&self) -> bool {
        true
    }

    /// Re-run the CPU preprocessing (triangles, mirror planes, room statistics) and
    /// replace the geometry buffers. On error the previous scene stays active.
    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError> {
        let (host, gpu) = Self::build_scene(&self.device, &self.queue, scene, &self.config)?;
        self.host = host;
        self.gpu = gpu;
        Ok(())
    }

    /// Closest hit along `ray`, brute force on the host triangles (a single ray is
    /// not worth a GPU round trip).
    fn trace_ray(&self, ray: &Ray) -> Vec<RayHit> {
        let mut best: Option<(f32, &Triangle)> = None;
        for tri in &self.host.triangles {
            let limit = best.map_or(ray.max_distance, |b| b.0);
            if let Some(t) = tri.intersect_max(ray, limit) {
                if best.map_or(true, |b| t < b.0) {
                    best = Some((t, tri));
                }
            }
        }
        best.map(|(t, tri)| RayHit {
            distance: t,
            point: ray.point_at(t),
            normal: tri.normal,
            material_handle: tri.material_handle,
            hit: true,
        })
        .into_iter()
        .collect()
    }
}

// ── Minimal executor ──────────────────────────────────────────────────

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Drive `fut` to completion on the calling thread (std only; wgpu's native futures
/// resolve as the device is polled, which `device` provides while pending).
fn block_on<F: Future>(device: Option<&wgpu::Device>, fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        if let Some(d) = device {
            d.poll(wgpu::Maintain::Poll);
        }
        std::thread::park_timeout(std::time::Duration::from_millis(1));
    }
}

// ── Shader-compatible structs ─────────────────────────────────────────
//
// Mirrors of the WGSL structs in shaders/ray_trace.wgsl. Every field is a 16-byte
// `vec4` (or a scalar array in a `storage` struct, stride 4), so no implicit padding
// exists on either side; `size_of` is asserted in the tests below. All are `repr(C)`
// with 4-byte alignment on the host, while WGSL aligns them to 16: harmless because
// every size is a multiple of 16 and buffers start at offset 0.

/// 416 bytes: 5 `vec4` + 13 + 8 `vec4` tables. Uniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    counts0: [u32; 4],
    counts1: [u32; 4],
    limits0: [f32; 4],
    limits1: [f32; 4],
    limits2: [f32; 4],
    disc: [[f32; 4]; 13],
    detour: [[f32; 4]; 8],
}

/// 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuQuery {
    source: [f32; 4],
    listener: [f32; 4],
}

/// 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTri {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    n: [f32; 4],
    mat: [u32; 4],
}

/// 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPlane {
    no: [f32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
    ranges: [u32; 4],
}

/// 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEdge {
    a: [f32; 4],
    b: [f32; 4],
}

/// 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCrossing {
    n_cos: [f32; 4],
    mat: [u32; 4],
}

/// 288 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRayOut {
    count: [u32; 4],
    first: [f32; 4],
    cr: [GpuCrossing; 8],
}

/// 3776 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHead {
    info: [u32; 4],
    delta: [f32; 4],
    rays: [GpuRayOut; 13],
}

/// 224 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCand {
    head: [f32; 4],
    info: [u32; 4],
    seq: [u32; 8],
    tri: [u32; 8],
    pts: [[f32; 4]; 8],
}

/// Output bytes per query: its head plus `max_candidates` candidate records.
fn per_query_bytes(max_candidates: u32) -> u64 {
    (std::mem::size_of::<GpuHead>() + max_candidates as usize * std::mem::size_of::<GpuCand>()) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirrors must match the byte sizes the WGSL structs have (documented in
    /// shaders/ray_trace.wgsl); a mismatch is the classic std140/std430 failure.
    #[test]
    fn struct_sizes_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GpuParams>(), 416);
        assert_eq!(std::mem::size_of::<GpuQuery>(), 32);
        assert_eq!(std::mem::size_of::<GpuTri>(), 80);
        assert_eq!(std::mem::size_of::<GpuPlane>(), 64);
        assert_eq!(std::mem::size_of::<GpuEdge>(), 32);
        assert_eq!(std::mem::size_of::<GpuCrossing>(), 32);
        assert_eq!(std::mem::size_of::<GpuRayOut>(), 288);
        assert_eq!(std::mem::size_of::<GpuHead>(), 3776);
        assert_eq!(std::mem::size_of::<GpuCand>(), 224);
        // Every size is a multiple of the 16-byte WGSL alignment of its vec4 members.
        for s in [416, 32, 80, 64, 32, 32, 288, 3776, 224] {
            assert_eq!(s % 16, 0);
        }
    }

    #[test]
    fn config_validation_rejects_unsupported_settings() {
        assert!(WgpuComputeConfig::default().validate().is_ok());
        let bad = |f: &dyn Fn(&mut WgpuComputeConfig)| {
            let mut c = WgpuComputeConfig::default();
            f(&mut c);
            c.validate().is_err()
        };
        assert!(bad(&|c| c.max_reflection_order = 9));
        assert!(bad(&|c| c.max_reflection_planes = 65));
        assert!(bad(&|c| c.max_reflection_planes = 0));
        assert!(bad(&|c| c.max_candidates_per_query = 0));
        assert!(bad(&|c| c.speed_of_sound = 0.0));
    }

    #[test]
    fn shader_source_has_no_placeholder_constants() {
        let src = include_str!("../shaders/ray_trace.wgsl");
        assert!(!src.contains("1000.0"));
        assert!(!src.contains("num_workgroups"));
    }
}
