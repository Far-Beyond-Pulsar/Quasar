//! Opt-in snapshots of geometry visited by the CPU acoustic solver.
//!
//! The solver fires thousands of BVH rays per query (soft-occlusion probes, diffraction
//! detour probing, reflection validation). Storing and drawing all of them is useless and
//! costly, so every ray carries a [`DebugRayKind`] and only the kinds enabled in the
//! [`CaptureDetail`] are stored (none of the probe kinds by default). Rays are collected
//! per query in a thread-local buffer (one query runs on one thread) and pushed to the
//! shared frame with a single lock per query, under hard per-frame caps.
use quasar_core::backend::EarlyReflection;
use quasar_core::rays::{Ray, RayHit};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;

/// Hard cap on stored probe rays per frame (`AcousticDebugFrame::rays`).
pub const MAX_STORED_RAYS: usize = 4096;
/// Hard cap on stored rejected reflection candidates per frame.
pub const MAX_STORED_REJECTED: usize = 512;
/// Cap on rejected candidates one query may contribute (the image tree has tens of
/// thousands of candidates; the first ones found in tree order are kept).
pub const MAX_REJECTED_PER_QUERY: usize = 64;

/// Which solver stage fired a stored ray.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DebugRayKind {
    /// Soft-occlusion probe rays listener -> disc around the source.
    OcclusionProbe,
    /// Diffraction detour search / bisection segments.
    DiffractionProbe,
    /// Per-segment visibility rays of a candidate reflection path.
    ReflectionValidation,
    /// Rays fired through the public `trace_ray`.
    #[default]
    Other,
}

/// Why an image-source candidate did not become a valid path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Every segment was traced and the blockers removed all energy.
    Blocked,
    /// A bounce point fell outside the bounds of its reflecting surface.
    OutsideSurface,
    /// The bounce point is on the border of its surface: the edge window is ~0.
    EdgeFade,
    /// The path energy is below the cutoff (absorption / distance).
    BelowEnergy,
}

/// What the capture stores besides the always-kept per-query data (direct segment summary
/// and valid reflection paths). All off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureDetail {
    /// Store the 13 soft-occlusion probe rays per direct-path query.
    pub occlusion_probes: bool,
    /// Store diffraction detour probe rays (many per blocked query).
    pub diffraction_probes: bool,
    /// Store reflection-path validation segments.
    pub reflection_validation: bool,
    /// Store rejected reflection candidates with their [`RejectReason`] (capped).
    pub rejected_paths: bool,
}

impl CaptureDetail {
    pub const NONE: Self = Self { occlusion_probes: false, diffraction_probes: false, reflection_validation: false, rejected_paths: false };
    pub const ALL: Self = Self { occlusion_probes: true, diffraction_probes: true, reflection_validation: true, rejected_paths: true };
    fn bits(self) -> u8 {
        self.occlusion_probes as u8 | (self.diffraction_probes as u8) << 1 | (self.reflection_validation as u8) << 2 | (self.rejected_paths as u8) << 3
    }
    fn from_bits(b: u8) -> Self {
        Self { occlusion_probes: b & 1 != 0, diffraction_probes: b & 2 != 0, reflection_validation: b & 4 != 0, rejected_paths: b & 8 != 0 }
    }
    fn stores(self, kind: DebugRayKind) -> bool {
        match kind {
            DebugRayKind::OcclusionProbe => self.occlusion_probes,
            DebugRayKind::DiffractionProbe => self.diffraction_probes,
            DebugRayKind::ReflectionValidation => self.reflection_validation,
            DebugRayKind::Other => true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DebugRay {
    pub ray: Ray,
    pub hit: Option<RayHit>,
    pub kind: DebugRayKind,
    /// `SpatialQuery::source_id` of the query that fired the ray (0 outside a query).
    pub source_id: u32,
    /// Index of the query within its `query_spatial` batch.
    pub query_index: u32,
}

/// Bounce coordinates and their real acoustic surface data, ordered source to listener.
#[derive(Clone, Debug)]
pub struct DebugReflectionPath {
    pub source: [f32; 3],
    pub listener: [f32; 3],
    pub bounces: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub material_handles: Vec<u32>,
    pub reflection: EarlyReflection,
    /// Whether the backend returned this path to the audio engine after ranking.
    pub selected: bool,
    pub source_id: u32,
    pub query_index: u32,
}

/// A reflection candidate that failed validation (stored in detailed mode only).
#[derive(Clone, Debug)]
pub struct DebugRejectedPath {
    pub source: [f32; 3],
    pub listener: [f32; 3],
    /// Bounce points found before the candidate failed (listener side first).
    pub bounces: Vec<[f32; 3]>,
    pub reason: RejectReason,
    pub source_id: u32,
    pub query_index: u32,
}

/// Always-kept summary of one source/listener query: the direct segment and what the
/// solver did for it.
#[derive(Clone, Debug)]
pub struct DebugDirect {
    pub source: [f32; 3],
    pub listener: [f32; 3],
    pub source_id: u32,
    pub query_index: u32,
    pub occluded: bool,
    /// Mean per-band occlusion amplitude (1 clear .. 0 blocked).
    pub occlusion_factor: f32,
    /// Source / listener lie outside the closed room shell (false when the scene is not closed).
    pub source_outside: bool,
    pub listener_outside: bool,
    /// Early reflections were skipped because an endpoint is outside the closed shell.
    pub reflections_skipped: bool,
    /// BVH rays this query traced (all kinds, stored or not).
    pub rays_traced: u32,
}

#[derive(Clone, Debug, Default)]
pub struct AcousticDebugFrame {
    /// Stored probe rays (kinds enabled by [`CaptureDetail`], at most [`MAX_STORED_RAYS`]).
    pub rays: Vec<DebugRay>,
    /// Valid reflection paths (`selected` marks those returned to the engine).
    pub paths: Vec<DebugReflectionPath>,
    /// One entry per query: the direct segment summary.
    pub directs: Vec<DebugDirect>,
    /// Rejected reflection candidates (detailed mode, at most [`MAX_STORED_REJECTED`]).
    pub rejected: Vec<DebugRejectedPath>,
    /// TOTAL rays traced by the captured queries (not the number stored).
    pub ray_count: u64,
    /// Rays not stored because their kind is disabled in the [`CaptureDetail`].
    pub rays_filtered: u64,
    /// Rays that were wanted but dropped by the per-frame cap.
    pub rays_dropped: u64,
    /// Rejected candidates dropped by the cap.
    pub rejected_dropped: u64,
}

/// Per-query collector (thread-local; a query never migrates between threads).
#[derive(Default)]
struct Local {
    active: bool,
    detail: CaptureDetail,
    kind: DebugRayKind,
    source_id: u32,
    query_index: u32,
    rays: Vec<DebugRay>,
    rejected: Vec<DebugRejectedPath>,
    paths: Vec<DebugReflectionPath>,
    traced: u64,
    filtered: u64,
    dropped: u64,
    rejected_dropped: u64,
}

thread_local! {
    static LOCAL: RefCell<Local> = RefCell::new(Local::default());
}

/// Shared with the demo UI. This capture runs only on the spatial compute thread.
#[derive(Default)]
pub struct AcousticDebugCapture {
    enabled: AtomicBool,
    detail: AtomicU8,
    frame: Mutex<AcousticDebugFrame>,
}

impl AcousticDebugCapture {
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        self.begin_update();
    }
    /// Which probe kinds are stored (default: none, see [`CaptureDetail`]).
    pub fn detail(&self) -> CaptureDetail {
        CaptureDetail::from_bits(self.detail.load(Ordering::Relaxed))
    }
    pub fn set_detail(&self, detail: CaptureDetail) {
        self.detail.store(detail.bits(), Ordering::Relaxed);
    }
    /// Clear the snapshot just before starting one full batch of source/listener queries.
    pub fn begin_update(&self) {
        *self.frame.lock().unwrap() = AcousticDebugFrame::default();
    }
    pub fn take_frame(&self) -> AcousticDebugFrame {
        std::mem::take(&mut *self.frame.lock().unwrap())
    }

    /// Start collecting one query on this thread.
    pub(crate) fn begin_query(&self, source_id: u32, query_index: u32) {
        let detail = self.detail();
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            l.active = true;
            l.detail = detail;
            l.kind = DebugRayKind::Other;
            l.source_id = source_id;
            l.query_index = query_index;
            l.rays.clear();
            l.rejected.clear();
            l.paths.clear();
            l.traced = 0;
            l.filtered = 0;
            l.dropped = 0;
            l.rejected_dropped = 0;
        });
    }

    /// Set the kind attributed to the rays traced next; returns the previous kind.
    pub(crate) fn set_kind(&self, kind: DebugRayKind) -> DebugRayKind {
        LOCAL.with(|l| std::mem::replace(&mut l.borrow_mut().kind, kind))
    }

    /// Record one traced ray (called only while capture is enabled).
    pub(crate) fn record_ray(&self, ray: &Ray, hit: &Option<RayHit>) {
        let handled = LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            if !l.active {
                return false;
            }
            l.traced += 1;
            if !l.detail.stores(l.kind) {
                l.filtered += 1;
            } else if l.rays.len() < MAX_STORED_RAYS {
                let sample = DebugRay { ray: ray.clone(), hit: hit.clone(), kind: l.kind, source_id: l.source_id, query_index: l.query_index };
                l.rays.push(sample);
            } else {
                l.dropped += 1;
            }
            true
        });
        if !handled {
            // Outside a query (public `trace_ray`): one lock for this one ray.
            let mut f = self.frame.lock().unwrap();
            f.ray_count += 1;
            if f.rays.len() < MAX_STORED_RAYS {
                f.rays.push(DebugRay { ray: ray.clone(), hit: hit.clone(), kind: DebugRayKind::Other, source_id: 0, query_index: 0 });
            } else {
                f.rays_dropped += 1;
            }
        }
    }

    /// Record a rejected candidate (kept only in [`CaptureDetail::rejected_paths`] mode).
    pub(crate) fn record_rejected(&self, source: [f32; 3], listener: [f32; 3], bounces: &[[f32; 3]], reason: RejectReason) {
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            if !l.active || !l.detail.rejected_paths {
                return;
            }
            if l.rejected.len() < MAX_REJECTED_PER_QUERY {
                let r = DebugRejectedPath { source, listener, bounces: bounces.to_vec(), reason, source_id: l.source_id, query_index: l.query_index };
                l.rejected.push(r);
            } else {
                l.rejected_dropped += 1;
            }
        });
    }

    pub(crate) fn rejected_enabled(&self) -> bool {
        self.detail().rejected_paths
    }

    /// Finish the query: push everything to the shared frame with one lock.
    /// `paths` are the valid reflection paths of this query.
    pub(crate) fn end_query(&self, mut direct: DebugDirect) {
        let (rays, rejected, paths, traced, filtered, dropped, rejected_dropped) = LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            l.active = false;
            (std::mem::take(&mut l.rays), std::mem::take(&mut l.rejected), std::mem::take(&mut l.paths), l.traced, l.filtered, l.dropped, l.rejected_dropped)
        });
        direct.rays_traced = traced.min(u32::MAX as u64) as u32;
        let mut f = self.frame.lock().unwrap();
        f.ray_count += traced;
        f.rays_filtered += filtered;
        f.rays_dropped += dropped;
        f.directs.push(direct);
        f.paths.extend(paths);
        let room = MAX_STORED_RAYS.saturating_sub(f.rays.len());
        if rays.len() > room {
            f.rays_dropped += (rays.len() - room) as u64;
        }
        f.rays.extend(rays.into_iter().take(room));
        let room = MAX_STORED_REJECTED.saturating_sub(f.rejected.len());
        f.rejected_dropped += rejected_dropped + rejected.len().saturating_sub(room) as u64;
        f.rejected.extend(rejected.into_iter().take(room));
    }
}

/// Restores the previous ray kind when dropped (no-op while capture is disabled).
pub(crate) struct KindGuard<'a> {
    capture: &'a AcousticDebugCapture,
    prev: Option<DebugRayKind>,
}

impl AcousticDebugCapture {
    /// Attribute the rays traced until the guard drops to `kind`.
    pub(crate) fn scoped(&self, kind: DebugRayKind) -> KindGuard<'_> {
        let prev = self.is_enabled().then(|| self.set_kind(kind));
        KindGuard { capture: self, prev }
    }
}

impl Drop for KindGuard<'_> {
    fn drop(&mut self) {
        if let Some(prev) = self.prev {
            self.capture.set_kind(prev);
        }
    }
}

impl AcousticDebugCapture {
    /// Record a valid reflection path of the current query (one lock only outside a query).
    pub(crate) fn record_path(&self, mut path: DebugReflectionPath) {
        let outside = LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            if l.active {
                path.source_id = l.source_id;
                path.query_index = l.query_index;
                l.paths.push(path.clone());
                false
            } else {
                true
            }
        });
        if outside {
            self.frame.lock().unwrap().paths.push(path);
        }
    }
}
