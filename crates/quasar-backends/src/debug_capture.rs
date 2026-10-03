//! Opt-in snapshots of geometry visited by the CPU acoustic solver.
use quasar_core::backend::EarlyReflection;
use quasar_core::rays::{Ray, RayHit};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct DebugRay {
    pub ray: Ray,
    pub hit: Option<RayHit>,
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
}

#[derive(Clone, Debug, Default)]
pub struct AcousticDebugFrame {
    pub rays: Vec<DebugRay>,
    pub paths: Vec<DebugReflectionPath>,
}

/// Shared with the demo UI. This capture runs only on the spatial compute thread.
#[derive(Default)]
pub struct AcousticDebugCapture {
    enabled: AtomicBool,
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
    /// Clear the snapshot just before starting one full batch of source/listener queries.
    pub fn begin_update(&self) {
        let mut frame = self.frame.lock().unwrap();
        frame.rays.clear();
        frame.paths.clear();
    }
    pub fn take_frame(&self) -> AcousticDebugFrame {
        std::mem::take(&mut *self.frame.lock().unwrap())
    }
    pub(crate) fn record_ray(&self, ray: Ray, hit: Option<RayHit>) {
        self.frame.lock().unwrap().rays.push(DebugRay { ray, hit });
    }
    pub(crate) fn record_path(&self, path: DebugReflectionPath) {
        self.frame.lock().unwrap().paths.push(path);
    }
}
