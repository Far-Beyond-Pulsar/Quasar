//! Quasar spatial audio in Helio's large HLFS cathedral.
//!
//! The scene is Helio's `indoor_cathedral_hlfs` example in its LARGE mode (a Gothic interior of about
//! 145 x 45 x 43 m: ribbed vaults, clustered limestone piers, marble paving, carved oak pews, bronze
//! chandeliers, leaded stained glass, one shadowed daylight sun and an incense medium), running on
//! Helio's SceneDB integration. Quasar's acoustic geometry is read back from the same SceneDB world
//! (see `acoustic_geometry.rs`): every renderable object is also an acoustic mesh.
//!
//! Eight stage speakers play an 8-channel test WAV through the Quasar engine; the listener is the
//! camera. Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit
//!   F1 / F2 / F3 — Helio shadow debug / perf overlay / debug overlay
//!   V (or R)    — start / pause acoustic ray capture (the last trace stays visible)
//!   C / B / N / M — cycle emitter / all emitters / rejected paths / solver probe rays
//!   T           — toggle the Quasar probe grid overlay
//!   Y           — toggle the acoustic scene bounds and print the acoustic material table
//!   G           — swap Aux Left/Right channels (live patch-bay remap)
//!   [ / ]       — master volume down / up (3 dB steps)
//!   - / =       — reverb trim down / up (2 dB);   ; / '  — early-reflection trim down / up
//!   1 / 2 / 3   — cycle DSP stage / print audio timing / reset timing counters
//!
//! `QUASAR_HEADLESS_CHECK=1` (or `--check`) builds the scene without a window and without an audio
//! device and prints geometry, material, timing and level statistics (see `headless_check.rs`).

mod acoustic_geometry;
mod acoustic_overlay;
mod architectural_materials;
mod architectural_mesh;
mod audio_demo;
mod cathedral_large;
mod headless_check;
mod hlfs_capture;
mod v3_demo_common;

use acoustic_overlay::AcousticView;
use audio_demo::{AudioEngine, ListenerPose, SpatialWorker, NUM_SPEAKERS};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    HelioAction, HelioCommandBridge, Renderer, RendererBuilder, RendererConfig,
};
use helio_default_graphs::build_hlfs_graph_with_context;
use helio_pass_billboard::{BillboardComponent, BillboardPass};
use helio_pass_perf_overlay::PerfOverlayMode;
use pulsar_scenedb::{Entity, SceneDb, World};
use quasar_backends::debug_capture::AcousticDebugFrame;
use v3_demo_common::{
    new_scene_db_with_gpu_mirror, point_light, scene_db_handle, spawn_indoor_cathedral_sky,
    spawn_light, update_light,
};

use std::io::{self, BufRead};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;

// ── Scene data (large cathedral) ──────────────────────────────────────────────

// Stained glass window lights of the legacy lighting setup (`HLFS_LEGACY_CATHEDRAL_LIGHTS`):
// (x_wall_side, y, z, r, g, b), scaled to the large hall where they are spawned.
const GLASS_LIGHTS: &[(f32, f32, f32, f32, f32, f32)] = &[
    (-10.3, 9.0, -22.0, 0.8, 0.2, 1.0), // violet
    (-10.3, 9.0, -6.0, 0.2, 0.7, 1.0),  // sky blue
    (-10.3, 9.0, 10.0, 0.2, 1.0, 0.4),  // emerald
    (-10.3, 9.0, 18.0, 1.0, 0.7, 0.1),  // gold
    (10.3, 9.0, -22.0, 1.0, 0.2, 0.3),  // ruby
    (10.3, 9.0, -6.0, 1.0, 0.5, 0.1),   // amber
    (10.3, 9.0, 10.0, 0.1, 0.8, 0.9),   // teal
    (10.3, 9.0, 18.0, 0.9, 0.1, 0.7),   // magenta
    (0.0, 13.0, 27.0, 1.0, 0.75, 0.3),  // rose window, warm gold
];

/// Chandelier positions along the nave (x = 0, hanging at y = 31).
pub(crate) const LARGE_CHANDELIER_Z: &[f32] = &[-54.0, -36.0, -18.0, 0.0, 18.0, 36.0, 54.0];

/// Candle clusters near the altar (z about -64).
pub(crate) const LARGE_CANDLES: &[(f32, f32, f32)] = &[
    (-4.0, 1.6, -64.0), (-2.0, 1.6, -63.5), (0.0, 1.6, -64.0),
    (2.0, 1.6, -63.5), (4.0, 1.6, -64.0),
];

fn main() {
    env_logger::init();
    if std::env::var_os("QUASAR_SWEEP").is_some() {
        headless_check::sweep();
        return;
    }
    if std::env::var_os("QUASAR_HEADLESS_CHECK").is_some() || std::env::args().any(|a| a == "--check") {
        match headless_check::run() {
            Ok(()) => return,
            Err(e) => {
                eprintln!("headless check failed: {e}");
                std::process::exit(1);
            }
        }
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Arc<Mutex<Renderer>>,
    action_rx: Receiver<HelioAction>,
    last_frame: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    alt_pressed: bool,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Debug
    debug_mode: u32,
    perf_overlay_mode: PerfOverlayMode,
    debug_overlay_enabled: bool,

    scene_db: SceneDb,
    acceleration: Option<helio_pass_hlfs::SceneDbRayTracing>,

    // Scene state
    chandelier_light_ids: Vec<Entity>,
    candle_light_ids: Vec<Entity>,
    start_time: std::time::Instant,
    motion_frame: u32,

    // Quasar spatial audio. The audio callback owns the renderer; `worker` runs the ~30 Hz
    // acoustic update on its own thread.
    audio: AudioEngine,
    worker: SpatialWorker,
    speaker_billboards: Vec<Entity>,
    /// V toggles trace capture; the last nonempty snapshot stays on screen when paused.
    show_rays: bool,
    acoustic_snapshot: Option<AcousticDebugFrame>,
    /// C cycles the emitter, B shows all, N rejected candidates, M probe rays.
    acoustic_view: AcousticView,
    show_probes: bool,
    show_scene_bounds: bool,
    /// Aux Left/Right pulls swapped live via the G key (patch-bay remap).
    aux_swapped: bool,
    last_stats_print: std::time::Instant,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

/// Procedural 32 x 32 white speaker icon (the billboard shader tints it) as RGBA pixels.
fn generate_speaker_icon() -> (Vec<u8>, u32, u32) {
    let w = 32u32;
    let h = 32u32;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let cx = x as i32 - 16;
            let cy = y as i32 - 16;
            let in_cabinet = cx >= -8 && cx <= -3 && cy >= -8 && cy <= 8;
            let in_cone = cx >= -2 && cx <= 8 && cy.abs() <= (10 - cx);
            let in_grill = cx == -3 && cy >= -6 && cy <= 6 && cy % 3 == 0;
            if in_cabinet || in_cone || in_grill {
                let idx = ((y * w + x) * 4) as usize;
                pixels[idx..idx + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
    }
    (pixels, w, h)
}

/// Replace the default billboard sprite with the speaker icon, now and in every graph the renderer
/// rebuilds afterwards (resize), and apply the HLFS ray-traced configuration to rebuilt graphs.
/// Billboards are debug markers: they are not occluded by geometry.
fn install_graph_hook(
    renderer: &mut Renderer,
    queue: Arc<wgpu::Queue>,
    hlfs: Option<helio_pass_hlfs::HlfsConfig>,
) {
    let camera_buf = renderer.camera_buf().clone();
    let format = renderer.renderer_config().surface_format;
    let (rgba, w, h) = generate_speaker_icon();
    renderer.set_graph_rebuild_hook(move |graph, device| {
        if let Some(config) = hlfs {
            graph.find_pass_mut::<helio_pass_hlfs::HlfsPass>().expect("HLFS pass").set_config(device, config);
        }
        if let Some(index) = graph.pass_index_of::<BillboardPass>() {
            let mut pass = BillboardPass::new_with_sprite_rgba(device, &queue, &camera_buf, format, &rgba, w, h);
            pass.set_occluded_by_geometry(false);
            graph.replace_pass_at(index, Box::new(pass));
        }
    });
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio & Quasar; Indoor Cathedral w/ Spatial Audio")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        eprintln!("Interactive cathedral adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("device");
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let config = RendererConfig::new(size.width, size.height, format)
            .with_tsr_quality(helio_pass_tsr::TsrQuality::Native)
            .with_shadow_quality(helio::ShadowQuality::High)
            .with_ssr(true)
            .with_environment_reflections(true);
        let mut scene_db = new_scene_db_with_gpu_mirror(&device, &queue);
        let (chandelier_light_ids, candle_light_ids) = populate_large_cathedral(&mut scene_db.world);
        let ray_traced = std::env::var_os("HLFS_RT").is_some();
        eprintln!("Interactive cathedral: ray_traced={ray_traced} presampled={}",
            std::env::var_os("HLFS_PRESAMPLED").is_some());
        if ray_traced {
            // The renderer captures SceneDB's initial light flags at build.
            // Match the offscreen path by setting RT flags before building it.
            hlfs_capture::enable_ray_shadows(&mut scene_db.world);
        }

        // Quasar: the acoustic scene is read from the SceneDB world as it is now (everything the
        // renderer will draw), before the speakers' own marker rows are added.
        let audio = audio_demo::setup_audio_engine(&scene_db.world);
        let speaker_billboards: Vec<Entity> = audio
            .speakers
            .iter()
            .map(|p| {
                let e = scene_db.world.spawn();
                scene_db.world.insert(
                    e,
                    BillboardComponent {
                        world_pos: [p.x, p.y + 1.2, p.z, 1.0],
                        scale_flags: [1.0, 1.0, 0.0, 0.0],
                        color: [1.0; 4],
                    },
                );
                e
            })
            .collect();

        let mut scene_handle = scene_db_handle(&scene_db);
        let stone_store = architectural_materials::load(&device, &queue, &mut scene_db.world);
        let has_stone = stone_store.is_some();
        if let Some(store) = stone_store {
            scene_handle = scene_handle.with_texture_store(store).unwrap();
        }
        let mut renderer = RendererBuilder::new(config, scene_handle)
            .with_external_device()
            .with_editor_mode(false)
            .with_pass_build_context(Box::new(build_hlfs_graph_with_context))
            .build(device.clone(), queue.clone(), size.width, size.height, format);
        if has_stone {
            architectural_materials::configure_sampler(&mut renderer);
        }
        let mut hlfs_config = None;
        let mut acceleration = if ray_traced {
            let config = if std::env::var_os("HLFS_PRESAMPLED").is_some() {
                helio_pass_hlfs::HlfsConfig::ray_traced_presampled()
            } else {
                helio_pass_hlfs::HlfsConfig { mode: helio_pass_hlfs::HlfsMode::RayTraced, ..Default::default() }
            };
            hlfs_config = Some(config);
            Some(helio_pass_hlfs::SceneDbRayTracing::new(device.clone(), queue.clone()))
        } else {
            None
        };
        install_graph_hook(&mut renderer, queue.clone(), hlfs_config);
        if ray_traced {
            eprintln!("Interactive HLFS configuration: {:?}",
                renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>().unwrap().config());
        }
        renderer.set_ambient([0.10, 0.09, 0.085], 1.0);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        let warmup_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cathedral startup warmup"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let warmup_view = warmup_texture.create_view(&Default::default());
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let warmup_camera = large_cathedral_camera(0.0, aspect);
        if let Some(acceleration) = acceleration.as_mut() {
            v3_demo_common::flush_scene_db(&scene_db, &queue);
            acceleration.prepare(&scene_db.world).expect("cathedral RT geometry");
        }
        hlfs_capture::warm_up_cathedral(&scene_db, &mut renderer, acceleration.as_ref(),
            &device, &queue, &warmup_camera, &warmup_view);

        let renderer = Arc::new(Mutex::new(renderer));
        let (bridge, action_rx) = HelioCommandBridge::new();
        let command_bridge = Arc::new(bridge);

        // REPL thread to drive commands from stdin
        {
            let bridge = command_bridge.clone();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    match line {
                        Ok(cmd) if !cmd.trim().is_empty() => match bridge.run(&cmd) {
                            Ok(()) => println!("OK: {}", cmd),
                            Err(e) => println!("ERR: {} -> {}", cmd, e),
                        },
                        _ => {}
                    }
                }
            });
        }

        let side_unknown = !audio.watertight;
        // The listener starts at the audience point, looking down the nave toward the altar.
        let start = glam::Vec3::from_array(audio_demo::AUDIENCE);
        let worker = SpatialWorker::spawn(
            audio.engine.clone(),
            audio.listener_id,
            audio.debug_capture.clone(),
            ListenerPose { position: start.to_array(), forward: [0.0, 0.0, -1.0] },
        );

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            action_rx,
            last_frame: std::time::Instant::now(),
            scene_db,
            acceleration,
            cam_pos: start,
            cam_yaw: 0.0,
            cam_pitch: 0.065,
            keys: HashSet::new(),
            alt_pressed: false,
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            debug_mode: 0,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            debug_overlay_enabled: false,
            chandelier_light_ids,
            candle_light_ids,
            start_time: std::time::Instant::now(),
            motion_frame: 0,
            audio,
            worker,
            speaker_billboards,
            show_rays: false,
            acoustic_snapshot: None,
            acoustic_view: AcousticView { side_unknown, ..Default::default() },
            show_probes: true,
            show_scene_bounds: false,
            aux_swapped: false,
            last_stats_print: std::time::Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(false) => {
                // Some platforms swallow key-up while a system menu or another
                // window owns focus. Never keep flying on a stale movement key.
                state.keys.clear();
                state.alt_pressed = false;
                state.mouse_delta = (0.0, 0.0);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                state.alt_pressed = modifiers.state().alt_key();
                if state.alt_pressed {
                    state.keys.clear();
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }

            // F1: cycle debug modes (0=normal → 10=shadow heatmap → 11=light-space depth → 0)
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::F1), .. },
                ..
            } => {
                state.debug_mode = match state.debug_mode {
                    0 => 10,
                    10 => 11,
                    _ => 0,
                };
                if let Ok(mut renderer) = state.renderer.lock() {
                    renderer.set_debug_mode(state.debug_mode);
                }
                println!("[debug] shadow debug mode = {}", state.debug_mode);
            }

            // F2: cycle perf overlay modes
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::F2), .. },
                ..
            } => {
                state.perf_overlay_mode = match state.perf_overlay_mode {
                    PerfOverlayMode::Disabled => PerfOverlayMode::PassOverdraw,
                    PerfOverlayMode::PassOverdraw => PerfOverlayMode::ShaderComplexity,
                    PerfOverlayMode::ShaderComplexity => PerfOverlayMode::TileLightCount,
                    PerfOverlayMode::TileLightCount => PerfOverlayMode::PassOutput,
                    PerfOverlayMode::PassOutput => PerfOverlayMode::Disabled,
                };
                if let Ok(mut renderer) = state.renderer.lock() {
                    if let Some(pass) = renderer.find_pass_mut::<helio_pass_perf_overlay::PerfOverlayPass>() {
                        pass.set_mode(state.perf_overlay_mode);
                    }
                }
                println!("[debug] perf overlay mode = {:?}", state.perf_overlay_mode);
            }

            // F3: toggle debug overlay
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::F3), .. },
                ..
            } => {
                state.debug_overlay_enabled = !state.debug_overlay_enabled;
                if let Ok(mut renderer) = state.renderer.lock() {
                    if let Some(pass) = renderer.find_pass_mut::<helio_pass_debug_overlay::DebugOverlayPass>() {
                        pass.set_enabled(state.debug_overlay_enabled);
                    }
                }
                println!("[debug] debug overlay = {:?}", state.debug_overlay_enabled);
            }

            // Quasar keys (see the header). Only the movement keys below are tracked as held.
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, repeat, physical_key: PhysicalKey::Code(key), .. },
                ..
            } if state.handle_audio_key(key, repeat) => {}

            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. },
                ..
            } => match ks {
                ElementState::Pressed => {
                    if !state.alt_pressed {
                        state.keys.insert(key);
                    }
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    let ok = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                if let Ok(mut renderer) = state.renderer.lock() {
                    renderer.set_render_size(s.width, s.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl AppState {
    /// Handle a Quasar key press; `true` when the key was consumed (so it is not tracked as a
    /// held movement key). V, R, C, B, N, M ignore key repeat.
    fn handle_audio_key(&mut self, key: KeyCode, repeat: bool) -> bool {
        match key {
            // V pauses/resumes trace capture. Keep the last captured drawing visible.
            KeyCode::KeyV | KeyCode::KeyR => {
                if !repeat {
                    self.show_rays = !self.show_rays;
                    self.audio.debug_capture.set_enabled(self.show_rays);
                    self.worker.set_capture(self.show_rays);
                    let status = if self.show_rays { "capturing" } else { "paused; showing last trace" };
                    self.window.set_title(&format!("Quasar acoustic rays: {status} | V to pause/resume"));
                    println!("[quasar] acoustic capture: {status}. The last nonempty trace stays visible. Direct segment: green clear, yellow partial, red blocked; reflection paths: green/cyan/violet by order, brighter = stronger. C cycles emitter, B all emitters, N rejected candidates, M probe rays.");
                }
            }
            // C cycles the traced emitter (auto = nearest), B shows all emitters,
            // N toggles rejected reflection candidates, M toggles the solver's probe rays.
            KeyCode::KeyC | KeyCode::KeyB | KeyCode::KeyN | KeyCode::KeyM => {
                if !repeat {
                    let view = &mut self.acoustic_view;
                    match key {
                        KeyCode::KeyC => view.cycle_emitter(self.acoustic_snapshot.as_ref()),
                        KeyCode::KeyB => view.all_emitters = !view.all_emitters,
                        KeyCode::KeyN => view.show_rejected = !view.show_rejected,
                        _ => view.show_probes = !view.show_probes,
                    }
                    self.audio.debug_capture.set_detail(view.detail());
                    // Re-capture right away so newly enabled detail has data.
                    self.worker.kick();
                    if let Some(frame) = &self.acoustic_snapshot {
                        let status = if self.show_rays { "capturing" } else { "paused" };
                        self.window.set_title(&acoustic_overlay::acoustic_title(frame, &self.acoustic_view, status));
                    }
                }
            }
            // T: toggle Quasar probe grid
            KeyCode::KeyT => self.show_probes = !self.show_probes,
            // Y: toggle the acoustic scene bounds and print the material table
            KeyCode::KeyY => {
                self.show_scene_bounds = !self.show_scene_bounds;
                if self.show_scene_bounds {
                    println!("[quasar] acoustic scene: {}", self.audio.scene_stats.summary());
                    for class in acoustic_geometry::AcousticClass::ALL {
                        println!(
                            "[quasar]   {:<14} absorption {:?} transmission {:?}",
                            class.name(),
                            class.absorption(),
                            class.transmission()
                        );
                    }
                }
            }
            // 1: cycle audio DSP stage (0=silence, 1=raw, 2=+occ, 3=+early, 4=full)
            KeyCode::Digit1 => {
                if let Ok(mut engine) = self.audio.engine.lock() {
                    let stage = (engine.debug_audio_stage + 1) % 5;
                    engine.set_debug_audio_stage(stage);
                    println!(
                        "[audio] dsp stage {}: {}",
                        stage,
                        match stage {
                            0 => "silence",
                            1 => "raw pull only (reduce master vol with [!)",
                            2 => "+ occlusion",
                            3 => "+ early reflections",
                            _ => "full pipeline",
                        }
                    );
                }
            }
            // 2: print audio timing snapshot (and the spatial worker's timing)
            KeyCode::Digit2 => {
                if let Ok(engine) = self.audio.engine.lock() {
                    let t = engine.timing_snapshot();
                    let ns_per_us = 1000.0;
                    println!("[timing] calls={}  max={:.1}μs  avg={:.1}μs  block={:.0}μs  headroom={:.1}μs",
                        t.call_count,
                        t.max_ns as f64 / ns_per_us,
                        t.avg_ns as f64 / ns_per_us,
                        t.block_us,
                        t.headroom_us);
                }
                println!("[timing] {}", self.worker.stats().describe());
            }
            // 3: reset audio timing counters
            KeyCode::Digit3 => {
                if let Ok(engine) = self.audio.engine.lock() {
                    engine.timing.reset();
                    println!("[timing] counters reset");
                }
            }
            // G: live patch-bay remap — swap Aux Left/Right channel pulls.
            KeyCode::KeyG => {
                if let Ok(mut engine) = self.audio.engine.lock() {
                    let src = self.audio.source_id;
                    let o = self.audio.outputs;
                    engine.disconnect_pull(o[6], src, 6);
                    engine.connect_pull(o[6], quasar_core::scene_output::ChannelPull::new(src, 7, 0.0));
                    engine.disconnect_pull(o[7], src, 7);
                    engine.connect_pull(o[7], quasar_core::scene_output::ChannelPull::new(src, 6, 0.0));
                }
                self.aux_swapped = !self.aux_swapped;
                println!("[audio] aux channels swapped = {}", self.aux_swapped);
            }
            // [ / ]: master volume down / up (3 dB steps).
            KeyCode::BracketLeft | KeyCode::BracketRight => {
                let step = if key == KeyCode::BracketLeft { -3.0 } else { 3.0 };
                let db = (self.audio.master_gain_db() + step).clamp(-60.0, 24.0);
                self.audio.set_master_gain_db(db);
                println!("[audio] master gain = {} dB", db);
            }
            // - / =: reverb trim down / up, ; / ': early-reflection trim down / up (2 dB steps).
            KeyCode::Minus | KeyCode::Equal | KeyCode::Semicolon | KeyCode::Quote => {
                let (early, delta) = match key {
                    KeyCode::Minus => (false, -2.0),
                    KeyCode::Equal => (false, 2.0),
                    KeyCode::Semicolon => (true, -2.0),
                    _ => (true, 2.0),
                };
                if let Some(db) = self.audio.adjust_mix_trim_db(early, delta) {
                    println!("[audio] {} trim = {db} dB", if early { "early reflection" } else { "reverb" });
                }
            }
            _ => return false,
        }
        true
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 5.0;
        const SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos += glam::Vec3::Y * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= glam::Vec3::Y * SPEED * dt;
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time = self.start_time.elapsed().as_secs_f32();

        let mut camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
        );
        let mut listener_pos = self.cam_pos;
        let mut listener_forward = forward;
        if std::env::var_os("HLFS_LIVE_MOTION_TEST").is_some() {
            // Travel the same path as the Helio capture, then reverse smoothly.
            // This exercises the real swapchain and resize path while moving.
            let phase = (self.motion_frame % 600) as f32 / 300.0;
            let t = if phase <= 1.0 { phase } else { 2.0 - phase };
            camera = large_cathedral_camera(t, aspect);
            listener_pos = glam::Vec3::new(4.0 * t, 2.3, 67.0 - 29.0 * t);
            listener_forward = (glam::Vec3::new(0.0, 10.0, -68.0) - listener_pos).normalize();
            self.motion_frame = self.motion_frame.wrapping_add(1);
        }

        // The listener IS the camera: hand its pose to the spatial worker (~30 Hz updates happen
        // on that thread, never here), and pick up the newest retained debug frame.
        self.worker.set_pose(ListenerPose { position: listener_pos.to_array(), forward: listener_forward.to_array() });
        if self.show_rays {
            if let Some(frame) = self.worker.take_frame() {
                self.window.set_title(&acoustic_overlay::acoustic_title(&frame, &self.acoustic_view, "capturing"));
                self.acoustic_snapshot = Some(frame);
            }
        }
        if self.last_stats_print.elapsed().as_secs() >= 30 {
            self.last_stats_print = std::time::Instant::now();
            eprintln!("[quasar] {}", self.worker.stats().describe());
        }

        // Apply commands from REPL / quark to renderer
        let mut renderer = self.renderer.lock().unwrap();
        while let Ok(action) = self.action_rx.try_recv() {
            match action {
                HelioAction::SetDebugMode(mode) => renderer.set_debug_mode(mode),
                HelioAction::SetEditorMode(enabled) => renderer.set_editor_mode(enabled),
                HelioAction::DebugClear => renderer.debug_clear(),
            }
        }

        flicker_lights(&mut self.scene_db.world, &self.chandelier_light_ids, &self.candle_light_ids,
            time, self.acceleration.is_some());

        // Speaker billboards flash with the level the audio callback reports.
        let levels = self.audio.speaker_levels();
        for (i, &entity) in self.speaker_billboards.iter().enumerate() {
            let lvl = levels[i];
            let active = lvl > 0.005;
            let scale = if active { (1.0 + lvl * 8.0).min(3.0) } else { 1.0 };
            let mut c = acoustic_overlay::hsl_to_rgba(i as f32 / NUM_SPEAKERS as f32, 0.9, 0.6, 1.0);
            if active {
                let boost = (lvl * 6.0).min(1.0);
                for ch in &mut c[..3] {
                    *ch = *ch * (1.0 - boost) + boost;
                }
            }
            if let Some(mut b) = self.scene_db.world.get_mut::<BillboardComponent>(entity) {
                b.scale_flags = [scale, scale, 0.0, 0.0];
                b.color = c;
            }
        }

        // ── Quasar debug overlay (Helio world-space debug lines, redrawn every frame) ──
        renderer.debug_clear();
        acoustic_overlay::draw_stage(
            &mut renderer,
            &self.audio.speakers,
            glam::Vec3::from_array(audio_demo::AUDIENCE),
            listener_pos,
            listener_forward,
        );
        if self.show_probes {
            let g = &self.audio.probe_grid;
            acoustic_overlay::draw_probe_grid(&mut renderer, g.origin, g.spacing, g.dims, 12.0);
        }
        if self.show_scene_bounds {
            let s = &self.audio.scene_stats;
            acoustic_overlay::draw_aabb(&mut renderer, s.aabb_min, s.aabb_max, [1.0, 0.8, 0.2, 0.8]);
        }
        if let Some(snapshot) = &self.acoustic_snapshot {
            acoustic_overlay::draw_acoustic_snapshot(&mut renderer, snapshot, &self.acoustic_view);
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        v3_demo_common::flush_scene_db(&self.scene_db, &self.queue);
        if let Some(acceleration) = &self.acceleration {
            renderer.set_ray_tracing_frame_with_transmission(acceleration.tlas(), acceleration.transmission());
        }
        if let Err(e) = renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        if self.acceleration.is_some() {
            static CONFIG_LOGGED: std::sync::Once = std::sync::Once::new();
            CONFIG_LOGGED.call_once(|| eprintln!("HLFS after first live render: {:?}",
                renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>().unwrap().config()));
        }
        self.queue.present(output);
    }
}

// ── Scene population (Helio's indoor_cathedral_hlfs, large mode) ────────────────

/// Afternoon sun, travelling +x (in through the left windows), down and toward the
/// entrance, so the view up the nave looks into the light where forward-
/// scattering smoke is brightest. 46° elevation lands the lancet patterns on the nave floor
/// through the large nave's clerestory.
const SUN_DIRECTION: [f32; 3] = [0.58, -0.72, 0.38];

/// Lets an interior light glow in the incense. These lights have no shadow
/// maps, so their scattering is unshadowed (decay 0): a soft halo, no shafts.
fn haze_light(mut light: helio::GpuLight) -> helio::GpuLight {
    if std::env::var_os("HLFS_NO_HAZE_LIGHTS").is_some() { return light; }
    light.god_rays_enabled = 1;
    light.god_rays_density = 1.0;
    light.god_rays_weight = 1.0;
    light.god_rays_exposure = 1.0;
    light.god_rays_decay = 0.0;
    light
}

/// Incense haze filling the interior: a local medium bounded by the walls,
/// lit by the sun through the stained glass. Physical fog only scatters light
/// that reaches it, so the shafts take the panes' colours and outlines.
/// `HLFS_NO_FOG=1` removes it; `HLFS_FOG_DENSITY=<m⁻¹>` overrides extinction.
fn configure_cathedral_fog(world: &mut World) {
    if std::env::var_os("HLFS_NO_FOG").is_some() { return; }
    let (half_x, height, half_z, extinction, range) = (22.0, 46.0, 71.5, 0.018, 170.0);
    let extinction = std::env::var("HLFS_FOG_DENSITY").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(extinction);
    v3_demo_common::spawn_local_fog(
        world,
        [-half_x, 0.0, -half_z],
        [half_x, height, half_z],
        v3_demo_common::GlobalFogComponent {
            // Drifting smoke under a height envelope: incense, not a flat haze.
            mode: std::env::var("HLFS_FOG_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            extinction,
            albedo: [0.92, 0.90, 0.86],
            // Forward-peaked, as smoke is: shafts brighten looking toward the sun.
            anisotropy: 0.6,
            // Denser low down, thinning toward the vault.
            height: 0.0,
            height_falloff: 0.03,
            ..Default::default()
        },
        1.0,
    );
    let settings = v3_demo_common::set_volumetric_quality(world, 1, range);
    if let Some(blend) = std::env::var("HLFS_FOG_BLEND").ok().and_then(|v| v.parse().ok()) {
        world.insert(settings, v3_demo_common::VolumetricFogSettingsComponent {
            quality: 1, max_distance: range, light_max_distance: range, temporal_blend: blend,
            ..Default::default()
        });
    }
}

/// Helio's scripted camera path through the large hall (used for the warm-up frames and by
/// `HLFS_LIVE_MOTION_TEST`).
fn large_cathedral_camera(t: f32, aspect: f32) -> Camera {
    Camera::perspective_look_at(
        glam::Vec3::new(4.0 * t, 2.3, 67.0 - 29.0 * t),
        glam::Vec3::new(0.0, 10.0, -68.0), glam::Vec3::Y,
        std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0,
    )
}

pub(crate) fn populate_large_cathedral(world: &mut World) -> (Vec<Entity>, Vec<Entity>) {
    configure_cathedral_fog(world);
    spawn_indoor_cathedral_sky(world);
    cathedral_large::populate(world);
    populate_cathedral_lights(world)
}

fn populate_cathedral_lights(world: &mut World) -> (Vec<Entity>, Vec<Entity>) {
    // Register lights (chandelier & candle light_ids stored for per-frame flicker updates)
    let mut chandelier_light_ids = Vec::new();
    for &z in LARGE_CHANDELIER_Z {
        chandelier_light_ids.push(spawn_light(
            world,
            haze_light(point_light([0.0_f32, 31.0, z], [1.0, 0.92, 0.78], 160.0, 22.0)),
        ));
    }
    // A single exterior sun supplies a coherent daylight direction. It owns the
    // first shadow slot and participates in the medium, so the glass colours
    // both the floor pattern and the shafts (raster: the shadow transmittance
    // layer; RT: thin-sheet transmission). Keep the older multi-window
    // emitter setup as an explicit transmission stress case.
    if std::env::var_os("HLFS_LEGACY_CATHEDRAL_LIGHTS").is_none() {
        let intensity = std::env::var("HLFS_SUN").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(20.0);
        spawn_light(world, v3_demo_common::volumetric_light(
            v3_demo_common::directional_light(SUN_DIRECTION, [1.0, 0.94, 0.84], intensity),
            v3_demo_common::SHADOW_BASES[0],
        ));
    } else {
        // Stained glass shafts — static, no need to store ids
        // RT uses white exterior sources: pane materials supply the transmitted tint.
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            let (x, y, z) = (x.signum() * 21.5, y * 1.8, z * 2.4);
            let light = if std::env::var_os("HLFS_RT").is_some() {
                let position = if x == 0.0 { [0.0, 34.0, 78.0] } else { [x.signum() * 29.0, 24.0, z] };
                point_light(position, [1.0; 3], 2500.0, 65.0)
            } else {
                point_light([x, y, z], [r, g, b], 35.0, 10.0)
            };
            spawn_light(world, light);
        }
    }
    let mut candle_light_ids = Vec::new();
    for &(x, y, z) in LARGE_CANDLES {
        candle_light_ids.push(spawn_light(
            world,
            haze_light(point_light([x, y, z], [1.0, 0.6, 0.15], 8.0, 4.0)),
        ));
    }

    (chandelier_light_ids, candle_light_ids)
}

/// Chandeliers flicker slightly and candles more.
fn flicker_lights(
    world: &mut World,
    chandeliers: &[Entity],
    candles: &[Entity],
    time: f32,
    ray_traced: bool,
) {
    let flicker = 1.0 + (time * 9.1).sin() * 0.03 + (time * 5.7).cos() * 0.02;
    let cflicker = 1.0 + (time * 14.3).sin() * 0.07 + (time * 8.9).cos() * 0.05;
    let with_shadows = |mut light: helio::GpuLight| {
        light.set_ray_traced_shadows(ray_traced);
        light
    };
    for (&id, &z) in chandeliers.iter().zip(LARGE_CHANDELIER_Z) {
        update_light(world, id, with_shadows(haze_light(point_light(
            [0.0_f32, 31.0, z], [1.0, 0.92, 0.78], 160.0 * flicker, 22.0))));
    }
    for (&id, &(x, y, z)) in candles.iter().zip(LARGE_CANDLES) {
        update_light(world, id, with_shadows(haze_light(point_light(
            [x, y, z], [1.0, 0.6, 0.15], 8.0 * cflicker, 4.0))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_icon_has_correct_dimensions() {
        let (pixels, w, h) = generate_speaker_icon();
        assert_eq!((w, h), (32, 32));
        assert_eq!(pixels.len(), 32 * 32 * 4);
    }

    #[test]
    fn speaker_icon_has_non_transparent_pixels_on_a_transparent_background() {
        let (pixels, _w, _h) = generate_speaker_icon();
        let opaque = pixels.chunks_exact(4).filter(|c| c[3] == 255).count();
        assert!(opaque > 0, "speaker icon must contain non-transparent pixels");
        assert!(opaque < pixels.len() / 4, "speaker icon should have transparent background");
    }

    #[test]
    fn speaker_icon_white_pixels() {
        let (pixels, _w, _h) = generate_speaker_icon();
        for chunk in pixels.chunks_exact(4).filter(|c| c[3] == 255) {
            // Opaque pixels must be fully white (the shader tints them).
            assert_eq!(&chunk[..3], &[255, 255, 255]);
        }
    }
}
