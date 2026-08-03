//! Indoor cathedral example – high complexity
//!
//! A large Gothic cathedral interior: a 60 m nave flanked by two side aisles,
//! 12 stone columns, a raised altar platform with a cross, carved stone pews
//! in 6 rows on each side, three ornate chandeliers, stained-glass window
//! shafts casting coloured light at intervals along both walls, and candle
//! clusters near the altar.
//!
//! No sky atmosphere — the scene relies entirely on the interplay of the
//! chandelier warm-white lights, the cool-coloured stained-glass fills, and
//! a very dim stone-cold ambient to create a moody sacred atmosphere. The
//! radiance cascades GI system bounces chandelier light deep into the side
//! aisles and onto the vaulted ceiling.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   R           — toggle Quasar ray visualization
//!   T           — toggle Quasar probe grid overlay
//!   Y           — toggle Quasar material zone colors
//!   F2          — toggle performance overlay modes (GPU heatmaps)
//!   F3          — toggle debug overlay (FPS, timings, texture stats)
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, BakeConfig, Camera, DebugDrawState, HelioAction, HelioCommandBridge, LightId, MeshId, Movability, Renderer, RendererConfig, Scene,
};
// (BillboardInstance referenced inline as helio::BillboardInstance)
use helio_pass_perf_overlay::PerfOverlayMode;
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, make_material, plane_mesh, point_light};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use quasar_audio::SpatialAudioEngine;
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::scene::{AcousticMesh as QMesh, AcousticScene as QScene};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;
use quasar_dsp::node_graph::AudioNode;

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

// ── Scene data ────────────────────────────────────────────────────────────────

// Column positions along the nave (Z axis), symmetric at x = ±5.5
const COLUMN_Z: &[f32] = &[-22.0, -14.0, -6.0, 2.0, 10.0, 18.0];

// Stained glass window lights: (x_wall_side, y, z, r, g, b)
// Positive x = right-side windows, negative = left-side; placed just inside the wall
const GLASS_LIGHTS: &[(f32, f32, f32, f32, f32, f32)] = &[
    // Left wall (x ≈ -10.5), windows between columns
    (-10.3, 9.0, -18.0, 0.8, 0.2, 1.0), // violet
    (-10.3, 9.0, -6.0, 0.2, 0.7, 1.0),  // sky blue
    (-10.3, 9.0, 6.0, 0.2, 1.0, 0.4),   // emerald
    (-10.3, 9.0, 18.0, 1.0, 0.7, 0.1),  // gold
    // Right wall (x ≈ +10.5)
    (10.3, 9.0, -18.0, 1.0, 0.2, 0.3), // ruby
    (10.3, 9.0, -6.0, 1.0, 0.5, 0.1),  // amber
    (10.3, 9.0, 6.0, 0.1, 0.8, 0.9),   // teal
    (10.3, 9.0, 18.0, 0.9, 0.1, 0.7),  // magenta
    // Rose window above entrance (back wall, z ≈ +28)
    (0.0, 13.0, 27.0, 1.0, 0.75, 0.3), // warm gold
];

// Chandelier positions (x=0, hanging from y≈19.5, at z intervals)
const CHANDELIER_Z: &[f32] = &[-16.0, 0.0, 16.0];

// Candle cluster positions near the altar (z ≈ -24)
const CANDLES: &[(f32, f32, f32)] = &[
    (-3.0, 1.6, -23.5),
    (-1.5, 1.6, -23.0),
    (0.0, 1.6, -23.5),
    (1.5, 1.6, -23.0),
    (3.0, 1.6, -23.5),
];

// Pew rows: 6 per side, spaced 2.4 m apart starting at z = -20
const PEW_Z_START: f32 = -20.0;
const PEW_Z_STEP: f32 = 3.2;
const PEW_COUNT: usize = 6;

// ── Quasar spatial audio engine (playback + spatial) ──────────────────────

const NUM_SOURCES: usize = 4;

struct WavPlayback {
    samples: Vec<f32>,
    num_channels: usize,
    read_pos: f64,
    rate_ratio: f64,
    occ_nodes: Vec<AirAbsorptionOcclusionNode>,
    reverb: FdnReverbNode,
}

struct AudioEngine {
    engine: Arc<Mutex<SpatialAudioEngine>>,
    _state: Arc<Mutex<WavPlayback>>,
    _stream: cpal::Stream,
}

fn zero_coeffs() -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0, direct_gain: Band8::splat(0.0), direct_delay_samples: 0.0,
        early_reflections: Vec::new(), late_t60: Band8::splat(0.0), late_gain_db: 0.0, version: 0,
    }
}

fn setup_audio_engine() -> AudioEngine {
    // Load WAV via hound
    let reader = hound::WavReader::open("assets/8_Channel_ID.wav").expect("open WAV");
    let spec = reader.spec();
    let wav_sr = spec.sample_rate;
    let nch_wav = spec.channels as usize;
    let samples: Vec<f32> = match spec.bits_per_sample {
        16 => reader.into_samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32).collect(),
        24 => reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / 8_388_608.0).collect(),
        32 => reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / i32::MAX as f32).collect(),
        b => panic!("unsupported bit depth: {b}"),
    };

    // Set up cpal
    let host = cpal::default_host();
    let device = host.default_output_device().expect("audio output device");
    let out_config = device.default_output_config().expect("output config");
    let out_sr = out_config.sample_rate().0;
    let out_ch = out_config.channels() as usize;
    let sr = out_sr as f32;

    // SpatialAudioEngine for coefficient computation (not used for audio graph)
    let mut engine = SpatialAudioEngine::new(NUM_SOURCES, sr, 15.0);
    {
        let mut qs = QScene::new();
        qs.add_mesh(QMesh::new(1, vec![[-11.,0.,-28.],[11.,0.,-28.],[11.,0.,28.],[-11.,0.,28.]], vec![0,1,2,0,2,3], 0));
        qs.add_mesh(QMesh::new(2, vec![[-11.,0.,-28.],[-11.,0.,28.],[-11.,21.,28.],[-11.,21.,-28.]], vec![0,1,2,0,2,3], 1));
        qs.add_mesh(QMesh::new(3, vec![[11.,0.,-28.],[11.,0.,28.],[11.,21.,28.],[11.,21.,-28.]], vec![0,1,2,0,2,3], 1));
        let cfg = CpuSimdConfig {
            max_reflection_order: 3, diffuse_rays_per_query: 128, max_reflection_distance: 60.,
            speed_of_sound: 343., temperature_celsius: 20., humidity_percent: 50.,
        };
        engine.set_backend(Box::new(CpuSimdComputeBackend::new(qs, cfg)));
    }
    let engine = Arc::new(Mutex::new(engine));
    let rate_ratio = wav_sr as f64 / out_sr as f64;

    let occ_nodes = (0..NUM_SOURCES)
        .map(|_| AirAbsorptionOcclusionNode::new(1, sr, 0.1))
        .collect();

    let state = Arc::new(Mutex::new(WavPlayback {
        samples, num_channels: nch_wav, read_pos: 0.0, rate_ratio, occ_nodes,
        reverb: FdnReverbNode::new(2, sr),
    }));

    let state_cb = state.clone();
    let err_fn = |e: cpal::StreamError| eprintln!("Audio error: {e}");

    let stream = device.build_output_stream(
        &out_config.config(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let total_frames = data.len() / out_ch;
            data.fill(0.0);
            if total_frames == 0 { return; }

            let mut st = state_cb.lock().unwrap();
            let total_raw = st.samples.len();
            let nch = st.num_channels;
            let ratio = st.rate_ratio;
            let mut remain = total_frames;
            let mut offset = 0;

            while remain > 0 {
                let block = (DEFAULT_BLOCK_SIZE as usize).min(remain);

                // 4-source stereo mix via occlusion node
                let mut l_acc = vec![0.0f32; block];
                let mut r_acc = vec![0.0f32; block];

                for src in 0..NUM_SOURCES.min(nch) {
                    let mut mono_in = AudioBuffer::new(1, block as u16);
                    {
                        let ch = mono_in.channel_mut(0);
                        for i in 0..block {
                            let pos = st.read_pos + i as f64 * ratio;
                            let fa = pos.floor() as usize;
                            let fb = fa + 1;
                            let frac = (pos - fa as f64) as f32;
                            let g = |f: usize| -> f32 {
                                if f < total_raw / nch { st.samples[f * nch + src] } else { 0.0 }
                            };
                            ch[i] = g(fa) + (g(fb) - g(fa)) * frac;
                        }
                    }

                    let mut occ_out = AudioBuffer::new(1, block as u16);
                    let zero = SpatialCoefficients {
                        source_id: 0, direct_gain: Band8::splat(0.0), direct_delay_samples: 0.0,
                        early_reflections: Vec::new(), late_t60: Band8::splat(0.0),
                        late_gain_db: 0.0, version: 0,
                    };
                    st.occ_nodes[src].process(&mono_in, &mut occ_out, &zero);

                    let pan = (src as f32 / (NUM_SOURCES - 1).max(1) as f32) * 2.0 - 1.0;
                    let t = (pan + 1.0) * 0.5;
                    let angle = std::f32::consts::FRAC_PI_2 * t;
                    let (lg, rg) = (angle.cos(), angle.sin());

                    for i in 0..block {
                        let s = occ_out.channel(0)[i];
                        l_acc[i] += s * lg;
                        r_acc[i] += s * rg;
                    }
                }
                st.read_pos += block as f64 * ratio;

                // Feed summed stereo through shared reverb
                let mut rev_out = AudioBuffer::new(2, block as u16);
                let stereo_buf = AudioBuffer::from_channels(&[&l_acc, &r_acc]);
                let zero = SpatialCoefficients {
                    source_id: 0, direct_gain: Band8::splat(0.0), direct_delay_samples: 0.0,
                    early_reflections: Vec::new(), late_t60: Band8::splat(0.0),
                    late_gain_db: 0.0, version: 0,
                };
                st.reverb.process(&stereo_buf, &mut rev_out, &zero);

                for i in 0..block {
                    data[(offset + i) * out_ch] = rev_out.channel(0)[i];
                    if out_ch > 1 { data[(offset + i) * out_ch + 1] = rev_out.channel(1)[i]; }
                }

                if st.read_pos >= (total_raw / nch) as f64 { st.read_pos = 0.0; }
                remain -= block;
                offset += block;
            }
        },
        err_fn, None,
    ).expect("build output stream");
    stream.play().expect("play stream");

    AudioEngine { engine, _state: state, _stream: stream }
}

// ── Billboard sprite replacement (Helio issue #192 workaround) ─────────────

/// Replace the default billboard sprite (spotlight.png) with the procedural
/// speaker icon in the render graph.
///
/// Must be called whenever the graph may have been rebuilt (see Helio #192:
/// `rebuild_graph_if_sky_changed()` and `apply_resize_now()` both invoke the
/// `GraphRebuilder`, which destroys custom pass replacements).  We re-apply
/// the replacement on every frame after acquiring the render lock so that
/// any graph rebuild is always caught.
///
/// Uses the *scene* camera buffer (`array<Camera,2>`, 736 bytes) — NOT the
/// `debug_camera_buf` (`DebugCameraUniform`, 64 bytes) — because the billboard
/// shader reads `cameras[0].view_proj` at offset 128 and
/// `cameras[0].position_near` at offset 256, both past a 64-byte buffer.
fn apply_billboard_replacement(
    renderer: &mut Renderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let camera_buf = renderer.scene().gpu_scene().camera.buffer();
    let fmt = renderer.renderer_config().surface_format;
    let (rgba, w, h) = generate_speaker_icon();
    if let Some(idx) = renderer.graph_pass_index::<helio_pass_billboard::BillboardPass>() {
        let custom_pass = helio_pass_billboard::BillboardPass::new_with_sprite_rgba(
            device, queue, camera_buf, fmt, &rgba, w, h,
        );
        renderer.replace_graph_pass(idx, Box::new(custom_pass));
    }
}

/// Generate a simple 16x16 white speaker icon as RGBA pixel data.
fn generate_speaker_icon() -> (Vec<u8>, u32, u32) {
    let w = 32u32; let h = 32u32;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h { for x in 0..w {
        let cx = x as i32 - 16; let cy = y as i32 - 16;
        let in_cabinet = cx >= -8 && cx <= -3 && cy >= -8 && cy <= 8;
        let in_cone = cx >= -2 && cx <= 8 && cy.abs() <= (10 - cx);
        let in_grill = cx == -3 && cy >= -6 && cy <= 6 && cy % 3 == 0;
        let lit = in_cabinet || in_cone || in_grill;
        if lit { let idx = ((y * w + x) * 4) as usize;
            pixels[idx]=255; pixels[idx+1]=255; pixels[idx+2]=255; pixels[idx+3]=255;
        }
    }}
    (pixels, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_icon_has_correct_dimensions() {
        let (pixels, w, h) = generate_speaker_icon();
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(pixels.len(), 32 * 32 * 4);
    }

    #[test]
    fn speaker_icon_has_non_transparent_pixels() {
        let (pixels, _w, _h) = generate_speaker_icon();
        let opaque = pixels.chunks_exact(4).filter(|c| c[3] == 255).count();
        // Should have many white pixels (the speaker shape), not all transparent
        assert!(opaque > 0, "speaker icon must contain non-transparent pixels");
        assert!(opaque < pixels.len() / 4, "speaker icon should have transparent background");
    }

    #[test]
    fn speaker_icon_white_pixels() {
        let (pixels, _w, _h) = generate_speaker_icon();
        for chunk in pixels.chunks_exact(4) {
            if chunk[3] == 255 {
                // Opaque pixels must be fully white (the shader tints them)
                assert_eq!(chunk[0], 255);
                assert_eq!(chunk[1], 255);
                assert_eq!(chunk[2], 255);
            }
        }
    }
}

fn hsl_to_rgba(h: f32, s: f32, l: f32, a: f32) -> [f32; 4] {
    let c = (1.0-(2.0*l-1.0).abs())*s; let x = c*(1.0-((h*6.0)%2.0-1.0).abs()); let m = l-c*0.5;
    let (r,g,b) = match (h*6.0).floor() as i32 { 0=>(c,x,0.),1=>(x,c,0.),2=>(0.,c,x),3=>(0.,x,c),4=>(x,0.,c),_=>(c,0.,x) };
    [r+m, g+m, b+m, a]
}

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
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

    // Major structural surfaces
    _floor: MeshId,
    _nave_ceiling: MeshId,
    _aisle_ceil_l: MeshId,
    _aisle_ceil_r: MeshId,
    _wall_left_outer: MeshId,
    _wall_right_outer: MeshId,
    _wall_front: MeshId,
    _wall_back: MeshId,
    // Colonnade arches (inner walls between nave and aisles, with gaps left for columns)
    _colonnade_l: Vec<MeshId>, // wall segments between columns
    _colonnade_r: Vec<MeshId>,
    // Columns
    _columns: Vec<MeshId>,
    // Altar
    _altar_plinth: MeshId,
    _altar_step: MeshId,
    _cross_vert: MeshId,
    _cross_horiz: MeshId,
    // Pews
    _pews_left: Vec<MeshId>,
    _pews_right: Vec<MeshId>,
    // Chandelier bodies (chain + ring)
    _chandelier_chains: Vec<MeshId>,
    _chandelier_rings: Vec<MeshId>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Debug
    debug_mode: u32,
    perf_overlay_mode: PerfOverlayMode,
    debug_overlay_enabled: bool,

    // Scene state
    chandelier_light_ids: Vec<LightId>,
    candle_light_ids: Vec<LightId>,
    start_time: std::time::Instant,

    // Quasar spatial audio
    _audio_engine: AudioEngine,
    show_rays: bool,
    show_probes: bool,
    show_material_zones: bool,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
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
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
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
                .with_shadow_quality(helio::ShadowQuality::Ultra);
        let mut scene = Scene::new(device.clone(), queue.clone());

        // Sky MUST be added to scene BEFORE build_default_graph / Renderer::new,
        // otherwise the first render() call triggers rebuild_graph_if_sky_changed()
        // which calls the GraphRebuilder and destroys any pass replacements made
        // after construction (see Helio issue #192).
        scene.insert_actor(helio::SceneActor::Sky(
            helio::SkyActor::indoor([0.05, 0.05, 0.1]).with_clouds(helio::VolumetricClouds {
                coverage: 0.7,
                density: 0.8,
                base: 1200.0,
                top: 1800.0,
                wind_x: 0.8,
                wind_z: 0.2,
                speed: 1.3,
                skylight_intensity: 0.25,
            }),
        ));
        let debug_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let graph = build_default_graph(&device, &queue, &scene, config, debug_state.clone(), &debug_camera_buf, &cull_stats_buf, None);
        let mut renderer = Renderer::new(
            device.clone(), queue.clone(),
            config.surface_format, config.width, config.height, config.render_scale,
            config, scene, graph, debug_state, debug_camera_buf.clone(), cull_stats_buf,
        );
        renderer.set_editor_mode(true);

        let mat = renderer.scene_mut().insert_material(make_material(
            [0.75, 0.72, 0.68, 1.0],
            0.85,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        // Sky was added to the Scene directly before Renderer creation above.
        // This avoids Helio issue #192: graph rebuild on sky-change that would
        // destroy any custom pass replacements made after construction.

        // Nave + aisles: total width = 22m (x: -11..+11), length = 60m (z: -28..+28), height = 21m
        // Expand floor to cover full cathedral footprint. 32m radius = 64m square.
        let _floor =            renderer.scene_mut().insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 32.0))).as_mesh().unwrap();
        let _wall_back =        renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [11.0, 10.5, 0.25]))).as_mesh().unwrap();
        let _wall_front =       renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [11.0, 10.5, 0.25]))).as_mesh().unwrap();
        let _aisle_ceil_l =     renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [2.5, 0.15, 28.0]))).as_mesh().unwrap();
        let _nave_ceiling =     renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [6.0, 0.18, 28.0]))).as_mesh().unwrap();
        let _aisle_ceil_r =     renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [2.5, 0.15, 28.0]))).as_mesh().unwrap();
        let _wall_left_outer =  renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 7.0, 28.0]))).as_mesh().unwrap();
        let _wall_right_outer = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 7.0, 28.0]))).as_mesh().unwrap();
        let _ =
            v3_demo_common::insert_object(&mut renderer, _floor, mat, glam::Mat4::IDENTITY, 11.0);
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _nave_ceiling,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 21.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _aisle_ceil_l,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(-8.5, 11.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _aisle_ceil_r,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(8.5, 11.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_left_outer,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(-11.0, 7.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_right_outer,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(11.0, 7.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_front,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 10.5, 28.0)),
            11.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_back,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 10.5, -28.0)),
            11.0,
        );

        // Colonnade: short wall segments between columns (between column z-positions)
        // 7 segments per side: before first col, between each pair, after last col
        let col_z_all: Vec<f32> = {
            let mut v = vec![-28.0_f32]; // south wall
            v.extend_from_slice(COLUMN_Z);
            v.push(28.0); // north wall
            v
        };
        let _colonnade_l: Vec<MeshId> = col_z_all
            .windows(2)
            .map(|w| {
                let mid_z = (w[0] + w[1]) * 0.5;
                let half_len = (w[1] - w[0]) * 0.5 - 0.9; // gap for column
                let id = renderer.scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 5.5, half_len.max(0.1)])))
                    .as_mesh()
                    .unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-5.5, 5.5, mid_z)),
                    5.5,
                );
                id
            })
            .collect();
        let _colonnade_r: Vec<MeshId> = col_z_all
            .windows(2)
            .map(|w| {
                let mid_z = (w[0] + w[1]) * 0.5;
                let half_len = (w[1] - w[0]) * 0.5 - 0.9;
                let id = renderer.scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 5.5, half_len.max(0.1)])))
                    .as_mesh()
                    .unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(5.5, 5.5, mid_z)),
                    5.5,
                );
                id
            })
            .collect();

        // Columns: 0.65 m square, 20 m tall, at x = ±5.5
        let _columns: Vec<MeshId> = COLUMN_Z
            .iter()
            .flat_map(|&z| {
                let l = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.65, 10.0, 0.65]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    l,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-5.5, 10.0, z)),
                    10.0,
                );
                let r = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.65, 10.0, 0.65]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    r,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(5.5, 10.0, z)),
                    10.0,
                );
                [l, r]
            })
            .collect();

        // Altar: at far end (z = -26)
        let _altar_step = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [5.5, 0.20, 3.0]))).as_mesh().unwrap();
        let _altar_plinth = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [3.0, 0.45, 1.5]))).as_mesh().unwrap();
        let _cross_vert = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.18, 2.2, 0.18]))).as_mesh().unwrap();
        let _cross_horiz = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.0, 0.18, 0.18]))).as_mesh().unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _altar_step,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.2, -24.5)),
            5.5,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _altar_plinth,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.65, -25.5)),
            3.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _cross_vert,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 3.2, -25.8)),
            2.2,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _cross_horiz,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 4.5, -25.8)),
            1.0,
        );

        // Pews: long narrow rect3d per row, 6 rows each side
        let _pews_left: Vec<MeshId> = (0..PEW_COUNT)
            .map(|i| {
                let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.5, 0.45, 0.5]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-3.2, 0.45, z)),
                    1.5,
                );
                id
            })
            .collect();
        let _pews_right: Vec<MeshId> = (0..PEW_COUNT)
            .map(|i| {
                let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.5, 0.45, 0.5]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(3.2, 0.45, z)),
                    1.5,
                );
                id
            })
            .collect();

        // Chandeliers: vertical chain + horizontal ring at each Z
        let chandelier_mat = renderer.scene_mut().insert_material(make_material(
            [0.3, 0.28, 0.25, 1.0],
            0.5,
            0.8,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let _chandelier_chains: Vec<MeshId> = CHANDELIER_Z
            .iter()
            .map(|&z| {
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.06, 2.0, 0.06]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    chandelier_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 17.5, z)),
                    2.0,
                );
                id
            })
            .collect();
        let _chandelier_rings: Vec<MeshId> = CHANDELIER_Z
            .iter()
            .map(|&z| {
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.2, 0.12, 1.2]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    chandelier_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 15.2, z)),
                    1.2,
                );
                id
            })
            .collect();

        // Register lights (chandelier & candle light_ids stored for per-frame flicker updates)
        let mut chandelier_light_ids = Vec::new();
        for &z in CHANDELIER_Z {
            chandelier_light_ids.push(renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
                [0.0_f32, 15.0, z],
                [1.0, 0.92, 0.78],
                8.0,
                22.0,
            ))).as_light().unwrap());
        }
        // Stained glass shafts — Stationary: they never animate, so they're excluded
        // from the real-time deferred-light loop once baked lighting is loaded.
        // Without this they were running full tiled PCF every frame despite being "baked".
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light_with_movability(
                point_light([x, y, z], [r, g, b], 1.8, 8.0),
                Some(Movability::Stationary),
            ));
        }
        let mut candle_light_ids = Vec::new();
        for &(x, y, z) in CANDLES {
            candle_light_ids.push(renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
                [x, y, z],
                [1.0, 0.6, 0.15],
                1.2,
                4.0,
            ))).as_light().unwrap());
        }
        renderer.set_ambient([0.65, 0.7, 0.85], 0.015);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        // Bake static/stationary lights so they're excluded from the real-time
        // deferred-light loop. Without this, all 9 glass window lights + environment
        // run full tiled PCF every frame even though they're fixed.
        renderer.auto_bake(BakeConfig::fast("indoor_cathedral"));

        // Replace default billboard sprite (spotlight.png) with speaker icon.
        // Also re-applied every frame (see render()) to survive graph rebuilds
        // triggered by resize or sky-change (Helio issue #192).
        apply_billboard_replacement(&mut renderer, &device, &queue);

        let audio_engine = setup_audio_engine();

        // Draw initial probe grid
        {
            for x in -2..=2 { for z in -2..=2 {
                let hue = (((x + 2) * 5 + (z + 2)) as f32 / 25.0) * 0.7;
                renderer.debug_sphere([x as f32 * 2.0, 0.3, z as f32 * 2.0], 0.08, hsl_to_rgba(hue, 0.8, 0.6, 1.0), 8);
            }}
        }

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
                        Ok(cmd) if !cmd.trim().is_empty() => {
                            match bridge.run(&cmd) {
                                Ok(()) => println!("OK: {}", cmd),
                                Err(e) => println!("ERR: {} -> {}", cmd, e),
                            }
                        }
                        _ => {}
                    }
                }
            });
        }

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            action_rx,
            last_frame: std::time::Instant::now(),
            _floor,
            _nave_ceiling,
            _aisle_ceil_l,
            _aisle_ceil_r,
            _wall_left_outer,
            _wall_right_outer,
            _wall_front,
            _wall_back,
            _colonnade_l,
            _colonnade_r,
            _columns,
            _altar_plinth,
            _altar_step,
            _cross_vert,
            _cross_horiz,
            _pews_left,
            _pews_right,
            _chandelier_chains,
            _chandelier_rings,
            // Start at entrance, looking toward the altar
            cam_pos: glam::Vec3::new(0.0, 2.0, 24.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: -0.05,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            debug_mode: 0,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            debug_overlay_enabled: false,
            chandelier_light_ids,
            candle_light_ids,
            start_time: std::time::Instant::now(),
            _audio_engine: audio_engine,
            show_rays: true,
            show_probes: true,
            show_material_zones: true,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
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

            // R: toggle Quasar ray visualization
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyR), .. },
                ..
            } => { state.show_rays = !state.show_rays; },
            // T: toggle Quasar probe grid
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyT), .. },
                ..
            } => { state.show_probes = !state.show_probes; },
            // Y: toggle Quasar material zones
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::KeyY), .. },
                ..
            } => { state.show_material_zones = !state.show_material_zones; },
            

            // F1: cycle debug modes (0=normal → 10=shadow heatmap → 11=light-space depth → 0)
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F1),
                        ..
                    },
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

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F2),
                        ..
                    },
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

            // F3: toggle debug overlay (FPS, timings, texture stats)
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F3),
                        ..
                    },
                ..
            } => {
                state.debug_overlay_enabled = !state.debug_overlay_enabled;
                if let Ok(mut renderer) = state.renderer.lock() {
                    if let Some(pass) = renderer.find_pass_mut::<helio_pass_debug_overlay::DebugOverlayPass>() {
                        pass.set_enabled(state.debug_overlay_enabled);
                    }
                }
                println!("[debug] debug overlay = {}", state.debug_overlay_enabled);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
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

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
        );

        // Apply commands from REPL / quark to renderer
        let mut renderer = self.renderer.lock().unwrap();
        while let Ok(action) = self.action_rx.try_recv() {
            match action {
                HelioAction::SetDebugMode(mode) => renderer.set_debug_mode(mode),
                HelioAction::SetEditorMode(enabled) => renderer.set_editor_mode(enabled),
                HelioAction::DebugClear => renderer.debug_clear(),
            }
        }

        // Re-apply billboard sprite replacement every frame to survive graph
        // rebuilds triggered by resize or sky-change (Helio issue #192).
        apply_billboard_replacement(&mut renderer, &*self.device, &*self.queue);

        // Chandeliers flicker slightly
        let flicker = 1.0 + (time * 9.1).sin() * 0.03 + (time * 5.7).cos() * 0.02;
        // Candle flicker — more pronounced
        let cflicker = 1.0 + (time * 14.3).sin() * 0.07 + (time * 8.9).cos() * 0.05;

        // Update flickering chandelier intensities
        for (i, &id) in self.chandelier_light_ids.iter().enumerate() {
            let z = CHANDELIER_Z[i];
            let _ = renderer.scene_mut().update_light(
                id,
                point_light([0.0_f32, 15.0, z], [1.0, 0.92, 0.78], 8.0 * flicker, 22.0),
            );
        }
        // Update flickering candle intensities
        for (i, &id) in self.candle_light_ids.iter().enumerate() {
            let (x, y, z) = CANDLES[i];
            let _ = renderer.scene_mut().update_light(
                id,
                point_light([x, y, z], [1.0, 0.6, 0.15], 1.2 * cflicker, 4.0),
            );
        }

        // ── Quasar spatial audio debug overlay ─────────────────────────
        let listener_pos = self.cam_pos;
        let source_positions = [
            glam::Vec3::new(-3.0, 1.2, -2.0), glam::Vec3::new(3.0, 0.8, 2.0),
            glam::Vec3::new(-1.0, 2.5, 3.0), glam::Vec3::new(4.0, 0.3, -3.0),
        ];

        // Update Quasar spatial audio with current positions
        if let Ok(engine) = self._audio_engine.engine.lock() {
            for (i, &pos) in source_positions.iter().enumerate().take(NUM_SOURCES) {
                engine.update_spatial(&quasar_core::backend::SpatialQuery {
                    source_position: pos.to_array(),
                    listener_position: listener_pos.to_array(),
                    source_id: i as u32,
                });
            }
        }

        renderer.debug_clear();
        for (i, &pos) in source_positions.iter().enumerate() {
            let hue = i as f32 / source_positions.len() as f32;
            let color = hsl_to_rgba(hue, 0.9, 0.6, 1.0);
            renderer.debug_sphere(pos.into(), 0.25, color, 16);
            let dir = glam::Vec3::new((i as f32 * 1.5).sin(), -0.2, (i as f32 * 1.5).cos()).normalize();
            renderer.debug_cone((pos + dir * 0.3).into(), dir.into(), 1.5, 0.8, [color[0], color[1], color[2], 0.3], 12);
            renderer.debug_circle(pos.into(), 5.0, [color[0], color[1], color[2], 0.15], 32);
        }
        renderer.debug_sphere(listener_pos.into(), 0.2, [0.0, 1.0, 0.3, 1.0], 12);
        let look_dir = (glam::Vec3::ZERO - listener_pos).normalize();
        renderer.debug_cone((listener_pos + look_dir * 0.2).into(), look_dir.into(), 0.4, 0.15, [0.0, 0.8, 0.0, 0.4], 8);

        // Billboard speaker icons at each source position
        let billboards: Vec<helio::BillboardInstance> = source_positions.iter().enumerate().map(|(i, &pos)| {
            let c = hsl_to_rgba(i as f32 / source_positions.len() as f32, 0.9, 0.6, 1.0);
            helio::BillboardInstance {
                world_pos: [pos.x, pos.y + 1.2, pos.z, 1.0],
                scale_flags: [0.5, 0.5, 0.0, 0.0],
                color: c,
            }
        }).collect();
        renderer.set_billboard_instances(&billboards);

        if self.show_rays {
            for (i, &src_pos) in source_positions.iter().enumerate() {
                let hue = i as f32 / source_positions.len() as f32;
                let base = hsl_to_rgba(hue, 0.9, 0.6, 1.0);
                renderer.debug_line(src_pos.into(), listener_pos.into(), base);
                let walls = [glam::Vec3::new(-11.0, 1.0, listener_pos.z * 0.5), glam::Vec3::new(11.0, 1.0, listener_pos.z * 0.3)];
                for (j, &wp) in walls.iter().enumerate() {
                    let f = 1.0 - j as f32 * 0.2;
                    let c = [base[0]*f, base[1]*f, base[2]*f, 0.6];
                    renderer.debug_line(src_pos.into(), wp.into(), c);
                    renderer.debug_line(wp.into(), listener_pos.into(), c);
                    renderer.debug_sphere(wp.into(), 0.08, [1.0, 1.0, 0.0, 0.8], 8);
                }
            }
        }
        if self.show_probes {
            for x in -2..=2 { for z in -2..=2 {
                let p = glam::Vec3::new(x as f32 * 3.0, 0.5, z as f32 * 3.0);
                renderer.debug_sphere(p.into(), 0.08, [0.3, 0.6, 1.0, 0.7], 6);
                if x < 2 { renderer.debug_line(p.into(), glam::Vec3::new((x+1) as f32*3.0, 0.5, z as f32*3.0).into(), [0.3,0.6,1.0,0.15]); }
                if z < 2 { renderer.debug_line(p.into(), glam::Vec3::new(x as f32*3.0, 0.5, (z+1) as f32*3.0).into(), [0.3,0.6,1.0,0.15]); }
            }}
        }
        if self.show_material_zones {
            for x in -4..=4 { for z in -4..=4 {
                let center = glam::Vec3::new(x as f32 * 2.0, 0.01, z as f32 * 2.0);
                let color = if (x+z)%2==0 { [0.8,0.2,0.2,0.3] } else { [0.3,0.3,0.8,0.15] };
                renderer.debug_filled_box(center.into(), 0.96, color);
            }}
        }

        // Scene state is persistent — no per-frame setup needed.

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        self.queue.present(output);
    }
}



