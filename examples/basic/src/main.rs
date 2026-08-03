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
//!   G           — swap Aux Left/Right channels (live patch-bay remap)
//!   [ / ]       — master volume down / up (3 dB steps; default +18 dB)
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
use v3_demo_common::{box_mesh, cube_mesh, make_material, plane_mesh, point_light};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use quasar_core::bands::Band8;
use quasar_core::hybrid::HybridSamplingStrategy;
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::scene::{AcousticMesh as QMesh, AcousticScene as QScene};
use quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig,
    SceneOutputId, SourceConfig, SourceId,
};
use quasar_audio::SpatialAudioEngine;
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use quasar_materials::instance::AcousticMaterialInstance;
use quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};

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
const COLUMN_Z: &[f32] = &[-22.0, 18.0];

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

/// Number of scene outputs / speakers in the cathedral stage.
const NUM_SPEAKERS: usize = 8;
/// Single source of truth for the 8 cathedral speaker positions.
/// index i == WAV channel i: 0 FrontLeft, 1 FrontRight, 2 Center, 3 BackLeft,
/// 4 BackRight, 5 Sub, 6 AuxLeft, 7 AuxRight.
const SPEAKER_POSITIONS: [glam::Vec3; 8] = [
    glam::Vec3::new(-7.0, 5.5,-12.0), // 0 Front Left
    glam::Vec3::new( 7.0, 5.5,-12.0), // 1 Front Right
    glam::Vec3::new( 0.0, 3.0,-12.0), // 2 Center
    glam::Vec3::new(-7.0, 2.0, 12.0), // 3 Back Left
    glam::Vec3::new( 7.0, 2.0, 12.0), // 4 Back Right
    glam::Vec3::new( 0.0, 0.3, -7.0), // 5 Sub
    glam::Vec3::new(-7.0, 0.5,-12.0), // 6 Aux Left
    glam::Vec3::new( 7.0, 0.5,-12.0), // 7 Aux Right
];

struct WavPlayback {
    samples: Vec<f32>,
    num_channels: usize,
    read_pos: f64,
    rate_ratio: f64,
    /// Presentation-stage makeup gain (dB). The engine models absolute SPL —
    /// inverse-distance attenuation across this ~30 m cathedral lands ~ −24 dB
    /// below the source — so the demo applies a master volume on top.
    /// Range -60..+24 dB via [/]; default +24 is the loudest clean setting.
    master_gain_db: f32,
}

struct AudioEngine {
    engine: Arc<Mutex<SpatialAudioEngine>>,
    _state: Arc<Mutex<WavPlayback>>,
    _stream: cpal::Stream,
    levels: Arc<Mutex<[f32; NUM_SPEAKERS]>>,
    source_id: SourceId,
    outputs: [SceneOutputId; NUM_SPEAKERS],
    listener_id: ListenerId,
}

fn setup_audio_engine() -> AudioEngine {
    // Load WAV via hound
    let reader = hound::WavReader::open("assets/8_Channel_ID.wav").expect("open WAV");
    let spec = reader.spec();
    let wav_sr = spec.sample_rate;
    let nch_wav = spec.channels as usize;
    let mut samples: Vec<f32> = match spec.bits_per_sample {
        16 => reader.into_samples::<i16>().filter_map(|s| s.ok()).map(|s| s as f32 / i16::MAX as f32).collect(),
        24 => reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / 8_388_608.0).collect(),
        32 => reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / i32::MAX as f32).collect(),
        b => panic!("unsupported bit depth: {b}"),
    };
    // Normalize to a healthy 1 m reference level (peak ~0.9 FS) so the engine's
    // inverse-distance attenuation across the cathedral stays clearly audible.
    // Capped at +24 dB so a quiet file is not boosted into a noise wall.
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 0.0 && peak < 1.0 {
        let gain = (0.9 / peak).min(16.0);
        samples = samples.into_iter().map(|s| s * gain).collect();
    }

    // Set up cpal
    let host = cpal::default_host();
    let device = host.default_output_device().expect("audio output device");
    let out_config = device.default_output_config().expect("output config");
    let out_sr = out_config.sample_rate().0;
    let out_ch = out_config.channels() as usize;
    let sr = out_sr as f32;

    // Scene-pipeline engine: the demo no longer uses the legacy per-WAV-channel
    // graph (occ → rev → dec). One multi-channel Source is loaded, positioned
    // SceneOutputs pull their content via ChannelPulls, and the engine renders
    // onto one Listener whose physical layout matches the real output device.
    let mut engine = SpatialAudioEngine::new(0, sr, 15.0);

    // Acoustic materials drive occlusion filtering and reflections. Polished
    // stone floor (low absorption 0.15), rough stone walls + columns (0.35).
    // Registered BEFORE the backend is created so mesh handles are valid.
    engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let floor_mat = engine.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.15), Band8::zeros(), Band8::zeros()),
    ));
    let wall_mat = engine.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.35), Band8::zeros(), Band8::zeros()),
    ));

    // Acoustic proxy scene: floor, two side walls, plus the 4 nave columns so
    // occluding a speaker behind a column is demonstrable (columns are 0.65 × 20
    // × 0.65 acoustic boxes at x = ±5.5, z = -22 / +18).
    let mut qs = QScene::new();
    qs.add_mesh(QMesh::new(1, vec![[-11.,0.,-28.],[11.,0.,-28.],[11.,0.,28.],[-11.,0.,28.]], vec![0,1,2,0,2,3], floor_mat));
    qs.add_mesh(QMesh::new(2, vec![[-11.,0.,-28.],[-11.,0.,28.],[-11.,21.,28.],[-11.,21.,-28.]], vec![0,1,2,0,2,3], wall_mat));
    qs.add_mesh(QMesh::new(3, vec![[11.,0.,-28.],[11.,0.,28.],[11.,21.,28.],[11.,21.,-28.]], vec![0,1,2,0,2,3], wall_mat));
    for (i, &cz) in COLUMN_Z.iter().enumerate() {
        for (j, &cx) in [-5.5_f32, 5.5].iter().enumerate() {
            qs.add_mesh(QMesh::new(
                4 + (i * 2 + j) as u64,
                vec![
                    [cx - 0.325, 0.0, cz - 0.325],
                    [cx + 0.325, 0.0, cz - 0.325],
                    [cx + 0.325, 0.0, cz + 0.325],
                    [cx - 0.325, 0.0, cz + 0.325],
                    [cx - 0.325, 20.0, cz - 0.325],
                    [cx + 0.325, 20.0, cz - 0.325],
                    [cx + 0.325, 20.0, cz + 0.325],
                    [cx - 0.325, 20.0, cz + 0.325],
                ],
                vec![
                    0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1,
                    2,6,7, 2,7,3, 0,3,7, 0,7,4, 1,5,6, 1,6,2,
                ],
                wall_mat,
            ));
        }
    }
    let cfg = CpuSimdConfig {
        max_reflection_order: 3, diffuse_rays_per_query: 128, max_reflection_distance: 60.,
        speed_of_sound: 343., temperature_celsius: 20., humidity_percent: 50.,
    };
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(qs, cfg)));

    // Baked probe grid covering the whole navigable cathedral so HybridBlend
    // late reverb is probe-driven everywhere the camera goes. T60 ramps from
    // ~7 s near the altar (z = -28) to ~4.2 s at the entrance (z = +28).
    // Probe order must match grid cell indexing (z*sy*sx + y*sx + x): z outer,
    // y middle, x inner. Any other order scrambles which t60 is sampled where.
    let mut probes = Vec::with_capacity(5 * 5 * 9);
    for z in 0..9 {
        for y in 0..5 {
            for x in 0..5 {
                let position = [
                    -12.0 + x as f32 * 6.0,
                    0.0 + y as f32 * 4.0,
                    -28.0 + z as f32 * 7.0,
                ];
                let f = ((-position[2] + 28.0) / 56.0).clamp(0.0, 1.0);
                let t60 = 4.2 + 2.8 * f;
                probes.push(AcousticProbe {
                    position,
                    rir_samples: Vec::new(),
                    sample_rate: 48000,
                    t60: Band8::splat(t60),
                    broadband_t60: t60,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }
    engine.set_probe_grid(
        AcousticProbeGrid::new(probes, [-12.0, 0.0, -28.0], [6.0, 4.0, 7.0], [5, 5, 9])
            .expect("probe grid"),
    );
    engine.set_strategy(HybridSamplingStrategy::HybridBlend);

    // Load the 8-channel WAV as ONE source; the patch bay taps individual channels.
    let source_id = engine
        .load_source(SourceConfig {
            path: "assets/8_Channel_ID.wav".to_string(),
            channels: nch_wav,
        })
        .expect("load source");

    // One scene output per cathedral speaker; output i pulls (source, ch i, 0 dB).
    // Using connect_pull (rather than pre-filled pulls) demonstrates the patch bay.
    let mut outputs = [SceneOutputId(0); NUM_SPEAKERS];
    for (i, &pos) in SPEAKER_POSITIONS.iter().enumerate() {
        let out_id = engine.add_scene_output(SceneOutputConfig::new(
            pos.to_array(),
            quasar_core::scene::Movability::Static,
        ));
        engine.connect_pull(out_id, ChannelPull::new(source_id, i as u32, 0.0));
        outputs[i] = out_id;
    }

    // Physical device layout derived from the real output channel count.
    let physical_layout = match out_ch {
        2 => PhysicalOutputLayout::Stereo,
        4 => PhysicalOutputLayout::Quad,
        6 => PhysicalOutputLayout::Surround51,
        8 => PhysicalOutputLayout::Custom { positions: SPEAKER_POSITIONS.map(|p| p.to_array()).to_vec() },
        n => PhysicalOutputLayout::Custom {
            positions: (0..n)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / n as f32;
                    [a.sin(), 0.0, -a.cos()]
                })
                .collect(),
        },
    };
    let listener_id = engine.add_listener(ListenerConfig {
        position: [0.0, 1.6, 0.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout,
    });

    let engine = Arc::new(Mutex::new(engine));
    let rate_ratio = wav_sr as f64 / out_sr as f64;

    let state = Arc::new(Mutex::new(WavPlayback {
        samples, num_channels: nch_wav, read_pos: 0.0, rate_ratio, master_gain_db: 24.0,
    }));

    let levels = Arc::new(Mutex::new([0.0_f32; NUM_SPEAKERS]));
    let eng_cb = engine.clone();
    let state_cb = state.clone();
    let levels_cb = levels.clone();
    let out_ch_cb = out_ch;
    let err_fn = |e: cpal::StreamError| eprintln!("Audio error: {e}");

    let stream = device.build_output_stream(
        &out_config.config(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let total_frames = data.len() / out_ch_cb;
            data.fill(0.0);
            if total_frames == 0 { return; }

            let mut w = match state_cb.lock() {
                Ok(guard) => guard,
                Err(_) => return, // poisoned playback state → keep this block silent
            };
            let total_raw = w.samples.len();
            let nch = w.num_channels;
            let ratio = w.rate_ratio;
            let master_gain = 10.0_f32.powf(w.master_gain_db / 20.0);
            let mut remain = total_frames;
            let mut offset = 0;

            while remain > 0 {
                let block = (DEFAULT_BLOCK_SIZE as usize).min(remain);

                // ONE source buffer: deinterleave/resample every WAV channel into
                // channel k of the source (the patch bay indexes by channel).
                let mut src = AudioBuffer::new(nch as u16, block as u16);
                for k in 0..nch.min(NUM_SPEAKERS) {
                    let ch = src.channel_mut(k as u16);
                    for i in 0..block {
                        let pos = w.read_pos + i as f64 * ratio;
                        let fa = pos.floor() as usize;
                        let fb = fa + 1;
                        let frac = (pos - fa as f64) as f32;
                        let g = |f: usize| -> f32 {
                            if f < total_raw / nch { w.samples[f * nch + k] } else { 0.0 }
                        };
                        ch[i] = g(fa) + (g(fb) - g(fa)) * frac;
                    }
                }
                // Compute per-channel RMS for billboard flash feedback
                if let Ok(mut lvls) = levels_cb.lock() {
                    for k in 0..nch.min(NUM_SPEAKERS) {
                        let ch = src.channel(k as u16);
                        let sum_sq: f32 = ch.iter().take(block).map(|&s| s * s).sum();
                        lvls[k] = (sum_sq / block as f32).sqrt();
                    }
                }
                w.read_pos += block as f64 * ratio;
                if w.read_pos >= (total_raw / nch) as f64 { w.read_pos = 0.0; }

                // Process through the scene pipeline: patch bay → spatial render →
                // decode onto the listener's physical layout. Zero allocations.
                let mut out = AudioBuffer::new(out_ch_cb as u16, block as u16);
                if let Ok(mut e) = eng_cb.lock() {
                    e.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
                }

                for i in 0..block {
                    let dst = offset + i;
                    for c in 0..out_ch_cb.min(out.channels() as usize) {
                        data[dst * out_ch_cb + c] = out.channel(c as u16)[i] * master_gain;
                    }
                }

                remain -= block;
                offset += block;
            }
        },
        err_fn, None,
    ).expect("build output stream");
    stream.play().expect("play stream");

    AudioEngine { engine, _state: state, _stream: stream, levels, source_id, outputs, listener_id }
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
    // Aux Left/Right pulls swapped live via the G key (patch-bay remap).
    aux_swapped: bool,
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

        // Listener position marker — a Helio cube mesh that all speakers point toward
        let listener_mat = renderer.scene_mut().insert_material(make_material(
            [0.0, 1.0, 0.3, 1.0],
            0.2,
            0.0,
            [0.0, 1.0, 0.3],
            1.5,
        ));
        let listener_mesh = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.4))).as_mesh().unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            listener_mesh,
            listener_mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.6, 0.0)),
            0.4,
        );

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
            aux_swapped: false,
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

            // G: live patch-bay remap — swap Aux Left/Right channel pulls.
            // Remapping a speaker is a single runtime connect_pull call.
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyG),
                        ..
                    },
                ..
            } => {
                if let Ok(mut engine) = state._audio_engine.engine.lock() {
                    let src = state._audio_engine.source_id;
                    engine.disconnect_pull(state._audio_engine.outputs[6], src, 6);
                    engine.connect_pull(state._audio_engine.outputs[6], ChannelPull::new(src, 7, 0.0));
                    engine.disconnect_pull(state._audio_engine.outputs[7], src, 7);
                    engine.connect_pull(state._audio_engine.outputs[7], ChannelPull::new(src, 6, 0.0));
                }
                state.aux_swapped = !state.aux_swapped;
                println!("[audio] aux channels swapped = {}", state.aux_swapped);
            }

            // [ / ]: master volume down / up (3 dB steps).
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::BracketLeft),
                        ..
                    },
                ..
            } => {
                if let Ok(mut w) = state._audio_engine._state.lock() {
                    w.master_gain_db = (w.master_gain_db - 3.0).max(-60.0);
                    println!("[audio] master gain = {} dB", w.master_gain_db);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::BracketRight),
                        ..
                    },
                ..
            } => {
                if let Ok(mut w) = state._audio_engine._state.lock() {
                    w.master_gain_db = (w.master_gain_db + 3.0).min(24.0);
                    println!("[audio] master gain = {} dB", w.master_gain_db);
                }
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
        // Stage speaker layout (all point toward the listener cube at (0, 1.6, 0)):
        // Index order matches WAV channel → scene output mapping (SPEAKER_POSITIONS):
        //  0: Front Left           — WAV ch 0 / scene output 0
        //  1: Front Right          — WAV ch 1 / scene output 1
        //  2: Center               — WAV ch 2 / scene output 2
        //  3: Back Left            — WAV ch 3 / scene output 3
        //  4: Back Right           — WAV ch 4 / scene output 4
        //  5: Sub                  — WAV ch 5 / scene output 5
        //  6: Aux Left             — WAV ch 6 / scene output 6
        //  7: Aux Right            — WAV ch 7 / scene output 7

        // Update the scene pipeline: move the listener with the camera, then
        // resolve every (scene output, listener) pair (compute thread side).
        if let Ok(mut engine) = self._audio_engine.engine.lock() {
            engine.update_listener(self._audio_engine.listener_id, self.cam_pos.to_array(), forward.to_array());
            engine.update_scene_spatial();
        }
        renderer.debug_clear();
        for (i, &pos) in SPEAKER_POSITIONS.iter().enumerate() {
            let hue = i as f32 / SPEAKER_POSITIONS.len() as f32;
            let color = hsl_to_rgba(hue, 0.9, 0.6, 1.0);
            renderer.debug_sphere(pos.into(), 0.25, color, 16);
            let dir = (glam::Vec3::new(0.0, 1.6, 0.0) - pos).normalize();
            renderer.debug_cone((pos + dir * 0.3).into(), dir.into(), 1.5, 0.8, [color[0], color[1], color[2], 0.3], 12);
            renderer.debug_circle(pos.into(), 2.0, [color[0], color[1], color[2], 0.12], 24);
        }
        renderer.debug_sphere(listener_pos.into(), 0.2, [0.0, 1.0, 0.3, 1.0], 12);
        renderer.debug_cone((listener_pos + forward * 0.2).into(), forward.into(), 0.4, 0.15, [0.0, 0.8, 0.0, 0.4], 8);

        // Billboard speaker icons at each speaker position, flash on audio activity
        let src_levels = if let Ok(lvls) = self._audio_engine.levels.lock() { *lvls } else { [0.0; NUM_SPEAKERS] };
        let billboards: Vec<helio::BillboardInstance> = SPEAKER_POSITIONS.iter().enumerate().map(|(i, &pos)| {
            let lvl = src_levels[i];
            let active = lvl > 0.005;
            let scale = if active { (0.5 + lvl * 4.0).min(1.5) } else { 0.5 };
            let mut c = hsl_to_rgba(i as f32 / SPEAKER_POSITIONS.len() as f32, 0.9, 0.6, 1.0);
            if active {
                let boost = (lvl * 6.0).min(1.0);
                c[0] = c[0] * (1.0 - boost) + boost;
                c[1] = c[1] * (1.0 - boost) + boost;
                c[2] = c[2] * (1.0 - boost) + boost;
            }
            helio::BillboardInstance {
                world_pos: [pos.x, pos.y + 1.2, pos.z, 1.0],
                scale_flags: [scale, scale, 0.0, 0.0],
                color: c,
            }
        }).collect();
        renderer.set_billboard_instances(&billboards);

        if self.show_rays {
            for (i, &src_pos) in SPEAKER_POSITIONS.iter().enumerate() {
                let hue = i as f32 / SPEAKER_POSITIONS.len() as f32;
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