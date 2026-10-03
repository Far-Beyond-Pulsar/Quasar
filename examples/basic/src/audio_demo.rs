//! The Quasar side of the demo: stage layout, engine construction from the SceneDB geometry, the
//! lock-free cpal playback callback, the background spatial worker and the live mix controls.
//!
//! Layout of the work between threads:
//!
//! * the **audio callback** (cpal) owns the [`quasar_audio::AudioRenderer`] and the streaming
//!   playback BY VALUE and shares only atomics with the rest of the program (no mutex);
//! * the **spatial worker** (`SpatialWorker`) is a dedicated thread: it receives the listener pose
//!   over a tiny shared slot, takes the engine mutex, runs `update_scene_spatial` at ~30 Hz and
//!   publishes the retained debug frame, so the render loop never waits for acoustic ray tracing;
//! * the **render thread** only writes the pose and reads the published frame / statistics.

use crate::acoustic_geometry::{self, AcousticClass, ExtractStats};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use quasar_audio::SpatialAudioEngine;
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_backends::debug_capture::{AcousticDebugCapture, AcousticDebugFrame};
use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::bands::Band8;
use quasar_core::hybrid::HybridSamplingStrategy;
use quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_core::streaming_source::StreamingSource;
use quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SceneOutputId,
    SourceConfig, SourceId,
};
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use quasar_materials::instance::AcousticMaterialInstance;
use quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Minimum seconds between `update_scene_spatial()` calls (~30 Hz).
pub const SPATIAL_UPDATE_INTERVAL: f32 = 1.0 / 30.0;

/// Number of scene outputs / speakers on the stage.
pub const NUM_SPEAKERS: usize = 8;

/// Where the stage speakers point and where the listener starts: the audience point, at ear (camera)
/// height. The old demo's audience point was the origin of its 22 x 56 m hall; this is the same
/// stage translated into the nave of the large cathedral, near the entrance end, looking down the
/// nave toward the altar (-z). It is 16 m from the entrance wall (z = +72): with the old
/// +12 m rear speakers (BL / BR) a point nearer the wall would put them outside the building.
pub const AUDIENCE: [f32; 3] = [0.0, 2.3, 56.0];

/// Old stage speaker offsets from the old audience point, `[dx, dz]` (metres; -z is toward the
/// altar). Device-channel order FL FR C Sub BL BR SL SR (see [`CHANNEL_MAP`]).
const STAGE_OFFSET_XZ: [[f32; 2]; NUM_SPEAKERS] = [
    [-7.0, -12.0],
    [7.0, -12.0],
    [0.0, -12.0],
    [0.0, -7.0],
    [-7.0, 12.0],
    [7.0, 12.0],
    [-7.0, -12.0],
    [7.0, -12.0],
];
/// Speaker heights ABOVE THE FLOOR (the old values; the floor is y = 0 in both halls).
const STAGE_HEIGHT: [f32; NUM_SPEAKERS] = [5.5, 5.5, 3.0, 0.3, 2.0, 2.0, 0.5, 0.5];

/// The 8 speaker positions in STANDARD DEVICE CHANNEL ORDER (FL, FR, C, Sub/LFE, BL, BR, SL, SR),
/// translated so the audience point replaces the old stage origin.
pub fn speaker_positions(audience: [f32; 3]) -> [glam::Vec3; NUM_SPEAKERS] {
    std::array::from_fn(|i| {
        glam::Vec3::new(audience[0] + STAGE_OFFSET_XZ[i][0], STAGE_HEIGHT[i], audience[2] + STAGE_OFFSET_XZ[i][1])
    })
}

/// Maps device-channel index -> WAV-channel index (WAV: 0=FL 1=FR 2=C 3=BL 4=BR 5=Sub 6=SL 7=SR).
pub const CHANNEL_MAP: [u32; NUM_SPEAKERS] = [0, 1, 2, 5, 3, 4, 6, 7];

// ── Demo mix (ARTISTIC defaults; the engine's own defaults stay physical) ───
//
// A speaker radiating omnidirectionally into a multi-second hall produces a diffuse field as loud
// as the direct sound at only a few metres, so from the audience the stage is mostly reverb. Real PA
// speakers are directional and a listener wants more direct sound than a hall gives, so the demo
// (a) aims every speaker at the audience and (b) pulls the reverb bus down a little. The numbers
// were measured in the old 22 x 56 m demo hall with `cargo test -p quasar-audio --release --test
// direct_vs_room_levels -- --nocapture`; the large cathedral has a different reverberation, the
// relative stage layout (hence the direct-sound distances) is unchanged. `-` / `=` and `;` / `'`
// change the trims live.
/// Speaker directivity (0 = omni, 1 = cardioid; 0.7 is a typical PA horn: rear about -10 dB at 1 kHz).
pub const DEMO_DIRECTIVITY: f32 = 0.7;
/// Initial reverb-bus trim in dB (0 = physical level).
pub const DEMO_REVERB_DB: f32 = -5.0;
/// Initial early-reflection trim in dB (0 = physical level).
pub const DEMO_EARLY_DB: f32 = 0.0;

/// Acoustic-proxy / tracer settings of this demo (see the README, "Geometry and materials").
pub fn tracer_config(sample_rate: f32) -> CpuSimdConfig {
    CpuSimdConfig {
        // Order 2: measured on the 415 k-triangle cathedral, order 3 with 32 mirror planes costs
        // ~100 ms and ~300 k rays per update (8 emitters), order 2 about 10-14 ms and ~50-65 k rays.
        // Third-order paths are far below the late field in a hall this size.
        max_reflection_order: 2,
        diffuse_rays_per_query: 128,
        // The hall is 144 m long: a reflected path from a stage speaker to the far end is well over
        // the old demo's 60 m limit.
        max_reflection_distance: 150.0,
        speed_of_sound: 343.0,
        temperature_celsius: 20.0,
        humidity_percent: 50.0,
        sample_rate,
        ..CpuSimdConfig::default()
    }
}

/// Probe-grid layout derived from the scene AABB.
#[derive(Clone, Debug)]
pub struct ProbeGridInfo {
    pub origin: [f32; 3],
    pub spacing: [f32; 3],
    pub dims: [u32; 3],
    pub t60: Band8,
}

impl ProbeGridInfo {
    pub fn probe_count(&self) -> usize {
        (self.dims[0] * self.dims[1] * self.dims[2]) as usize
    }
}

/// Probe-grid bounds from the scene AABB: about `target_spacing` m apart horizontally, vertically
/// 11 m (the camera hovers near the floor; the grid still covers the vault).
pub fn probe_grid_layout(aabb_min: [f32; 3], aabb_max: [f32; 3], t60: Band8) -> ProbeGridInfo {
    let target = [12.0_f32, 11.0, 12.0];
    let dims: [u32; 3] = std::array::from_fn(|i| (((aabb_max[i] - aabb_min[i]) / target[i]).ceil() as u32 + 1).max(2));
    let spacing: [f32; 3] =
        std::array::from_fn(|i| ((aabb_max[i] - aabb_min[i]) / (dims[i] - 1) as f32).max(0.5));
    ProbeGridInfo { origin: aabb_min, spacing, dims, t60 }
}

/// Probes in grid-cell order (z outer, y middle, x inner: any other order scrambles which probe is
/// sampled where). Every probe carries the same statistical T60 (see the README).
pub fn build_probe_grid(info: &ProbeGridInfo) -> AcousticProbeGrid {
    let mut probes = Vec::with_capacity(info.probe_count());
    let broadband = info.t60.0.iter().sum::<f32>() / 8.0;
    for z in 0..info.dims[2] {
        for y in 0..info.dims[1] {
            for x in 0..info.dims[0] {
                probes.push(AcousticProbe {
                    position: [
                        info.origin[0] + x as f32 * info.spacing[0],
                        info.origin[1] + y as f32 * info.spacing[1],
                        info.origin[2] + z as f32 * info.spacing[2],
                    ],
                    rir_samples: Vec::new(),
                    sample_rate: 48_000,
                    t60: info.t60,
                    broadband_t60: broadband,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }
    AcousticProbeGrid::new(probes, info.origin, info.spacing, info.dims).expect("probe grid")
}

/// Everything [`build_engine`] reports besides the engine itself.
pub struct BuiltEngine {
    pub engine: SpatialAudioEngine,
    pub debug_capture: Arc<AcousticDebugCapture>,
    pub source_id: SourceId,
    pub outputs: [SceneOutputId; NUM_SPEAKERS],
    pub listener_id: ListenerId,
    pub scene_stats: ExtractStats,
    pub probe_grid: ProbeGridInfo,
    /// Wall-clock milliseconds of the SceneDB read-back, and of the backend construction (BVH, planes).
    pub extract_ms: f64,
    pub backend_build_ms: f64,
    pub reflection_planes: usize,
    pub room_closed: bool,
    /// Every edge is shared by two triangles: inside / outside queries (and the closed-room shortcuts) work.
    pub watertight: bool,
    /// Statistical late-field T60 per band of the backend at the audience point (source at the centre speaker).
    pub statistical_t60: Band8,
    /// Engine material handle of each acoustic class.
    pub class_handles: HashMap<AcousticClass, u32>,
}

/// Build the engine from the geometry found in `world` (SceneDB): one acoustic mesh per renderable
/// object, one tabular material per acoustic class, the CPU-SIMD backend, a probe grid over the scene AABB
/// (T60 from the backend's own statistical estimate) and the 8-speaker stage around `audience`.
pub fn build_engine(
    world: &pulsar_scenedb::World,
    sample_rate: f32,
    wav_path: &str,
    wav_channels: usize,
    physical_layout: PhysicalOutputLayout,
    audience: [f32; 3],
    config: CpuSimdConfig,
) -> BuiltEngine {
    let mut engine = SpatialAudioEngine::new(0, sample_rate, 15.0);
    engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));

    // One acoustic material instance per class, registered BEFORE the backend so handles are valid.
    let mut handles: HashMap<AcousticClass, u32> = HashMap::new();
    for class in AcousticClass::ALL {
        let (absorption, scattering, mut transmission) = class.bands();
        if std::env::var_os("QUASAR_OPAQUE").is_some() {
            // Diagnostic: every surface fully opaque (the old demo shell's setting), to compare the
            // inside level balance with and without the mapped transmission.
            transmission = Band8::zeros();
        }
        let handle = engine.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(absorption, scattering, transmission),
        ));
        handles.insert(class, handle);
    }

    let t = Instant::now();
    let extracted = acoustic_geometry::extract_acoustic_scene(world, |class| handles[&class]);
    let extract_ms = t.elapsed().as_secs_f64() * 1e3;
    let scene_stats = extracted.stats;

    let t = Instant::now();
    let backend = CpuSimdComputeBackend::new(extracted.scene, config);
    let backend_build_ms = t.elapsed().as_secs_f64() * 1e3;
    let reflection_planes = backend.reflection_plane_count();
    let room_closed = backend.room_is_closed();
    // `room_side` answers `Outside` for a point far outside the bounding box only when the soup is
    // watertight (every edge shared by two triangles); otherwise it is always `Unknown`.
    let watertight = backend.room_side([1.0e6; 3]) == quasar_backends::cpu_simd::RoomSide::Outside;
    let debug_capture = backend.debug_capture();

    // The backend's own statistical late-field estimate for this geometry and these materials.
    let speakers = speaker_positions(audience);
    let statistical_t60 = backend
        .query_spatial(
            &[SpatialQuery { source_position: speakers[2].to_array(), listener_position: audience, source_id: 0 }],
            engine.materials(),
        )
        .first()
        .map(|r| r.late_reverb.t60)
        .unwrap_or_else(|| Band8::splat(5.0));
    backend.reset_ray_counter();
    engine.set_backend(Box::new(backend));

    let probe_grid = probe_grid_layout(scene_stats.aabb_min, scene_stats.aabb_max, statistical_t60);
    engine.set_probe_grid(build_probe_grid(&probe_grid));
    engine.set_strategy(HybridSamplingStrategy::HybridBlend);

    // Load the 8-channel WAV as ONE source; the patch bay taps individual channels.
    let source_id = engine
        .load_source(SourceConfig { path: wav_path.to_string(), channels: wav_channels })
        .expect("load source");

    // One scene output per speaker in device-channel order; CHANNEL_MAP routes the right WAV channel.
    let mut outputs = [SceneOutputId(0); NUM_SPEAKERS];
    for (dev_ch, &pos) in speakers.iter().enumerate() {
        let out_id = engine.add_scene_output(SceneOutputConfig::new(
            pos.to_array(),
            quasar_core::scene::Movability::Static,
        ));
        let wav_ch = *CHANNEL_MAP.get(dev_ch).unwrap_or(&(dev_ch as u32));
        engine.connect_pull(out_id, ChannelPull::new(source_id, wav_ch, 0.0));
        let to_audience = (glam::Vec3::from_array(audience) - pos).normalize();
        engine.set_scene_output_directivity(out_id, Some(to_audience.to_array()), DEMO_DIRECTIVITY);
        outputs[dev_ch] = out_id;
    }
    // The Sub/LFE output is also sent to the listener's LFE channel through the 120 Hz LFE low-pass.
    engine.set_scene_output_lfe_send(outputs[3], 1.0);

    let listener_id = engine.add_listener(ListenerConfig {
        position: audience,
        heading: [0.0, 0.0, -1.0],
        physical_layout,
    });
    engine.set_reverb_gain_db(listener_id, DEMO_REVERB_DB);
    engine.set_early_reflection_gain_db(listener_id, DEMO_EARLY_DB);

    BuiltEngine {
        engine,
        debug_capture,
        source_id,
        outputs,
        listener_id,
        scene_stats,
        probe_grid,
        extract_ms,
        backend_build_ms,
        reflection_planes,
        room_closed,
        watertight,
        statistical_t60,
        class_handles: handles,
    }
}

/// Physical device layout from the real output channel count (standard channel orders; 7.1 is
/// FL FR C LFE BL BR SL SR with LFE excluded from panning).
pub fn physical_layout_for(out_ch: usize) -> PhysicalOutputLayout {
    match out_ch {
        2 => PhysicalOutputLayout::Stereo,
        4 => PhysicalOutputLayout::Quad,
        6 => PhysicalOutputLayout::Surround51,
        8 => PhysicalOutputLayout::Surround714,
        n => PhysicalOutputLayout::Custom {
            positions: (0..n)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / n as f32;
                    [a.sin(), 0.0, -a.cos()]
                })
                .collect(),
        },
    }
}

/// Thin wrapper around `BufferedStream` for the audio callback. All disk I/O happens on a
/// background thread; the callback never blocks.
struct StreamingPlayback {
    stream: quasar_audio::streaming_source::BufferedStream,
    channels: usize,
    read_pos: f64,
    rate_ratio: f64,
}

impl StreamingPlayback {
    fn open(path: &str, output_sample_rate: f32) -> Self {
        // Scan the first chunk for the peak (blocking, setup only).
        let mut wave = quasar_audio::streaming_source::WaveFileStream::open(path).expect("open WAV for streaming");
        let sample_rate = wave.sample_rate();
        let channels = wave.channels();
        let total_frames = wave.total_frames().unwrap_or(0);

        let mut scan_buf = vec![0.0_f32; 4096 * channels];
        let n = wave.read_frames(&mut scan_buf);
        let peak = scan_buf[..n * channels].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        wave.seek_frames(0);

        // This demo's ambient bed is meant to loop forever: declare that explicitly.
        let policy_source = quasar_audio::streaming_source::PolicyOverride::new(
            wave,
            quasar_core::streaming_source::StreamingPolicy::Common,
        );
        let stream = quasar_audio::streaming_source::BufferedStream::new(Box::new(policy_source));

        eprintln!(
            "[quasar-stream] opened {} ({} ch, {} Hz, {} frames, peak={:.3})",
            path, channels, sample_rate, total_frames, peak,
        );

        Self { stream, channels, read_pos: 0.0, rate_ratio: sample_rate as f64 / output_sample_rate as f64 }
    }

    /// Read one sample from the ring buffer (non-blocking).
    fn source_sample(&self, frame: u64, ch: usize) -> f32 {
        self.stream.sample_at(frame, ch)
    }
}

/// Handles the UI / compute side keeps. The audio callback owns the [`quasar_audio::AudioRenderer`]
/// and the [`StreamingPlayback`] outright and shares only atomics with this side, so it never takes
/// a lock the compute pass (which holds `engine`) could be holding.
pub struct AudioEngine {
    /// See `BuiltEngine::watertight`.
    pub watertight: bool,
    pub debug_capture: Arc<AcousticDebugCapture>,
    /// Compute / configuration side (registries, ray tracing, command queue to the renderer).
    pub engine: Arc<Mutex<SpatialAudioEngine>>,
    _stream: cpal::Stream,
    /// Master gain in dB (f32 bits): UI state; applied as the output stage's pre-limiter gain, so the
    /// limiter ceiling holds whatever the master volume.
    pub master_gain_db: Arc<AtomicU32>,
    /// Per-speaker RMS (f32 bits), written by the callback, read by the UI.
    pub levels: Arc<[AtomicU32; NUM_SPEAKERS]>,
    pub source_id: SourceId,
    pub outputs: [SceneOutputId; NUM_SPEAKERS],
    pub listener_id: ListenerId,
    pub scene_stats: ExtractStats,
    pub probe_grid: ProbeGridInfo,
    pub speakers: [glam::Vec3; NUM_SPEAKERS],
}

impl AudioEngine {
    /// Nudge the listener's reverb (`early == false`) or early-reflection trim by `delta_db`
    /// (clamped to -30 ..= +12 dB); returns the new value.
    pub fn adjust_mix_trim_db(&self, early: bool, delta_db: f32) -> Option<f32> {
        let mut e = self.engine.lock().ok()?;
        let cur = if early { e.early_reflection_gain_db(self.listener_id) } else { e.reverb_gain_db(self.listener_id) };
        let db = (cur + delta_db).clamp(-30.0, 12.0);
        if early {
            e.set_early_reflection_gain_db(self.listener_id, db);
        } else {
            e.set_reverb_gain_db(self.listener_id, db);
        }
        Some(db)
    }

    /// Master volume: the output stage's pre-limiter gain (set through the engine's lock-free
    /// command queue), so the limiter ceiling (-1 dBFS) holds at any master volume.
    pub fn set_master_gain_db(&self, db: f32) {
        self.master_gain_db.store(db.to_bits(), AtomicOrdering::Relaxed);
        if let Ok(mut e) = self.engine.lock() {
            e.set_output_safety(
                self.listener_id,
                quasar_dsp::limiter::OutputSafetyConfig { headroom_db: db, ..Default::default() },
            );
        }
    }

    pub fn master_gain_db(&self) -> f32 {
        f32::from_bits(self.master_gain_db.load(AtomicOrdering::Relaxed))
    }

    pub fn speaker_levels(&self) -> [f32; NUM_SPEAKERS] {
        std::array::from_fn(|i| f32::from_bits(self.levels[i].load(AtomicOrdering::Relaxed)))
    }
}

/// Open the output device, stream the WAV and start playback. The engine is built from the
/// geometry in `world`.
pub fn setup_audio_engine(world: &pulsar_scenedb::World) -> AudioEngine {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("audio output device");
    let out_config = device.default_output_config().expect("output config");
    let out_sr = out_config.sample_rate().0;
    let out_ch = out_config.channels() as usize;
    let sr = out_sr as f32;

    // Open the WAV as a streaming source (no full-file load).
    let wav_path = crate::asset_path("8_Channel_ID.wav");
    let mut playback = StreamingPlayback::open(&wav_path, sr);
    let nch_wav = playback.channels;

    let built = build_engine(
        world,
        sr,
        &wav_path,
        nch_wav,
        physical_layout_for(out_ch),
        AUDIENCE,
        tracer_config(sr),
    );
    eprintln!(
        "[quasar] acoustic scene from SceneDB: {} | planes {} | closed {} | watertight {} | extract {:.1} ms, backend build {:.1} ms | probe grid {:?} @ {:?} m, T60 {:?}",
        built.scene_stats.summary(),
        built.reflection_planes,
        built.room_closed,
        built.watertight,
        built.extract_ms,
        built.backend_build_ms,
        built.probe_grid.dims,
        built.probe_grid.spacing,
        built.probe_grid.t60.0,
    );
    let built_watertight = built.watertight;
    let BuiltEngine { mut engine, debug_capture, source_id, outputs, listener_id, scene_stats, probe_grid, .. } = built;

    // Split the engine (#75): the audio callback takes the `AudioRenderer` (render state, triple
    // buffer readers, command-queue consumer) and the streaming playback state BY VALUE. Nothing
    // the callback touches is behind a mutex that the UI / compute thread also takes: the compute
    // pass (`update_scene_spatial`, ray tracing) holds only the `engine` mutex, and configuration
    // reaches the renderer through the engine's lock-free command queue.
    let mut renderer = engine.audio_handle();
    let engine = Arc::new(Mutex::new(engine));

    let master_gain_db = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
    let levels: Arc<[AtomicU32; NUM_SPEAKERS]> = Arc::new(std::array::from_fn(|_| AtomicU32::new(0)));
    let levels_cb = levels.clone();
    let out_ch_cb = out_ch;
    let err_fn = |e: cpal::StreamError| eprintln!("Audio error: {e}");

    let stream = device
        .build_output_stream(
            &out_config.config(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let total_frames = data.len() / out_ch_cb;
                data.fill(0.0);
                if total_frames == 0 {
                    return;
                }

                let nch = playback.channels;
                let ratio = playback.rate_ratio;
                let mut remain = total_frames;
                let mut offset = 0;

                while remain > 0 {
                    let block = (DEFAULT_BLOCK_SIZE).min(remain);

                    let mut src = AudioBuffer::new(nch as u16, block as u16);
                    for k in 0..nch.min(NUM_SPEAKERS) {
                        let ch = src.channel_mut(k as u16);
                        for i in 0..block {
                            let pos = playback.read_pos + i as f64 * ratio;
                            let fa = pos.floor() as u64;
                            let fb = fa + 1;
                            let frac = (pos - fa as f64) as f32;
                            ch[i] = playback.source_sample(fa, k)
                                + (playback.source_sample(fb, k) - playback.source_sample(fa, k)) * frac;
                        }
                    }
                    for k in 0..nch.min(NUM_SPEAKERS) {
                        let ch = src.channel(k as u16);
                        let sum_sq: f32 = ch.iter().take(block).map(|&s| s * s).sum();
                        levels_cb[k].store((sum_sq / block as f32).sqrt().to_bits(), AtomicOrdering::Relaxed);
                    }
                    let source_frames = (block as f64 * ratio).ceil() as u64;
                    playback.stream.advance_read(source_frames);
                    playback.read_pos += block as f64 * ratio;

                    let mut out = AudioBuffer::new(out_ch_cb as u16, block as u16);
                    renderer.process_audio_scene(&[&src], std::slice::from_mut(&mut out));

                    for i in 0..block {
                        let dst = offset + i;
                        for c in 0..out_ch_cb.min(out.channels() as usize) {
                            data[dst * out_ch_cb + c] = out.channel(c as u16)[i];
                        }
                    }

                    remain -= block;
                    offset += block;
                }
            },
            err_fn,
            None,
        )
        .expect("build output stream");
    stream.play().expect("play stream");

    AudioEngine {
        watertight: built_watertight,
        debug_capture,
        engine,
        _stream: stream,
        master_gain_db,
        levels,
        source_id,
        outputs,
        listener_id,
        scene_stats,
        probe_grid,
        speakers: speaker_positions(AUDIENCE),
    }
}

// ── Background spatial worker ───────────────────────────────────────────────

/// Listener pose handed from the render thread to the worker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ListenerPose {
    pub position: [f32; 3],
    pub forward: [f32; 3],
}

/// Per-update timing the worker publishes (microseconds, f64 bits; counters are relaxed atomics).
#[derive(Default)]
pub struct WorkerStats {
    pub updates: AtomicU64,
    pub last_us: AtomicU64,
    pub max_us: AtomicU64,
    pub total_us: AtomicU64,
    /// Rays traced by the last update, when the backend counter is readable through the capture frame.
    pub last_rays: AtomicU64,
}

impl WorkerStats {
    pub fn describe(&self) -> String {
        let n = self.updates.load(AtomicOrdering::Relaxed).max(1);
        format!(
            "spatial worker: {} updates, last {:.2} ms, avg {:.2} ms, max {:.2} ms (budget {:.1} ms at 30 Hz)",
            self.updates.load(AtomicOrdering::Relaxed),
            self.last_us.load(AtomicOrdering::Relaxed) as f64 / 1e3,
            self.total_us.load(AtomicOrdering::Relaxed) as f64 / 1e3 / n as f64,
            self.max_us.load(AtomicOrdering::Relaxed) as f64 / 1e3,
            SPATIAL_UPDATE_INTERVAL as f64 * 1e3,
        )
    }
}

struct WorkerShared {
    pose: Mutex<ListenerPose>,
    wake: Condvar,
    stop: AtomicBool,
    /// Capture the debug trace on the next updates (V key).
    capture: AtomicBool,
    /// Ask for one immediate update (after a key changed what is captured).
    kick: AtomicBool,
    frame: Mutex<Option<AcousticDebugFrame>>,
    stats: WorkerStats,
}

/// Dedicated thread running the ~30 Hz acoustic spatial update so the render loop never waits for it.
pub struct SpatialWorker {
    shared: Arc<WorkerShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SpatialWorker {
    pub fn spawn(
        engine: Arc<Mutex<SpatialAudioEngine>>,
        listener_id: ListenerId,
        debug_capture: Arc<AcousticDebugCapture>,
        initial: ListenerPose,
    ) -> Self {
        let shared = Arc::new(WorkerShared {
            pose: Mutex::new(initial),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            capture: AtomicBool::new(false),
            kick: AtomicBool::new(true),
            frame: Mutex::new(None),
            stats: WorkerStats::default(),
        });
        let worker = shared.clone();
        let thread = std::thread::Builder::new()
            .name("quasar-spatial".into())
            .spawn(move || {
                let interval = Duration::from_secs_f32(SPATIAL_UPDATE_INTERVAL);
                let mut next = Instant::now();
                while !worker.stop.load(AtomicOrdering::Relaxed) {
                    let now = Instant::now();
                    if now < next && !worker.kick.swap(false, AtomicOrdering::Relaxed) {
                        // Not due: sleep until the next tick, a kick or a stop.
                        let guard = worker.pose.lock().unwrap();
                        let _ = worker.wake.wait_timeout(guard, next - now).unwrap();
                        continue;
                    }
                    next = now + interval;
                    let pose = *worker.pose.lock().unwrap();
                    let capturing = worker.capture.load(AtomicOrdering::Relaxed);
                    let started = Instant::now();
                    if let Ok(mut engine) = engine.lock() {
                        engine.update_listener(listener_id, pose.position, pose.forward);
                        if capturing {
                            debug_capture.begin_update();
                        }
                        engine.update_scene_spatial();
                    }
                    let us = started.elapsed().as_micros() as u64;
                    if capturing {
                        let frame = debug_capture.take_frame();
                        // Unchanged scenes or a skipped engine query may yield an empty batch:
                        // keep the last useful drawing rather than erasing it.
                        if !frame.directs.is_empty() {
                            worker.stats.last_rays.store(frame.ray_count as u64, AtomicOrdering::Relaxed);
                            *worker.frame.lock().unwrap() = Some(frame);
                        }
                    }
                    let s = &worker.stats;
                    s.updates.fetch_add(1, AtomicOrdering::Relaxed);
                    s.last_us.store(us, AtomicOrdering::Relaxed);
                    s.max_us.fetch_max(us, AtomicOrdering::Relaxed);
                    s.total_us.fetch_add(us, AtomicOrdering::Relaxed);
                }
            })
            .expect("spawn spatial worker");
        Self { shared, thread: Some(thread) }
    }

    /// Publish the listener pose (render thread, every frame; cheap).
    pub fn set_pose(&self, pose: ListenerPose) {
        *self.shared.pose.lock().unwrap() = pose;
    }

    /// Start / stop recording debug frames; the last frame stays available while stopped.
    pub fn set_capture(&self, on: bool) {
        self.shared.capture.store(on, AtomicOrdering::Relaxed);
        self.kick();
    }

    /// Request an immediate update (a key changed what the overlay needs).
    pub fn kick(&self) {
        self.shared.kick.store(true, AtomicOrdering::Relaxed);
        self.shared.wake.notify_all();
    }

    /// Take the newest debug frame published since the last call, if any.
    pub fn take_frame(&self) -> Option<AcousticDebugFrame> {
        self.shared.frame.lock().unwrap().take()
    }

    pub fn stats(&self) -> &WorkerStats {
        &self.shared.stats
    }
}

impl Drop for SpatialWorker {
    fn drop(&mut self) {
        self.shared.stop.store(true, AtomicOrdering::Relaxed);
        self.shared.wake.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_keeps_the_old_relative_layout() {
        let a = AUDIENCE;
        let p = speaker_positions(a);
        // Same x / z offsets from the audience point as the old demo; heights are floor-relative.
        assert_eq!(p[0], glam::Vec3::new(a[0] - 7.0, 5.5, a[2] - 12.0));
        assert_eq!(p[3], glam::Vec3::new(a[0], 0.3, a[2] - 7.0));
        assert_eq!(p[5], glam::Vec3::new(a[0] + 7.0, 2.0, a[2] + 12.0));
        // Distance of every speaker to the audience point at the old ear height (1.6 m) is preserved
        // to within the 0.7 m ear-height difference.
        let old_audience = glam::Vec3::new(0.0, 1.6, 0.0);
        let old = [
            glam::Vec3::new(-7.0, 5.5, -12.0),
            glam::Vec3::new(7.0, 5.5, -12.0),
            glam::Vec3::new(0.0, 3.0, -12.0),
            glam::Vec3::new(0.0, 0.3, -7.0),
            glam::Vec3::new(-7.0, 2.0, 12.0),
            glam::Vec3::new(7.0, 2.0, 12.0),
            glam::Vec3::new(-7.0, 0.5, -12.0),
            glam::Vec3::new(7.0, 0.5, -12.0),
        ];
        for i in 0..NUM_SPEAKERS {
            let d_old = (old[i] - old_audience).length();
            let d_new = (p[i] - glam::Vec3::from_array(a)).length();
            assert!((d_old - d_new).abs() < 0.8, "speaker {i}: {d_old} vs {d_new}");
        }
    }

    #[test]
    fn probe_grid_covers_the_aabb_in_cell_order() {
        let info = probe_grid_layout([-23.0, 0.0, -72.5], [23.0, 43.0, 72.5], Band8::splat(6.0));
        assert!(info.dims.iter().all(|&d| d >= 2));
        let grid = build_probe_grid(&info);
        assert_eq!(grid.probes.len(), info.probe_count());
        let first = grid.probes.first().unwrap().position;
        let last = grid.probes.last().unwrap().position;
        assert_eq!(first, [-23.0, 0.0, -72.5]);
        for i in 0..3 {
            assert!((last[i] - [23.0, 43.0, 72.5][i]).abs() < 1e-3);
        }
        // x runs fastest, then y, then z.
        let p1 = grid.probes[1].position;
        assert!(p1[0] > first[0] && p1[1] == first[1] && p1[2] == first[2]);
    }

    #[test]
    fn channel_map_is_a_permutation() {
        let mut m = CHANNEL_MAP;
        m.sort_unstable();
        assert_eq!(m, [0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
