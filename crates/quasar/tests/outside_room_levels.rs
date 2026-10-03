//! Diagnostic: what does the engine output for a listener OUTSIDE the demo cathedral (5 m
//! beyond a side wall) with the demo's wall transmission of 0 versus a physically plausible
//! per-band transmission? Same offline method as `direct_vs_room_levels.rs`: noise through
//! debug stages 2 (direct) / 3 (+ early) / 4 (full), TOTAL power over all output channels.
//!
//! Run: `cargo test -p quasar-audio --release --test outside_room_levels -- --nocapture`

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// The demo's 8 stage speakers (device order).
const SPEAKERS: [[f32; 3]; 8] = [
    [-7.0, 5.5, -12.0],
    [7.0, 5.5, -12.0],
    [0.0, 3.0, -12.0],
    [0.0, 0.3, -7.0],
    [-7.0, 2.0, 12.0],
    [7.0, 2.0, 12.0],
    [-7.0, 0.5, -12.0],
    [7.0, 0.5, -12.0],
];

/// Proposed demo shell transmission (amplitude per band 62.5 Hz .. 8 kHz): about 1 m of
/// stone with stained-glass lancets. Mass law `TL = 20 log10(m f) - 47 dB`: 0.5 m of stone
/// (1200 kg/m2) is 50 dB at 62 Hz and 90 dB at 8 kHz, but glazing (about 50 kg/m2, roughly
/// 10 % of the area) leaks far more (about 23 dB at 62 Hz, 65 dB at 8 kHz) and dominates;
/// the area-weighted result is about -32 dB at LF falling to -60 dB at HF.
pub const PROPOSED_SHELL_TRANSMISSION: [f32; 8] = [0.025, 0.016, 0.009, 0.005, 0.0028, 0.0016, 0.001, 0.001];

fn quad(id: u64, p: [[f32; 3]; 4], mat: u32) -> AcousticMesh {
    AcousticMesh::new(id, p.to_vec(), vec![0, 1, 2, 0, 2, 3], mat)
}

fn cathedral(layout: PhysicalOutputLayout, listener: [f32; 3], transmission: [f32; 8]) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let mat = |e: &SpatialAudioEngine, a: [f32; 8]| {
        e.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::new(a), Band8::zeros(), Band8::new(transmission)),
        ))
    };
    let floor = mat(&e, [0.08, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22]);
    let wall = mat(&e, [0.06, 0.05, 0.05, 0.06, 0.07, 0.09, 0.12, 0.15]);
    let ceil = mat(&e, [0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07]);
    let mut s = AcousticScene::new();
    s.add_mesh(quad(1, [[-11., 0., -28.], [11., 0., -28.], [11., 0., 28.], [-11., 0., 28.]], floor));
    s.add_mesh(quad(2, [[-11., 0., -28.], [-11., 0., 28.], [-11., 21., 28.], [-11., 21., -28.]], wall));
    s.add_mesh(quad(3, [[11., 0., -28.], [11., 0., 28.], [11., 21., 28.], [11., 21., -28.]], wall));
    s.add_mesh(quad(8, [[-11., 21., -28.], [11., 21., -28.], [11., 21., 28.], [-11., 21., 28.]], ceil));
    s.add_mesh(quad(9, [[-11., 0., -28.], [11., 0., -28.], [11., 21., -28.], [-11., 21., -28.]], wall));
    s.add_mesh(quad(10, [[-11., 0., 28.], [11., 0., 28.], [11., 21., 28.], [-11., 21., 28.]], wall));
    let cfg = CpuSimdConfig {
        max_reflection_order: 3,
        diffuse_rays_per_query: 128,
        max_reflection_distance: 60.,
        sample_rate: SR,
        ..CpuSimdConfig::default()
    };
    e.set_backend(Box::new(CpuSimdComputeBackend::new(s, cfg)));

    let mut probes = Vec::new();
    for z in 0..9 {
        for y in 0..5 {
            for x in 0..5 {
                let position = [-12.0 + x as f32 * 6.0, y as f32 * 4.0, -28.0 + z as f32 * 7.0];
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
    e.set_probe_grid(AcousticProbeGrid::new(probes, [-12.0, 0.0, -28.0], [6.0, 4.0, 7.0], [5, 5, 9]).unwrap());
    e.set_strategy(HybridSamplingStrategy::HybridBlend);

    let src = e
        .load_source(SourceConfig { path: "n.wav".into(), channels: SPEAKERS.len() })
        .unwrap();
    for (i, &pos) in SPEAKERS.iter().enumerate() {
        let out = e.add_scene_output(SceneOutputConfig::new(pos, Movability::Static));
        e.connect_pull(out, ChannelPull::new(src, i as u32, 0.0));
    }
    e.add_listener(ListenerConfig { position: listener, heading: [0.0, 0.0, -1.0], physical_layout: layout });
    e.update_scene_spatial();
    e
}

fn render_rms(e: &mut SpatialAudioEngine, stage: u8, channels: u16, sources: usize) -> (f32, f32) {
    e.debug_audio_stage = stage;
    let mut seeds: Vec<u32> = (0..sources as u32).map(|i| 0x1234_5678 ^ (i + 1).wrapping_mul(0x9E37_79B9)).collect();
    let mut lps = vec![0.0_f32; sources];
    let mut out = AudioBuffer::new(channels, BLOCK as u16);
    let (mut sum, mut n, mut sum_in) = (0.0_f64, 0usize, 0.0_f64);
    let blocks = 600; // 3.2 s
    for b in 0..blocks {
        let mut input = AudioBuffer::new(sources as u16, BLOCK as u16);
        for s in 0..sources {
            for i in 0..BLOCK {
                seeds[s] = seeds[s].wrapping_mul(1664525).wrapping_add(1013904223);
                let white = (seeds[s] >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                lps[s] += 0.2 * (white - lps[s]);
                input.set(s as u16, i as u16, 1.5 * lps[s]);
            }
        }
        e.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
        if b >= blocks - 100 {
            for i in 0..BLOCK {
                let mut p = 0.0_f64;
                for c in 0..channels {
                    let v = out.channel(c)[i] as f64;
                    p += v * v;
                }
                sum += p;
                sum_in += (input.channel(0)[i] as f64).powi(2);
                n += 1;
            }
        }
    }
    ((sum / n as f64).sqrt() as f32, (sum_in / n as f64).sqrt() as f32)
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// (direct, early, reverb) RMS, total power over channels.
fn levels(listener: [f32; 3], transmission: [f32; 8]) -> ([f32; 3], f32) {
    let mut lv = [0.0_f32; 3];
    let mut input = 1.0;
    for (i, stage) in [2u8, 3, 4].iter().enumerate() {
        let mut e = cathedral(PhysicalOutputLayout::Stereo, listener, transmission);
        let (o, inp) = render_rms(&mut e, *stage, 2, SPEAKERS.len());
        lv[i] = o;
        input = inp;
    }
    let early = (lv[1] * lv[1] - lv[0] * lv[0]).max(0.0).sqrt();
    let reverb = (lv[2] * lv[2] - lv[1] * lv[1]).max(0.0).sqrt();
    ([lv[0], early, reverb], input)
}

fn print(label: &str, l: [f32; 3], input: f32) {
    println!(
        "{label:<46} direct {:7.1} dB | early {:7.1} dB | reverb {:7.1} dB   (re one input channel, total power, stereo)",
        db(l[0] / input), db(l[1] / input), db(l[2] / input)
    );
}

#[test]
fn outside_levels_report_and_muffled_transmission_keeps_inside_balance() {
    let inside = [0.0, 1.6, 0.0];
    let outside = [-16.0, 1.6, 0.0]; // 5 m beyond the left wall at x = -11
    let (i0, in0) = levels(inside, [0.0; 8]);
    let (o0, on0) = levels(outside, [0.0; 8]);
    let (i1, in1) = levels(inside, PROPOSED_SHELL_TRANSMISSION);
    let (o1, on1) = levels(outside, PROPOSED_SHELL_TRANSMISSION);
    print("inside origin, transmission 0 (demo now)", i0, in0);
    print("inside origin, proposed transmission", i1, in1);
    print("5 m outside, transmission 0 (demo now)", o0, on0);
    print("5 m outside, proposed transmission", o1, on1);

    // Transmission must not change early / reverb inside. The direct level moves by a fraction
    // of a dB: speakers at y = 0.3 / 0.5 m have occlusion probe points below the floor, which
    // opaque walls count as blocked (a pre-existing soft-occlusion edge effect, see notes).
    for k in 0..3 {
        assert!((db(i0[k] / in0) - db(i1[k] / in1)).abs() < if k == 0 { 1.0 } else { 0.1 }, "inside level {k} changed: {i0:?} vs {i1:?}");
    }
    // Opaque walls: no direct sound outside. Proposed transmission: audible but far quieter
    // than the inside direct sound.
    assert!(db(o0[0] / on0) < -90.0, "direct outside with opaque walls: {} dB", db(o0[0] / on0));
    let muffled = db(o1[0] / on1);
    let inside_direct = db(i1[0] / in1);
    assert!(muffled > -80.0 && muffled < inside_direct - 20.0, "muffled direct {muffled} dB vs inside {inside_direct} dB");
}
