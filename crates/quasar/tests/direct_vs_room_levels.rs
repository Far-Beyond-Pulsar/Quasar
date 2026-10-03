//! Diagnostic: how loud is the DIRECT sound compared with the early reflections and the late
//! reverb in the demo's cathedral, for an emitter close to the listener?
//!
//! Noise is rendered through debug stages 2 (direct only) / 3 (+ early) / 4 (full) and the TOTAL
//! power over all output channels is printed.  A close emitter must be dominated by the direct
//! sound.

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

fn quad(id: u64, p: [[f32; 3]; 4], mat: u32) -> AcousticMesh {
    AcousticMesh::new(id, p.to_vec(), vec![0, 1, 2, 0, 2, 3], mat)
}

/// The demo's closed cathedral shell + probe grid + HybridBlend, one emitter per entry of
/// `emitters` (emitter `i` pulls channel `i` of a `emitters.len()`-channel source).
fn cathedral(layout: PhysicalOutputLayout, listener: [f32; 3], emitters: &[[f32; 3]], aim: Option<([f32; 3], f32)>) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let mat = |e: &SpatialAudioEngine, a: [f32; 8]| {
        e.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::new(a), Band8::zeros(), Band8::zeros()),
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
    e.set_probe_grid(
        AcousticProbeGrid::new(probes, [-12.0, 0.0, -28.0], [6.0, 4.0, 7.0], [5, 5, 9]).unwrap(),
    );
    e.set_strategy(HybridSamplingStrategy::HybridBlend);

    let src = e
        .load_source(SourceConfig { path: "n.wav".into(), channels: emitters.len() })
        .unwrap();
    for (i, &pos) in emitters.iter().enumerate() {
        let out = e.add_scene_output(SceneOutputConfig::new(pos, Movability::Static));
        e.connect_pull(out, ChannelPull::new(src, i as u32, 0.0));
        if let Some((target, directivity)) = aim {
            let d = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            e.set_scene_output_directivity(out, Some([d[0] / l, d[1] / l, d[2] / l]), directivity);
        }
    }
    e.add_listener(ListenerConfig { position: listener, heading: [0.0, 0.0, -1.0], physical_layout: layout });
    e.update_scene_spatial();
    e
}

/// RMS of the TOTAL power over all output channels (last 100 of 600 blocks), plus the RMS of one
/// input channel for reference.
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
                lps[s] += 0.2 * (white - lps[s]); // gentle low-pass: roughly music-like spectrum
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

/// Returns (direct, early, reverb) levels in dB re ONE input channel's RMS, total power over channels.
fn report(label: &str, layout: PhysicalOutputLayout, ch: u16, listener: [f32; 3], emitters: &[[f32; 3]], aim: Option<([f32; 3], f32)>) -> (f32, f32, f32) {
    let mut lv = [0.0_f32; 3];
    let mut input = 1.0;
    for (i, stage) in [2u8, 3, 4].iter().enumerate() {
        let mut e = cathedral(layout.clone(), listener, emitters, aim);
        let (o, inp) = render_rms(&mut e, *stage, ch, emitters.len());
        lv[i] = o;
        input = inp;
    }
    let direct = lv[0];
    let early = (lv[1] * lv[1] - lv[0] * lv[0]).max(0.0).sqrt();
    let reverb = (lv[2] * lv[2] - lv[1] * lv[1]).max(0.0).sqrt();
    println!(
        "{label:<34} direct {:6.1} dB | early {:6.1} dB | reverb {:6.1} dB   (re one input channel)   early-direct {:+5.1} | reverb-direct {:+5.1} dB",
        db(direct / input), db(early / input), db(reverb / input), db(early) - db(direct), db(reverb) - db(direct),
    );
    (direct, early, reverb)
}

#[test]
fn close_emitter_is_dominated_by_direct_sound() {
    // One emitter at increasing distance straight ahead of the listener.
    for d in [1.0_f32, 2.0, 4.0, 12.0] {
        report(&format!("1 emitter @ {d:>4} m  (stereo)"), PhysicalOutputLayout::Stereo, 2, [0.0, 1.6, 0.0], &[[0.0, 1.6, -d]], None);
    }
    // The demo: all 8 speakers play uncorrelated signals.  Listener 1.5 m in front of the
    // centre speaker (speaker C is at (0, 3, -12)) and at the origin (12 m from the front stage).
    for (label, listener) in [("DEMO 8 spk, 2 m from centre spk", [0.0, 1.6, -10.5]), ("DEMO 8 spk, at origin (12 m)", [0.0, 1.6, 0.0])] {
        report(&format!("{label} (stereo)"), PhysicalOutputLayout::Stereo, 2, listener, &SPEAKERS, None);
        report(&format!("{label} (7.1)"), PhysicalOutputLayout::Surround714, 8, listener, &SPEAKERS, None);
    }
}

#[test]
fn aimed_speakers_improve_the_direct_to_reverb_balance() {
    // Same demo scenario with every stage speaker aimed at the audience (the origin).
    let target = [0.0_f32, 1.6, 0.0];
    for (label, listener) in [("2 m from centre spk", [0.0, 1.6, -10.5]), ("at origin (12 m)", [0.0, 1.6, 0.0])] {
        for d in [0.0_f32, 0.5, 1.0] {
            report(&format!("DEMO {label} directivity {d:.1}"), PhysicalOutputLayout::Stereo, 2, listener, &SPEAKERS, Some((target, d)));
        }
    }
}
