//! Level balance of the DIRECT sound against the early reflections and the late reverb: a printed
//! diagnostic table of the demo's cathedral plus regression tests with assertions.
//!
//! Noise is rendered through debug stages 2 (direct only) / 3 (+ early) / 4 (full) and the TOTAL
//! power over all output channels is measured, in dB re ONE input channel.
//!
//! Run with `cargo test -p quasar-audio --release --test direct_vs_room_levels -- --nocapture`.
//!
//! * `reverb_level_matches_the_diffuse_field_formula`: the engine's calibrated reverb power equals
//!   `312.2 T60 / (V Q)` (re the 1 m direct sound) within 1 dB, omni and directional.
//! * `direct_path_is_not_attenuated_by_reverb_settings` / `direct_arrives_first_...`: the direct
//!   sound follows 1/d, is present in the right channels and arrives before every reflection.
//! * `demo_defaults_balance_...`: the artistic demo mix (aimed speakers + reverb trim) keeps the
//!   direct sound dominant near a speaker and the hall audible far away. Keep `DEMO_*` below in
//!   sync with `examples/basic/src/main.rs`.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_audio::quasar_core::source_directivity::diffuse_send_gain;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::limiter::OutputSafetyConfig;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Artistic demo mix (engine defaults stay physical): speakers aimed at the audience with this
/// directivity, and these trims of the reverb / early-reflection buses. See main.rs `DEMO_*`.
const DEMO_DIRECTIVITY: f32 = 0.7;
const DEMO_REVERB_DB: f32 = -5.0;
const DEMO_EARLY_DB: f32 = 0.0;
const AUDIENCE: [f32; 3] = [0.0, 1.6, 0.0];
/// Listener 2 m from the centre speaker (0, 3, -12), and at the demo's start position.
const NEAR: [f32; 3] = [0.0, 1.6, -10.5];
const ORIGIN: [f32; 3] = [0.0, 1.6, 0.0];

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
fn cathedral(
    layout: PhysicalOutputLayout,
    listener: [f32; 3],
    emitters: &[[f32; 3]],
    aim: Option<([f32; 3], f32)>,
    mix: (f32, f32),
) -> SpatialAudioEngine {
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
    let lid = e.add_listener(ListenerConfig { position: listener, heading: [0.0, 0.0, -1.0], physical_layout: layout });
    e.set_reverb_gain_db(lid, mix.0);
    e.set_early_reflection_gain_db(lid, mix.1);
    // Measure linear levels: the (transparent but peak-limiting) output limiter is off.
    e.set_output_safety(lid, OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
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
fn report(
    label: &str,
    layout: PhysicalOutputLayout,
    ch: u16,
    listener: [f32; 3],
    emitters: &[[f32; 3]],
    aim: Option<([f32; 3], f32)>,
    mix: (f32, f32),
) -> (f32, f32, f32) {
    let mut lv = [0.0_f32; 3];
    let mut input = 1.0;
    for (i, stage) in [2u8, 3, 4].iter().enumerate() {
        let mut e = cathedral(layout.clone(), listener, emitters, aim, mix);
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
        report(&format!("1 emitter @ {d:>4} m  (stereo)"), PhysicalOutputLayout::Stereo, 2, [0.0, 1.6, 0.0], &[[0.0, 1.6, -d]], None, (0.0, 0.0));
    }
    // The demo: all 8 speakers play uncorrelated signals.  Listener 1.5 m in front of the
    // centre speaker (speaker C is at (0, 3, -12)) and at the origin (12 m from the front stage).
    for (label, listener) in [("DEMO 8 spk, 2 m from centre spk", [0.0, 1.6, -10.5]), ("DEMO 8 spk, at origin (12 m)", [0.0, 1.6, 0.0])] {
        report(&format!("{label} (stereo)"), PhysicalOutputLayout::Stereo, 2, listener, &SPEAKERS, None, (0.0, 0.0));
        report(&format!("{label} (7.1)"), PhysicalOutputLayout::Surround714, 8, listener, &SPEAKERS, None, (0.0, 0.0));
    }
}

#[test]
#[ignore = "diagnostic directivity sweep (slow in debug): cargo test -p quasar-audio --release --test direct_vs_room_levels -- --ignored --nocapture"]
fn aimed_speakers_improve_the_direct_to_reverb_balance() {
    // Same demo scenario with every stage speaker aimed at the audience (the origin).
    let target = [0.0_f32, 1.6, 0.0];
    for (label, listener) in [("2 m from centre spk", [0.0, 1.6, -10.5]), ("at origin (12 m)", [0.0, 1.6, 0.0])] {
        for d in [0.0_f32, 0.5, 0.7, 0.85, 1.0] {
            report(&format!("DEMO {label} directivity {d:.1}"), PhysicalOutputLayout::Stereo, 2, listener, &SPEAKERS, Some((target, d)), (0.0, 0.0));
        }
    }
}

// ── assertions ───────────────────────────────────────────────────────────

fn amp_db(x: f32) -> f32 {
    db(x)
}

/// Engine with a constant probe grid (`dims` 3x3x3, `spacing` metres, so the grid volume is
/// `(2 spacing)^3`) and `BakedOnly` sampling: no ray tracing, no early reflections, the late
/// level comes straight from `late_loudness_from_t60_volume`. One emitter 1 m in front of the
/// listener at the grid centre; `directional` = `Some(directivity)` aims it at the listener.
fn baked_engine(t60: f32, spacing: f32, directional: Option<f32>) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_strategy(HybridSamplingStrategy::BakedOnly);
    let mut probes = Vec::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                probes.push(AcousticProbe {
                    position: [x as f32 * spacing, y as f32 * spacing, z as f32 * spacing],
                    rir_samples: Vec::new(),
                    sample_rate: 48000,
                    t60: Band8::splat(t60),
                    broadband_t60: t60,
                    early_late_split_secs: 0.05,
                });
            }
        }
    }
    e.set_probe_grid(AcousticProbeGrid::new(probes, [0.0; 3], [spacing; 3], [3, 3, 3]).unwrap());
    let c = spacing; // grid centre
    let src = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).unwrap();
    let out = e.add_scene_output(SceneOutputConfig::new([c, c, c - 1.0], Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    if let Some(d) = directional {
        e.set_scene_output_directivity(out, Some([0.0, 0.0, 1.0]), d); // faces the listener (+z)
    }
    let lid = e.add_listener(ListenerConfig { position: [c, c, c], heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.set_output_safety(lid, OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
    e.update_scene_spatial();
    e
}

/// Late-field power re the direct sound at 1 m, in dB, measured on the engine output.
fn measured_reverb_db(mk: impl Fn() -> SpatialAudioEngine) -> f32 {
    let (full, input) = render_rms(&mut mk(), 4, 2, 1);
    let (direct, _) = render_rms(&mut mk(), 2, 2, 1);
    let reverb = (full * full - direct * direct).max(0.0).sqrt();
    // direct (1 m, unit gain, constant-power pan) has total power input^2 -> 0 dB.
    assert!((amp_db(direct / input)).abs() < 0.5, "direct at 1 m must be 0 dB re the input: {}", amp_db(direct / input));
    amp_db(reverb / input)
}

#[test]
fn reverb_level_matches_the_diffuse_field_formula() {
    // rev/direct power = 312.2 T60 / (V Q): omni (Q = 1) over three rooms, then a cardioid-ish
    // emitter (Q = 1 / mean-square pattern = 1 / diffuse_send_gain^2).
    let mut bad: Vec<String> = Vec::new();
    for (t60, spacing) in [(2.0_f32, 5.0_f32), (4.0, 10.0), (7.0, 15.0), (1.0, 10.0), (4.0, 5.0), (7.0, 8.0)] {
        let v = (2.0 * spacing).powi(3);
        let want = 10.0 * (312.2 * t60 / v).log10();
        let got = measured_reverb_db(|| baked_engine(t60, spacing, None));
        println!("omni T60 {t60} s, V {v} m3: measured {got:.2} dB, formula {want:.2} dB");
        if (got - want).abs() >= 1.0 {
            bad.push(format!("omni T60 {t60} s, V {v} m3: measured {got:.2} dB, formula {want:.2} dB"));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
    let (t60, spacing, d) = (4.0_f32, 10.0_f32, 1.0_f32);
    let q_db = -20.0 * diffuse_send_gain(d).log10(); // 10 log10 Q
    let want = 10.0 * (312.2 * t60 / (2.0 * spacing).powi(3)).log10() - q_db;
    let got = measured_reverb_db(|| baked_engine(t60, spacing, Some(d)));
    assert!(q_db > 3.0, "a cardioid has Q ~ 4.8 dB (diffuse-field average), got {q_db:.2}");
    assert!((got - want).abs() < 1.0, "directional (d = {d}): measured {got:.2} dB, formula {want:.2} dB (Q = {q_db:.2} dB)");
}

#[test]
fn direct_path_follows_one_over_d_and_ignores_reverb_settings() {
    // Demo scene, one emitter (HybridBlend, CpuSimd). Direct sound: -6.02 dB per doubling, and
    // bit-identical whatever the reverb / early trims are.
    let mut levels = Vec::new();
    for d in [1.0_f32, 2.0, 4.0] {
        let mut e = cathedral(PhysicalOutputLayout::Stereo, ORIGIN, &[[0.0, 1.6, -d]], None, (0.0, 0.0));
        let (o, i) = render_rms(&mut e, 2, 2, 1);
        levels.push(amp_db(o / i));
    }
    for (k, want) in [0.0_f32, -6.02, -12.04].iter().enumerate() {
        // Air absorption adds a little at 4 m.
        assert!((levels[k] - want).abs() < 0.5, "direct at {} m: {:.2} dB, expected {want:.2}", 1 << k, levels[k]);
    }
    let (base, _) = render_rms(&mut cathedral(PhysicalOutputLayout::Stereo, ORIGIN, &[[0.0, 1.6, -2.0]], None, (0.0, 0.0)), 2, 2, 1);
    for mix in [(20.0_f32, 0.0_f32), (-120.0, -120.0), (-30.0, 12.0)] {
        let (o, _) = render_rms(&mut cathedral(PhysicalOutputLayout::Stereo, ORIGIN, &[[0.0, 1.6, -2.0]], None, mix), 2, 2, 1);
        assert_eq!(o, base, "the direct sound must not depend on the reverb / early trims {mix:?}");
    }
}

#[test]
fn direct_arrives_first_and_the_first_reflection_is_geometric() {
    // Demo room, 7.1 listener 2.05 m from the centre speaker, a unit impulse (nearest speaker only).
    let (spk, lis) = ([0.0_f32, 3.0, -12.0], NEAR);
    let mut e = cathedral(PhysicalOutputLayout::Surround714, lis, &[spk], None, (0.0, 0.0));
    e.debug_audio_stage = 3;
    let mut out = AudioBuffer::new(8, BLOCK as u16);
    let mut ch: Vec<Vec<f32>> = vec![Vec::new(); 8];
    let t0 = 2 * BLOCK; // impulse position
    for b in 0..12 {
        let mut input = AudioBuffer::new(1, BLOCK as u16);
        if b == 2 {
            input.set(0, 0, 1.0);
        }
        out.clear();
        e.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
        for c in 0..8 {
            ch[c].extend_from_slice(&out.channel(c as u16)[..BLOCK]);
        }
    }
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let c_sound = 343.0_f32;
    let d_direct = dist(spk, lis);
    // First reflection: the floor (image source below y = 0).
    let d_floor = dist([spk[0], -spk[1], spk[2]], lis);
    let n_direct = t0 + (d_direct * SR / c_sound).round() as usize;
    let n_floor = t0 + (d_floor * SR / c_sound).round() as usize;
    // Total power per sample.
    let p: Vec<f32> = (0..ch[0].len()).map(|i| ch.iter().map(|c| c[i] * c[i]).sum()).collect();
    let win = |c: usize, n: usize| -> f32 { ch[c][n - 4..n + 5].iter().map(|v| v * v).sum() };
    let pw = |n: usize| -> f32 { (0..8).map(|c| win(c, n)).sum() };
    // Silence before the direct arrival (allow the 4-sample interpolation spread).
    let pre: f32 = p[..n_direct - 6].iter().sum();
    assert!(pre < 1e-7, "nothing may arrive before the direct sound: {pre}");
    // Direct sound: 1/d amplitude, i.e. total energy 1/d^2.
    let e_direct = pw(n_direct);
    let want = 1.0 / (d_direct * d_direct);
    assert!((e_direct / want).log10().abs() * 10.0 < 1.0, "direct energy {e_direct} vs 1/d^2 = {want}");
    // The centre speaker (index 2) carries it; the LFE slot (index 3) none.
    assert!(win(2, n_direct) > 0.6 * e_direct, "front-centre emitter must land in the centre channel");
    assert!(win(3, n_direct) < 1e-9, "no direct sound in the LFE slot");
    // Floor reflection arrives where the image-source geometry says, after the direct sound, and
    // between the two arrivals (up to the floor bounce) there is no other energy.
    let e_floor = pw(n_floor);
    assert!(e_floor > 0.02 * e_direct, "floor reflection missing at sample {n_floor}: {e_floor}");
    let gap: f32 = p[n_direct + 6..n_floor - 6].iter().sum();
    assert!(gap < 1e-3 * e_direct, "unexpected energy between the direct sound and the floor bounce: {gap}");
    assert!(n_floor > n_direct + 100, "the reflection must follow the direct sound by the path difference");
}

#[test]
fn demo_defaults_balance_direct_early_and_reverb() {
    // The demo's artistic mix (see DEMO_*): aimed speakers, reverb trim. All 8 speakers play
    // uncorrelated noise, stereo listener (relative levels are layout independent).
    let aim = Some((AUDIENCE, DEMO_DIRECTIVITY));
    let mix = (DEMO_REVERB_DB, DEMO_EARLY_DB);
    let lin = |x: f32| (x * x) as f64;
    let (d, e, r) = report("DEMO MIX 2 m from centre speaker", PhysicalOutputLayout::Stereo, 2, NEAR, &SPEAKERS, aim, mix);
    let (direct_over_reverb, direct_over_early) = (10.0 * (lin(d) / lin(r)).log10(), 10.0 * (lin(d) / lin(e)).log10());
    assert!(direct_over_reverb >= 3.0, "near a speaker the direct sound must exceed the reverb by >= 3 dB: {direct_over_reverb:.1} dB");
    assert!(direct_over_early >= 3.0, "... and the early reflections by >= 3 dB: {direct_over_early:.1} dB");
    let direct_plus_early = 10.0 * ((lin(d) + lin(e)) / lin(r)).log10();
    assert!(direct_plus_early >= 4.0, "direct + early must clearly dominate the reverb: {direct_plus_early:.1} dB");

    let (d, _e, r) = report("DEMO MIX at the origin (12 m)", PhysicalOutputLayout::Stereo, 2, ORIGIN, &SPEAKERS, aim, mix);
    let reverb_over_direct = 10.0 * (lin(r) / lin(d)).log10();
    assert!(reverb_over_direct > 0.0, "far from the stage the hall must still be audible over the direct sound: {reverb_over_direct:.1} dB");
    assert!(reverb_over_direct <= 8.0, "... but not swamp it (was +15 dB before the demo mix): {reverb_over_direct:.1} dB");
}
