//! Spatialised early reflections (#58): each reflection tap is rendered at its own
//! arrival direction (rotated by the listener heading) through the listener's
//! decoder, instead of being folded to mono and panned from the direct azimuth.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, ListenerId, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const C: f32 = 343.0;
const BLOCK: usize = 256;
const SETTLE: usize = 24;
const TAIL_BLOCKS: usize = 90;

/// Emitter at the origin, listener at (0, 0, -4); a single wall at x = `wall_x`
/// (normal along x) gives one first-order reflection with the bounce at
/// (wall_x, 0, -2): world direction from the listener (wall_x, 0, 2).
fn build(layout: PhysicalOutputLayout, wall_x: f32, heading: [f32; 3]) -> (SpatialAudioEngine, ListenerId) {
    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let wall = engine.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::zeros(), Band8::zeros(), Band8::zeros()),
    ));
    let mut scene = AcousticScene::new();
    scene.add_mesh(AcousticMesh::new(
        1,
        vec![[wall_x, -30.0, -30.0], [wall_x, 30.0, -30.0], [wall_x, 30.0, 30.0], [wall_x, -30.0, 30.0]],
        vec![0, 1, 2, 0, 2, 3],
        wall,
    ));
    let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
    engine.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let src = engine.load_source(SourceConfig { path: "imp.wav".into(), channels: 1 }).expect("source");
    let out = engine.add_scene_output(SceneOutputConfig::new([0.0, 0.0, 0.0], Movability::Static));
    engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    let lis = engine.add_listener(ListenerConfig { position: [0.0, 0.0, -4.0], heading, physical_layout: layout });
    engine.update_scene_spatial();
    (engine, lis)
}

fn channels_of(layout: &PhysicalOutputLayout) -> u16 {
    match layout {
        PhysicalOutputLayout::Stereo | PhysicalOutputLayout::Hrtf => 2,
        PhysicalOutputLayout::Quad => 4,
        PhysicalOutputLayout::Surround51 => 6,
        _ => 8,
    }
}

/// Unit impulse at the start of block SETTLE; per-channel output (stage 3).
fn impulse_response(engine: &mut SpatialAudioEngine, channels: u16) -> Vec<Vec<f32>> {
    engine.debug_audio_stage = 3;
    let silence = AudioBuffer::new(1, BLOCK as u16);
    let mut imp = AudioBuffer::new(1, BLOCK as u16);
    imp.set(0, 0, 1.0);
    let mut out = AudioBuffer::new(channels, BLOCK as u16);
    let mut res = vec![Vec::new(); channels as usize];
    for b in 0..(SETTLE + TAIL_BLOCKS) {
        out.clear();
        let input = if b == SETTLE { &imp } else { &silence };
        engine.process_audio_scene(&[input], std::slice::from_mut(&mut out));
        if b >= SETTLE {
            for (c, r) in res.iter_mut().enumerate() {
                r.extend_from_slice(&out.channel(c as u16)[..BLOCK]);
            }
        }
    }
    res
}

/// Per-channel energy over the reflection window (path 2 sqrt(wall_x^2 + 4) m).
fn reflection_energies(ir: &[Vec<f32>], wall_x: f32) -> Vec<f32> {
    let path = 2.0 * (wall_x * wall_x + 4.0).sqrt();
    let centre = (path * SR / C).round() as usize;
    ir.iter().map(|ch| ch[centre - 100..centre + 100].iter().map(|v| v * v).sum()).collect()
}

fn run_layout(layout: PhysicalOutputLayout, wall_x: f32, heading: [f32; 3]) -> Vec<f32> {
    let ch = channels_of(&layout);
    let (mut engine, _) = build(layout, wall_x, heading);
    reflection_energies(&impulse_response(&mut engine, ch), wall_x)
}

#[test]
fn lateral_reflection_is_localised_on_its_own_side_stereo() {
    // Listener faces +z, so +x is on its LEFT: the wall at x = +5 reflects from the left.
    let left_wall = run_layout(PhysicalOutputLayout::Stereo, 5.0, [0.0, 0.0, 1.0]);
    assert!(left_wall[0] > 20.0 * left_wall[1], "wall on the left: L {} R {}", left_wall[0], left_wall[1]);
    // Mirror image: wall at -5 reflects from the right.
    let right_wall = run_layout(PhysicalOutputLayout::Stereo, -5.0, [0.0, 0.0, 1.0]);
    assert!(right_wall[1] > 20.0 * right_wall[0], "wall on the right: L {} R {}", right_wall[0], right_wall[1]);
    // And it is NOT panned from the direct azimuth (dead ahead = equal channels).
    assert!((left_wall[0] / left_wall[1].max(1e-12)) > 5.0);
}

#[test]
fn lateral_reflection_is_localised_on_its_own_side_51_and_quad() {
    // 5.1: FL FR C LFE BL BR. Bounce direction (5, 0, 2) seen from a +z-facing listener is
    // 68 deg to the left: only the left pair (FL, BL) carries it.
    let e = run_layout(PhysicalOutputLayout::Surround51, 5.0, [0.0, 0.0, 1.0]);
    let (left, right) = (e[0] + e[4], e[1] + e[5]);
    assert!(left > 20.0 * right, "5.1 left pair {left} vs right pair {right}");
    assert!(e[3] == 0.0, "LFE never receives panned signal");
    assert!(e[2] < 0.01 * left, "centre must stay quiet for a 68 degree reflection: {}", e[2]);

    // Quad: FL FR BL BR. 68 deg left lies between FL (-45) and BL (-135), closer to FL.
    let q = run_layout(PhysicalOutputLayout::Quad, 5.0, [0.0, 0.0, 1.0]);
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(argmax(&q), 0, "FL must dominate: {q:?}");
    assert!(q[0] + q[2] > 20.0 * (q[1] + q[3]), "left side only: {q:?}");
}

#[test]
fn reflection_arrival_direction_rotates_with_the_listener_heading() {
    // World bounce direction (5, 0, 2). Facing +z it is 68 deg LEFT; facing +x it is
    // 22 deg RIGHT (+z is the right-hand side of a +x-facing listener).
    let facing_z = run_layout(PhysicalOutputLayout::Quad, 5.0, [0.0, 0.0, 1.0]);
    let facing_x = run_layout(PhysicalOutputLayout::Quad, 5.0, [1.0, 0.0, 0.0]);
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(argmax(&facing_z), 0, "{facing_z:?}");
    assert_eq!(argmax(&facing_x), 1, "FR must dominate when the reflection is 22 deg right: {facing_x:?}");
    assert!(facing_x[1] + facing_x[3] > 5.0 * (facing_x[0] + facing_x[2]) * 0.2, "{facing_x:?}");
    assert!(facing_x[0] + facing_x[1] > 5.0 * (facing_x[2] + facing_x[3]), "front pair dominates at 22 deg: {facing_x:?}");

    // Turning the listener in place moves the reflection between the stereo channels.
    let (mut engine, id) = build(PhysicalOutputLayout::Stereo, 5.0, [0.0, 0.0, 1.0]);
    let a = reflection_energies(&impulse_response(&mut engine, 2), 5.0);
    engine.update_listener(id, [0.0, 0.0, -4.0], [1.0, 0.0, 0.0]);
    engine.update_scene_spatial();
    let b = reflection_energies(&impulse_response(&mut engine, 2), 5.0);
    assert!(a[0] > 20.0 * a[1], "before turning: {a:?}");
    assert!(b[1] > b[0], "after turning to face the wall the reflection is on the right: {b:?}");
}

#[test]
fn hrtf_listener_localises_the_reflection_with_itd_and_ild() {
    // Wall on the left (68 deg): the left ear is louder AND leads the right ear.
    let (mut engine, _) = build(PhysicalOutputLayout::Hrtf, 5.0, [0.0, 0.0, 1.0]);
    let ir = impulse_response(&mut engine, 2);
    let path = 2.0 * (25.0_f32 + 4.0).sqrt();
    let centre = (path * SR / C).round() as usize;
    let win = centre - 100..centre + 100;
    let energy = |ch: &Vec<f32>| ch[win.clone()].iter().map(|v| v * v).sum::<f32>();
    let (el, er) = (energy(&ir[0]), energy(&ir[1]));
    assert!(el > 1.3 * er, "left ear must be louder: {el} vs {er}");
    let peak = |ch: &Vec<f32>| {
        win.clone().max_by(|&a, &b| ch[a].abs().partial_cmp(&ch[b].abs()).unwrap()).unwrap() as i32
    };
    let itd = peak(&ir[1]) - peak(&ir[0]);
    assert!((10..=45).contains(&itd), "left ear must lead by ~0.3-0.6 ms (Woodworth ~26 samples at 68 deg): {itd}");

    // Mirror: wall on the right swaps the ears.
    let (mut engine, _) = build(PhysicalOutputLayout::Hrtf, -5.0, [0.0, 0.0, 1.0]);
    let ir = impulse_response(&mut engine, 2);
    assert!(energy(&ir[1]) > 1.3 * energy(&ir[0]));
    assert!(peak(&ir[0]) - peak(&ir[1]) >= 10);
}

// ── no click when taps appear / disappear ─────────────────────────────

fn tone_block(buf: &mut AudioBuffer, phase: &mut f32, hz: f32, amp: f32) {
    let inc = hz / SR * std::f32::consts::TAU;
    for i in 0..BLOCK {
        buf.set(0, i as u16, amp * phase.sin());
        *phase = (*phase + inc) % std::f32::consts::TAU;
    }
}

/// Renders `blocks` of a 300 Hz tone with the SAME engine settings at stage 2 and stage 3
/// and returns the reflection-only signal `stage3 - stage2` (channel `ch`). `between`
/// is called before every block with the block index (to move the listener, toggle
/// geometry, ...) on both engines.
fn reflection_only(
    blocks: usize,
    channels: u16,
    layout: PhysicalOutputLayout,
    mut between: impl FnMut(usize, &mut SpatialAudioEngine),
) -> Vec<Vec<f32>> {
    let (mut e2, _) = build(layout.clone(), 5.0, [0.0, 0.0, 1.0]);
    let (mut e3, _) = build(layout, 5.0, [0.0, 0.0, 1.0]);
    e2.debug_audio_stage = 2;
    e3.debug_audio_stage = 3;
    let mut src = AudioBuffer::new(1, BLOCK as u16);
    let mut phase = 0.0_f32;
    let mut o2 = AudioBuffer::new(channels, BLOCK as u16);
    let mut o3 = AudioBuffer::new(channels, BLOCK as u16);
    let mut res = vec![Vec::new(); channels as usize];
    for b in 0..blocks {
        between(b, &mut e2);
        between(b, &mut e3);
        tone_block(&mut src, &mut phase, 300.0, 0.5);
        o2.clear();
        o3.clear();
        e2.process_audio_scene(&[&src], std::slice::from_mut(&mut o2));
        e3.process_audio_scene(&[&src], std::slice::from_mut(&mut o3));
        for (c, r) in res.iter_mut().enumerate() {
            for i in 0..BLOCK {
                r.push(o3.channel(c as u16)[i] - o2.channel(c as u16)[i]);
            }
        }
    }
    res
}

fn max_step(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
}

#[test]
fn reflections_fade_in_and_out_without_clicks() {
    // The tone is 0.5 at 300 Hz: a steady tap of gain g steps by at most 2 pi f / fs * 0.5 g
    // per sample. Moving the listener each few blocks makes the tracer retarget; the
    // reflection must never step by more than ~1.5x that steady bound.
    let blocks = 160;
    let sig = reflection_only(blocks, 2, PhysicalOutputLayout::Stereo, |b, e| {
        // Walk the listener back and forth along z (retargets taps every 2 blocks).
        if b % 2 == 0 {
            let z = -4.0 + 1.5 * ((b as f32) * 0.07).sin();
            e.update_listener(ListenerId(0), [0.0, 0.0, z], [0.0, 0.0, 1.0]);
            e.update_scene_spatial();
        }
    });
    let amp = sig[0].iter().chain(sig[1].iter()).fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(amp > 0.01, "the reflection must be present: {amp}");
    let steady = std::f32::consts::TAU * 300.0 / SR * amp;
    for (c, ch) in sig.iter().enumerate() {
        let s = max_step(&ch[BLOCK * 30..]);
        assert!(s < 2.0 * steady + 1e-4, "channel {c}: step {s} vs steady bound {steady}");
    }
}

#[test]
fn a_tap_that_appears_or_vanishes_does_not_click() {
    // Hard geometry change: swap the scene for one WITHOUT the wall (the single
    // reflection vanishes) and back (it appears). Each change crossfades.
    let wall_scene = |with_wall: bool, e: &mut SpatialAudioEngine| {
        let wall = e.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::zeros(), Band8::zeros(), Band8::zeros()),
        ));
        let mut scene = AcousticScene::new();
        if with_wall {
            scene.add_mesh(AcousticMesh::new(
                1,
                vec![[5.0, -30.0, -30.0], [5.0, 30.0, -30.0], [5.0, 30.0, 30.0], [5.0, -30.0, 30.0]],
                vec![0, 1, 2, 0, 2, 3],
                wall,
            ));
        }
        let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
        e.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
        e.update_scene_spatial();
    };
    let sig = reflection_only(150, 2, PhysicalOutputLayout::Stereo, |b, e| {
        if b == 40 {
            wall_scene(false, e);
        }
        if b == 90 {
            wall_scene(true, e);
        }
    });
    let amp = sig[0].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(amp > 0.01, "wall present at the start: {amp}");
    // The signal must end up silent while the wall is gone (blocks 60..85) ...
    let gone: f32 = sig[0][BLOCK * 60..BLOCK * 85].iter().fold(0.0, |m, v| m.max(v.abs()));
    assert!(gone < 1e-3, "reflection must have faded out completely: {gone}");
    // ... and come back after block 90.
    let back: f32 = sig[0][BLOCK * 120..].iter().fold(0.0, |m, v| m.max(v.abs()));
    assert!(back > 0.5 * amp, "reflection must return: {back} vs {amp}");
    let steady = std::f32::consts::TAU * 300.0 / SR * amp;
    for (c, ch) in sig.iter().enumerate() {
        let s = max_step(&ch[BLOCK * 20..]);
        assert!(s < 2.0 * steady + 1e-4, "channel {c}: step {s} vs steady bound {steady}");
    }
}

// ── cost ──────────────────────────────────────────────────────────────

/// Shoebox with 6 walls so every output has a full set of reflections.
fn shoebox_scene(mat: u32) -> AcousticScene {
    let (lx, ly, lz) = (10.0_f32, 4.0, 8.0);
    let mut s = AcousticScene::new();
    let quads: [[[f32; 3]; 4]; 6] = [
        [[0.0, 0.0, 0.0], [0.0, ly, 0.0], [0.0, ly, lz], [0.0, 0.0, lz]],
        [[lx, 0.0, 0.0], [lx, ly, 0.0], [lx, ly, lz], [lx, 0.0, lz]],
        [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, 0.0, lz], [0.0, 0.0, lz]],
        [[0.0, ly, 0.0], [lx, ly, 0.0], [lx, ly, lz], [0.0, ly, lz]],
        [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, ly, 0.0], [0.0, ly, 0.0]],
        [[0.0, 0.0, lz], [lx, 0.0, lz], [lx, ly, lz], [0.0, ly, lz]],
    ];
    for (i, q) in quads.iter().enumerate() {
        s.add_mesh(AcousticMesh::new(i as u64 + 1, q.to_vec(), vec![0, 1, 2, 0, 2, 3], mat));
    }
    s
}

#[test]
fn cost_report_8_outputs_x_16_taps() {
    // Printed per-block cost (run with `cargo test --release -p quasar-audio
    // cost_report -- --nocapture`). Asserts a generous bound only in release.
    for (name, layout) in [
        ("stereo", PhysicalOutputLayout::Stereo),
        ("5.1", PhysicalOutputLayout::Surround51),
        ("hrtf", PhysicalOutputLayout::Hrtf),
    ] {
        let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
        engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
        let wall = engine.materials().add_instance(AcousticMaterialInstance::new(
            TABULAR_MODEL_ID,
            Tabular8BandEvaluator::create_params(Band8::splat(0.2), Band8::zeros(), Band8::zeros()),
        ));
        let cfg = CpuSimdConfig { max_reflection_order: 3, max_reflections: 16, ..CpuSimdConfig::default() };
        engine.set_backend(Box::new(CpuSimdComputeBackend::new(shoebox_scene(wall), cfg)));
        engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
        let src = engine.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).expect("source");
        for o in 0..8 {
            let out = engine.add_scene_output(SceneOutputConfig::new(
                [1.5 + o as f32, 1.0 + 0.2 * o as f32, 2.0 + 0.5 * o as f32],
                Movability::Static,
            ));
            engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
        }
        let channels = channels_of(&layout);
        engine.add_listener(ListenerConfig { position: [5.0, 1.6, 4.0], heading: [0.0, 0.0, -1.0], physical_layout: layout });
        engine.update_scene_spatial();

        let mut input = AudioBuffer::new(1, BLOCK as u16);
        let mut phase = 0.0;
        tone_block(&mut input, &mut phase, 220.0, 0.3);
        let mut out = AudioBuffer::new(channels, BLOCK as u16);
        let mut time = |engine: &mut SpatialAudioEngine, stage: u8| -> f64 {
            engine.debug_audio_stage = stage;
            for _ in 0..40 {
                out.clear();
                engine.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
            }
            let reps = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                out.clear();
                engine.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
            }
            t0.elapsed().as_secs_f64() / reps as f64
        };
        let direct = time(&mut engine, 2);
        let with_refl = time(&mut engine, 3);
        let block_ms = BLOCK as f64 / SR as f64 * 1e3;
        println!(
            "reflection cost [{name}]: 8 outputs x 16 taps: direct only {:.3} ms/block, +reflections {:.3} ms/block (reflections {:.3} ms = {:.1}% of the {:.2} ms block)",
            direct * 1e3,
            with_refl * 1e3,
            (with_refl - direct) * 1e3,
            (with_refl - direct) / (block_ms / 1e3) * 100.0,
            block_ms
        );
        if !cfg!(debug_assertions) {
            assert!(with_refl - direct < 0.5 * block_ms / 1e3, "reflections must stay well below the block budget");
        }
    }
}
