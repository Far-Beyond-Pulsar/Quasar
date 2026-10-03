//! Per-listener mix trims: `set_reverb_gain_db` / `set_early_reflection_gain_db`.
//!
//! Default (0 dB) is bit-identical to an engine that never heard of the trims, a trim changes
//! exactly its own component by the requested dB, it is applied with a per-sample ramp (no click)
//! and it never touches the direct sound.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
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
const BLOCK: usize = 256;

/// Stub backend (direct = 1/d, constant late field -10 dB, T60 0.5 s, no early reflections).
fn stub_engine() -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([0.0, 0.0, -2.0], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig { position: [0.0; 3], heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.update_scene_spatial();
    e
}

/// CPU backend with ONE wall at x = 4: a single strong first-order reflection.
fn wall_engine() -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let mat = e.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::zeros(), Band8::zeros(), Band8::zeros()),
    ));
    let mut scene = AcousticScene::new();
    scene.add_mesh(AcousticMesh::new(
        1,
        vec![[4.0, -30.0, -30.0], [4.0, 30.0, -30.0], [4.0, 30.0, 30.0], [4.0, -30.0, 30.0]],
        vec![0, 1, 2, 0, 2, 3],
        mat,
    ));
    let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
    e.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let s = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).expect("source");
    let o = e.add_scene_output(SceneOutputConfig::new([0.0; 3], Movability::Static));
    e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
    e.add_listener(ListenerConfig { position: [0.0, 0.0, -4.0], heading: [0.0, 0.0, 1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.update_scene_spatial();
    e
}

struct Noise(u32);
impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Render `blocks` blocks of noise (or a 300 Hz tone) at `stage`, calling `hook(block, engine)`
/// before each block; returns the L and R channels.
fn render(
    e: &mut SpatialAudioEngine,
    stage: u8,
    blocks: usize,
    tone: bool,
    mut hook: impl FnMut(usize, &mut SpatialAudioEngine),
) -> [Vec<f32>; 2] {
    e.debug_audio_stage = stage;
    let mut rng = Noise(0x2468_ace1);
    let mut src = AudioBuffer::new(1, BLOCK as u16);
    let mut out = AudioBuffer::new(2, BLOCK as u16);
    let mut res = [Vec::new(), Vec::new()];
    for b in 0..blocks {
        hook(b, e);
        for i in 0..BLOCK {
            let v = if tone {
                0.4 * (std::f32::consts::TAU * 300.0 * (b * BLOCK + i) as f32 / SR).sin()
            } else {
                0.4 * rng.next()
            };
            src.set(0, i as u16, v);
        }
        out.clear();
        e.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        for c in 0..2 {
            res[c].extend_from_slice(&out.channel(c as u16)[..BLOCK]);
        }
    }
    res
}

/// Mean power over both channels from sample `from` on.
fn power(x: &[Vec<f32>; 2], from: usize) -> f64 {
    let n = x[0].len() - from;
    x.iter().flat_map(|c| c[from..].iter()).map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64
}

fn db(p: f64) -> f64 {
    10.0 * p.max(1e-30).log10()
}

#[test]
fn defaults_are_zero_db_and_bit_identical() {
    let mut a = stub_engine();
    let mut b = stub_engine();
    assert_eq!(a.reverb_gain_db(ListenerId(0)), 0.0);
    assert_eq!(a.early_reflection_gain_db(ListenerId(0)), 0.0);
    b.set_reverb_gain_db(ListenerId(0), 0.0);
    b.set_early_reflection_gain_db(ListenerId(0), 0.0);
    let ya = render(&mut a, 4, 120, false, |_, _| {});
    let yb = render(&mut b, 4, 120, false, |_, _| {});
    assert!(ya[0] == yb[0] && ya[1] == yb[1], "an explicit 0 dB trim must not change a single sample");
    let mut c = wall_engine();
    let mut d = wall_engine();
    d.set_reverb_gain_db(ListenerId(0), 0.0);
    d.set_early_reflection_gain_db(ListenerId(0), 0.0);
    let yc = render(&mut c, 4, 120, false, |_, _| {});
    let yd = render(&mut d, 4, 120, false, |_, _| {});
    assert!(yc[0] == yd[0] && yc[1] == yd[1], "same with early reflections");
}

#[test]
fn reverb_trim_changes_only_the_reverb_by_the_requested_db() {
    let blocks = 260;
    let measure = |trim: f32| {
        let mut e = stub_engine();
        e.set_reverb_gain_db(ListenerId(0), trim);
        let full = render(&mut e, 4, blocks, false, |_, _| {});
        let mut e2 = stub_engine();
        e2.set_reverb_gain_db(ListenerId(0), trim);
        let direct = render(&mut e2, 2, blocks, false, |_, _| {});
        (power(&full, 100 * BLOCK), power(&direct, 100 * BLOCK))
    };
    let (full0, direct0) = measure(0.0);
    let reverb0 = full0 - direct0; // uncorrelated with the direct sound
    assert!(reverb0 > 0.0);
    for trim in [-6.0_f32, -12.0, 6.0] {
        let (full, direct) = measure(trim);
        assert!((direct - direct0).abs() <= 1e-12 * direct0, "the direct sound must not depend on the reverb trim");
        let got = db((full - direct).max(1e-30)) - db(reverb0);
        assert!((got - trim as f64).abs() < 0.3, "reverb trim {trim} dB: measured {got:.2} dB");
    }
    // Mute: the late field disappears (the direct sound stays).
    let (full, direct) = measure(-120.0);
    assert!((full - direct).abs() < 1e-6 * direct, "-120 dB mutes the reverb: {full} vs {direct}");
}

#[test]
fn early_trim_changes_only_the_early_reflections_by_the_requested_db() {
    let measure = |trim: f32| {
        let mut e = wall_engine();
        e.set_early_reflection_gain_db(ListenerId(0), trim);
        let s3 = render(&mut e, 3, 200, false, |_, _| {});
        let mut e2 = wall_engine();
        e2.set_early_reflection_gain_db(ListenerId(0), trim);
        let s2 = render(&mut e2, 2, 200, false, |_, _| {});
        (power(&s3, 100 * BLOCK), power(&s2, 100 * BLOCK))
    };
    let (p0, d0) = measure(0.0);
    let early0 = p0 - d0;
    assert!(early0 > 1e-6, "the wall must produce a reflection: {early0}");
    for trim in [-6.0_f32, -15.0, 4.0] {
        let (p, d) = measure(trim);
        assert!((d - d0).abs() <= 1e-12 * d0, "direct sound unchanged");
        let got = db(p - d) - db(early0);
        assert!((got - trim as f64).abs() < 0.5, "early trim {trim} dB: measured {got:.2} dB");
    }
}

fn max_step(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
}

#[test]
fn trim_changes_are_ramped_without_clicks() {
    // A 300 Hz tone through the full chain; at block 100 the reverb and early trims drop by 20 dB,
    // at block 130 they come back. The trim only acts on the reverb / reflection component, so the
    // DIFFERENCE to the untouched render is that component's own change: its largest
    // sample-to-sample step must stay near the component's natural tone slope (a block-constant
    // 20 dB step of a component this loud would be an order of magnitude larger).
    // (The raw output cannot be compared to the reference directly: a tone through the FDN has a
    // slowly varying envelope, so its own steps differ from block to block.)
    for wall in [false, true] {
        let mk = || if wall { wall_engine() } else { stub_engine() };
        let mut steady = mk();
        let reference = render(&mut steady, 4, 200, true, |_, _| {});
        let mut e = mk();
        let y = render(&mut e, 4, 200, true, |b, e| {
            let db = if (100..130).contains(&b) { -20.0 } else { 0.0 };
            e.set_reverb_gain_db(ListenerId(0), db);
            e.set_early_reflection_gain_db(ListenerId(0), db);
        });
        for c in 0..2 {
            let s_ref = max_step(&reference[c][60 * BLOCK..]);
            let diff: Vec<f32> = y[c].iter().zip(&reference[c]).map(|(a, b)| a - b).collect();
            let s = max_step(&diff[60 * BLOCK..]);
            assert!(s <= 1.0 * s_ref, "wall={wall} channel {c}: trimmed-component step {s} vs tone slope {s_ref}");
        }
    }
}

#[test]
fn trims_are_per_listener_clamped_and_ignore_nan() {
    let mut e = stub_engine();
    let l2 = e.add_listener(ListenerConfig { position: [0.0, 0.0, -1.0], heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.set_reverb_gain_db(l2, -9.0);
    e.set_reverb_gain_db(l2, f32::NAN);
    assert_eq!(e.reverb_gain_db(l2), -9.0);
    assert_eq!(e.reverb_gain_db(ListenerId(0)), 0.0);
    e.set_early_reflection_gain_db(ListenerId(0), 500.0);
    assert_eq!(e.early_reflection_gain_db(ListenerId(0)), 24.0, "clamped to +24 dB");
}

