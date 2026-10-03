//! Shared reverb bus (#62): one FDN per listener, per-output sends that do not
//! follow the direct distance attenuation, decoded diffusely.

use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::Movability;
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Engine with the geometry-free stub backend: direct path = distance law only, and a
/// constant late field (T60 0.5 s, -10 dB), so these tests isolate the reverb bus.
fn engine_with(layout: PhysicalOutputLayout, emitters: &[[f32; 3]], listener: [f32; 3]) -> SpatialAudioEngine {
    let mut engine = SpatialAudioEngine::new(0, SR, 15.0);
    engine.set_backend(Box::new(HardwareAcceleratorStub::new()));
    engine.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let src = engine.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).expect("source");
    for &p in emitters {
        let out = engine.add_scene_output(SceneOutputConfig::new(p, Movability::Static));
        engine.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    }
    engine.add_listener(ListenerConfig { position: listener, heading: [0.0, 0.0, -1.0], physical_layout: layout });
    engine.update_scene_spatial();
    engine
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

/// Render `blocks` blocks of white noise (RMS ~ 0.3) at `stage`; returns per-channel samples.
fn render(engine: &mut SpatialAudioEngine, stage: u8, channels: u16, blocks: usize) -> Vec<Vec<f32>> {
    engine.debug_audio_stage = stage;
    let mut rng = Noise(0x1234_5678);
    let mut src = AudioBuffer::new(1, BLOCK as u16);
    let mut out = AudioBuffer::new(channels, BLOCK as u16);
    let mut res = vec![Vec::new(); channels as usize];
    for _ in 0..blocks {
        for i in 0..BLOCK {
            src.set(0, i as u16, 0.5 * rng.next());
        }
        out.clear();
        engine.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        for (c, r) in res.iter_mut().enumerate() {
            r.extend_from_slice(&out.channel(c as u16)[..BLOCK]);
        }
    }
    res
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Reverb-only signal = stage 4 minus stage 3 (same engine config and input), after the
/// first `skip` blocks.
fn reverb_only(mk: &dyn Fn() -> SpatialAudioEngine, channels: u16, blocks: usize, skip: usize) -> Vec<Vec<f32>> {
    let a = render(&mut mk(), 4, channels, blocks);
    let b = render(&mut mk(), 3, channels, blocks);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x[skip * BLOCK..].iter().zip(y[skip * BLOCK..].iter()).map(|(p, q)| p - q).collect())
        .collect()
}

fn db(r: f32) -> f32 {
    20.0 * r.max(1e-12).log10()
}

#[test]
fn reverb_node_count_is_per_listener_not_per_emitter() {
    for n in [1usize, 4, 16] {
        let emitters: Vec<[f32; 3]> = (0..n).map(|i| [-10.0 + i as f32, 1.5, -20.0 + i as f32]).collect();
        let e = engine_with(PhysicalOutputLayout::Stereo, &emitters, [0.0, 1.6, 20.0]);
        assert_eq!(e.reverb_node_count(), 1, "{n} emitters, one listener");
    }
    let mut e = engine_with(PhysicalOutputLayout::Stereo, &[[0.0, 1.5, 0.0]; 8], [0.0, 1.6, 20.0]);
    e.add_listener(ListenerConfig {
        position: [3.0, 1.6, 10.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Surround51,
    });
    assert_eq!(e.reverb_node_count(), 2, "8 emitters, two listeners");
}

/// Documented tolerance: the late-field level of an emitter must stay within this many dB
/// when only the emitter-listener distance changes (the diffuse field is distance independent
/// to first order; the model's own distance term is the Barron late-energy factor
/// `exp(-0.04 r / T)` which is below 2 dB over 20 m for reverberation times above 2 s).
const REVERB_DISTANCE_TOLERANCE_DB: f32 = 3.0;

#[test]
fn moving_an_emitter_away_changes_the_direct_level_but_not_the_reverberant_level() {
    let listener = [0.0, 1.6, 25.0];
    let level = |emitter: [f32; 3]| {
        let mk = || engine_with(PhysicalOutputLayout::Stereo, &[emitter], listener);
        let direct = render(&mut mk(), 2, 2, 120);
        let rev = reverb_only(&mk, 2, 120, 60);
        let d = rms(&direct[0][60 * BLOCK..]).hypot(rms(&direct[1][60 * BLOCK..]));
        let r = rms(&rev[0]).hypot(rms(&rev[1]));
        (d, r)
    };
    let (d_near, r_near) = level([0.0, 1.6, 20.0]); // 5 m
    let (d_far, r_far) = level([0.0, 1.6, -15.0]); // 40 m
    // The direct sound falls by the distance law (8x = 18 dB) ...
    assert!(db(d_near / d_far) > 12.0, "direct {} -> {} ({} dB)", d_near, d_far, db(d_near / d_far));
    // ... the reverberant level does not.
    println!("direct {d_near:.4} -> {d_far:.4}; reverb {r_near:.4} -> {r_far:.4}");
    assert!(r_near > 1e-3 && r_far > 1e-3, "reverb must be present: {r_near} {r_far}");
    assert!(
        db(r_near / r_far).abs() < REVERB_DISTANCE_TOLERANCE_DB,
        "reverberant level moved by {} dB with distance (tolerance {REVERB_DISTANCE_TOLERANCE_DB} dB)",
        db(r_near / r_far)
    );
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let (mut ab, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b.iter()) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64) * (*x as f64);
        bb += (*y as f64) * (*y as f64);
    }
    (ab / (aa * bb).sqrt().max(1e-30)) as f32
}

/// Documented bound on the zero-lag inter-channel correlation of the decoded reverb
/// (2 s of white-noise excitation): the Hadamard output tap sets are orthogonal, so the
/// channels are decorrelated copies of the same tail.
const REVERB_CORRELATION_BOUND: f32 = 0.15; // measured max 0.085 (5.1), 0.04 stereo / HRTF

#[test]
fn decoded_reverb_channels_are_decorrelated() {
    for (name, layout, channels, lfe) in [
        ("stereo", PhysicalOutputLayout::Stereo, 2u16, None),
        ("quad", PhysicalOutputLayout::Quad, 4, None),
        ("5.1", PhysicalOutputLayout::Surround51, 6, Some(3usize)),
        ("hrtf", PhysicalOutputLayout::Hrtf, 2, None),
    ] {
        // Emitter hard to the left: a panned send would put all reverb on the left.
        let mk = || engine_with(layout.clone(), &[[-15.0, 1.6, 10.0]], [0.0, 1.6, 25.0]);
        let rev = reverb_only(&mk, channels, 240, 60);
        let levels: Vec<f32> = rev.iter().map(|c| rms(c)).collect();
        for c in 0..channels as usize {
            if Some(c) == lfe {
                assert!(levels[c] == 0.0, "{name}: LFE must not receive reverb");
                continue;
            }
            assert!(levels[c] > 1e-5, "{name} ch {c}: reverb present: {}", levels[c]);
        }
        let mut max_rho = 0.0_f32;
        for a in 0..channels as usize {
            for b in a + 1..channels as usize {
                if Some(a) == lfe || Some(b) == lfe {
                    continue;
                }
                let rho = correlation(&rev[a], &rev[b]);
                max_rho = max_rho.max(rho.abs());
                assert!(
                    rho.abs() < REVERB_CORRELATION_BOUND,
                    "{name}: channels {a}/{b} correlation {rho} >= {REVERB_CORRELATION_BOUND}"
                );
            }
        }
        println!("{name}: max |rho| = {max_rho:.3}, channel RMS = {levels:?}");
        // Diffuse, not panned from the emitter: no channel dominates by more than 6 dB
        // (all non-LFE channels carry the same constant-power share).
        let live: Vec<f32> = levels.iter().enumerate().filter(|(c, _)| Some(*c) != lfe).map(|(_, v)| *v).collect();
        let (mx, mn) = (live.iter().cloned().fold(0.0, f32::max), live.iter().cloned().fold(f32::MAX, f32::min));
        assert!(db(mx / mn) < 6.0, "{name}: channel levels {levels:?}");
    }
}

#[test]
fn reverb_is_not_panned_from_the_emitter_direction() {
    // Direct path of a hard-left emitter is far louder on the left; the reverb is balanced.
    let mk = || engine_with(PhysicalOutputLayout::Stereo, &[[-8.0, 1.6, 17.0]], [0.0, 1.6, 25.0]);
    let direct = render(&mut mk(), 2, 2, 100);
    let dl = rms(&direct[0][40 * BLOCK..]);
    let dr = rms(&direct[1][40 * BLOCK..]);
    assert!(dl > 8.0 * dr, "direct is panned left: {dl} vs {dr}");
    let rev = reverb_only(&mk, 2, 160, 60);
    let (rl, rr) = (rms(&rev[0]), rms(&rev[1]));
    assert!(db(rl / rr).abs() < 1.5, "reverb must be balanced between L and R: {rl} vs {rr}");
}
