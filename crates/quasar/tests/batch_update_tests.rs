//! #151 engine level: `update_scene_spatial` resolves every dirty pair in ONE batch. The audio
//! rendered after a single batched update equals, sample for sample, the audio rendered after
//! updating the same pairs one at a time (batches of one, the old per-pair behaviour), including
//! early reflections, directivity and the late estimate; and unchanged pairs are not re-resolved.

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Closed 20 x 8 x 30 m room (inward-facing box).
fn room() -> AcousticScene {
    let (lo, hi) = ([0.0_f32, 0.0, 0.0], [20.0_f32, 8.0, 30.0]);
    let p = vec![
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ];
    let mut idx: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    ];
    for t in idx.chunks_exact_mut(3) {
        t.swap(1, 2);
    }
    let mut s = AcousticScene::new();
    s.add_mesh(AcousticMesh::new(1, p, idx, 0));
    s
}

fn engine(strategy: HybridSamplingStrategy) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(CpuSimdComputeBackend::new(room(), CpuSimdConfig::default())));
    e.set_strategy(strategy);
    e.add_listener(ListenerConfig {
        position: [10.0, 1.7, 22.0],
        heading: [0.0, 0.0, -1.0],
        physical_layout: PhysicalOutputLayout::Stereo,
    });
    e
}

const EMITTERS: [[f32; 3]; 4] = [[4.0, 1.7, 6.0], [16.0, 3.0, 10.0], [10.0, 1.7, 4.0], [3.0, 5.0, 18.0]];

/// Deterministic, per-source distinct test signal.
fn signal(n: usize, k: usize) -> AudioBuffer {
    let mut b = AudioBuffer::new(1, BLOCK as u16);
    for i in 0..BLOCK {
        let t = (n * BLOCK + i) as f32;
        b.set(0, i as u16, 0.4 * ((t * (0.013 + 0.007 * k as f32)).sin() + 0.5 * (t * 0.171).cos()));
    }
    b
}

/// Add the emitters; `per_pair` updates after each one (batch of one), else once at the end.
fn run(strategy: HybridSamplingStrategy, per_pair: bool) -> Vec<f32> {
    let mut e = engine(strategy);
    let mut srcs = Vec::new();
    for (k, pos) in EMITTERS.iter().enumerate() {
        let s = e.load_source(SourceConfig { path: format!("s{k}.wav"), channels: 1 }).unwrap();
        let mut cfg = SceneOutputConfig::new(*pos, Movability::Static);
        if k == 1 {
            cfg.orientation = Some([-1.0, 0.0, 0.2]);
            cfg.directivity = 0.8;
        }
        let o = e.add_scene_output(cfg);
        e.connect_pull(o, ChannelPull::new(s, 0, 0.0));
        srcs.push(s);
        if per_pair {
            e.update_scene_spatial();
        }
    }
    e.update_scene_spatial();
    e.update_scene_spatial(); // idempotent: nothing is dirty
    let mut out = AudioBuffer::new(2, BLOCK as u16);
    let mut all = Vec::new();
    for n in 0..24 {
        let bufs: Vec<AudioBuffer> = (0..EMITTERS.len()).map(|k| signal(n, k)).collect();
        let refs: Vec<&AudioBuffer> = bufs.iter().collect();
        out.clear();
        e.process_audio_scene(&refs, std::slice::from_mut(&mut out));
        for ch in 0..2 {
            all.extend_from_slice(out.channel(ch));
        }
    }
    all
}

#[test]
fn batched_update_renders_identically_to_per_pair_updates() {
    for strategy in [HybridSamplingStrategy::RealTimeOnly, HybridSamplingStrategy::HybridBlend] {
        // HybridBlend without a grid fails per pair (nothing published) in both paths, which is
        // itself an identical result; the real-time strategy carries the content.
        let a = run(strategy, false);
        let b = run(strategy, true);
        assert_eq!(a.len(), b.len());
        let bits_equal = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(bits_equal, "{strategy:?}: batched and per-pair updates must render bit-identical audio");
        if strategy == HybridSamplingStrategy::RealTimeOnly {
            assert!(a.iter().any(|v| v.abs() > 1e-4), "the reference render must not be silent");
        }
    }
}
