//! #91 DSP performance: measurement harness + golden-output regression.
//!
//! * `golden_*` tests (always run) replay deterministic scenarios through every hot DSP
//!   component and compare the output against `tests/golden/*.f32`, recorded from the
//!   scalar implementation that existed BEFORE the #91 optimisations. Tolerance: 1e-5
//!   relative to the local magnitude (plus 1e-5 of the scenario peak as a floor for values
//!   near zero crossings).
//!   Re-record with `QUASAR_GEN_GOLDEN=1 cargo test -p quasar-dsp --test dsp_perf golden`.
//! * `bench_*` tests are `#[ignore]`d. Run with
//!   `cargo test -p quasar-dsp --release --test dsp_perf -- --ignored --nocapture --test-threads=1`
//!   and read the `ns/block` / `ns/sample` lines (std::time::Instant, best of 7 repeats).
//! * `decay_has_no_denormal_spike` (ignored, timing based) checks that a decaying FDN tail does
//!   not get slower than its steady cost.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use quasar_core::bands::Band8;
use quasar_core::param_exchange::{EarlyReflectionCoeffs, SpatialCoefficients};
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::binaural::{BinauralConfig, BinauralRenderer, ParametricBinauralRenderer};
use quasar_dsp::early_reflections::EarlyReflectionDelayNode;
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::node_graph::AudioNode;
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;
use quasar_dsp::reflection_decoder::{ReflectionDecoder, TapTarget};
use quasar_dsp::vbap::VbapPanner;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

// ── deterministic signal source ──────────────────────────────────────────

struct Noise(u32);
impl Noise {
    fn next(&mut self) -> f32 {
        // xorshift32 -> [-0.5, 0.5)
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }
    fn fill(&mut self, dst: &mut [f32]) {
        for v in dst.iter_mut() {
            *v = self.next();
        }
    }
}

fn coeffs(direct: [f32; 8], delay: f32, t60: Band8, late_db: f32, refl: Vec<EarlyReflectionCoeffs>) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::new(direct),
        direct_delay_samples: delay,
        direct_azimuth: 0.0,
        direct_elevation: 0.0,
        early_reflections: refl,
        late_t60: t60,
        late_gain_db: late_db,
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version: 0,
    }
}

fn layout_5_1() -> VbapPanner {
    let d = |az_deg: f32| {
        let a = az_deg.to_radians();
        [a.sin(), 0.0, -a.cos()]
    };
    VbapPanner::new(&[d(-30.0), d(30.0), d(0.0), d(0.0), d(-110.0), d(110.0)], &[3])
}

fn layout_7_1_4() -> VbapPanner {
    let d = |az_deg: f32, el_deg: f32| {
        let (a, e) = (az_deg.to_radians(), el_deg.to_radians());
        [a.sin() * e.cos(), e.sin(), -a.cos() * e.cos()]
    };
    VbapPanner::new(
        &[
            d(-30.0, 0.0),
            d(30.0, 0.0),
            d(0.0, 0.0),
            d(0.0, 0.0),
            d(-90.0, 0.0),
            d(90.0, 0.0),
            d(-135.0, 0.0),
            d(135.0, 0.0),
            d(-45.0, 45.0),
            d(45.0, 45.0),
            d(-135.0, 45.0),
            d(135.0, 45.0),
        ],
        &[3],
    )
}

// ── scenarios: one object per component, `block(b)` renders block `b` ─────

trait Scenario {
    fn block(&mut self, b: usize);
    /// Samples (per output channel group) rendered by one `block` call.
    fn samples_per_block(&self) -> usize {
        BLOCK
    }
    /// The signal that is compared against the golden file after block `b`.
    fn capture(&self, out: &mut Vec<f32>);
}

// 1. FDN (one per listener): mono bus in, 6 diffuse channels out.
struct FdnSc {
    node: FdnReverbNode,
    noise: Noise,
    input: Vec<f32>,
    out: AudioBuffer,
}
impl FdnSc {
    fn new() -> Self {
        let mut node = FdnReverbNode::new(1, SR);
        node.set_t60(&Band8::new([2.4, 2.2, 2.0, 1.8, 1.5, 1.2, 0.9, 0.6]));
        node.set_wet(0.7);
        node.set_pre_delay(0.004);
        Self { node, noise: Noise(0x1234_5678), input: vec![0.0; BLOCK], out: AudioBuffer::new(6, BLOCK as u16) }
    }
}
impl Scenario for FdnSc {
    fn block(&mut self, b: usize) {
        self.noise.fill(&mut self.input);
        if b % 3 == 0 {
            let k = 1.0 + 0.1 * ((b / 3) % 4) as f32;
            self.node.set_t60(&Band8::new([2.4 * k, 2.2 * k, 2.0 * k, 1.8, 1.5, 1.2, 0.9 / k, 0.6 / k]));
        }
        self.node.process_bus(&self.input, &mut self.out, 6);
    }
    fn capture(&self, out: &mut Vec<f32>) {
        for c in [0u16, 1, 5] {
            out.extend_from_slice(self.out.channel(c));
        }
    }
}

// 2. Early reflection node (mono fold, 16 taps, stereo input).
struct EarlySc {
    node: EarlyReflectionDelayNode,
    noise: Noise,
    input: AudioBuffer,
    out: AudioBuffer,
}
fn refl_set(b: usize, n: usize) -> Vec<EarlyReflectionCoeffs> {
    (0..n)
        .map(|i| EarlyReflectionCoeffs {
            azimuth: (i as f32 * 0.7).sin() * 3.0,
            elevation: (i as f32 * 0.3).sin() * 0.5,
            delay_samples: 211.3 + 397.7 * i as f32 + 0.04 * b as f32 * (1.0 + 0.1 * i as f32),
            gain: Band8::new([0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]).add(&Band8::splat(0.01 * i as f32)),
        })
        .collect()
}
impl EarlySc {
    fn new() -> Self {
        Self {
            node: EarlyReflectionDelayNode::new(2, SR, 0.5, 16),
            noise: Noise(0x9e37_79b9),
            input: AudioBuffer::new(2, BLOCK as u16),
            out: AudioBuffer::new(1, BLOCK as u16),
        }
    }
}
impl Scenario for EarlySc {
    fn block(&mut self, b: usize) {
        for c in 0..2u16 {
            let mut tmp = [0.0f32; BLOCK];
            self.noise.fill(&mut tmp);
            self.input.channel_mut(c).copy_from_slice(&tmp);
        }
        self.node.update_reflections(&refl_set(b, 16));
        let p = coeffs([1.0; 8], 0.0, Band8::splat(1.0), -60.0, Vec::new());
        self.node.process(&self.input, &mut self.out, &p);
    }
    fn capture(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(self.out.channel(0));
    }
}

// 3. Reflection decoder (16 tap targets): speaker (VBAP 5.1) or HRTF.
struct DecoderSc {
    line: EarlyReflectionDelayNode,
    dec: ReflectionDecoder,
    panner: Option<VbapPanner>,
    noise: Noise,
    input: AudioBuffer,
    out: AudioBuffer,
    taps: usize,
}
impl DecoderSc {
    fn new(hrtf: bool, taps: usize) -> Self {
        let panner = if hrtf { None } else { Some(layout_5_1()) };
        let ch = if hrtf { 2 } else { 6 };
        Self {
            line: EarlyReflectionDelayNode::new(1, SR, 0.5, 16),
            dec: ReflectionDecoder::new(SR, hrtf),
            panner,
            noise: Noise(0xdead_beef),
            input: AudioBuffer::new(1, BLOCK as u16),
            out: AudioBuffer::new(ch, BLOCK as u16),
            taps,
        }
    }
}
impl Scenario for DecoderSc {
    fn block(&mut self, b: usize) {
        let mut tmp = [0.0f32; BLOCK];
        self.noise.fill(&mut tmp);
        self.input.channel_mut(0).copy_from_slice(&tmp);
        self.line.push_block(&self.input);
        let mut targets = [TapTarget::default(); 16];
        for (i, t) in targets.iter_mut().enumerate().take(self.taps) {
            *t = TapTarget {
                delay_samples: 311.7 + 411.3 * i as f32 + 0.05 * b as f32,
                gain_lo: 0.5 / (1.0 + i as f32 * 0.3),
                gain_hi: 0.3 / (1.0 + i as f32 * 0.3),
                azimuth: (i as f32 * 0.9 + 0.02 * b as f32).sin() * 3.0,
                elevation: (i as f32 * 0.5).sin() * 0.6,
            };
        }
        self.out.clear();
        self.dec.render_add(&self.line, &targets[..self.taps], self.panner.as_ref(), &mut self.out, BLOCK);
    }
    fn capture(&self, out: &mut Vec<f32>) {
        for c in 0..self.out.channels() {
            out.extend_from_slice(self.out.channel(c));
        }
    }
}

// 4. One binaural renderer (one tap).
struct BinauralSc {
    r: ParametricBinauralRenderer,
    noise: Noise,
    input: Vec<f32>,
    l: Vec<f32>,
    rr: Vec<f32>,
}
impl BinauralSc {
    fn new() -> Self {
        Self {
            r: ParametricBinauralRenderer::new(BinauralConfig::new(SR)),
            noise: Noise(0x0bad_cafe),
            input: vec![0.0; BLOCK],
            l: vec![0.0; BLOCK],
            rr: vec![0.0; BLOCK],
        }
    }
}
impl Scenario for BinauralSc {
    fn block(&mut self, b: usize) {
        self.noise.fill(&mut self.input);
        self.l.iter_mut().for_each(|v| *v = 0.0);
        self.rr.iter_mut().for_each(|v| *v = 0.0);
        let az = ((b as f32) * 0.37).sin() * 2.5;
        let el = ((b as f32) * 0.21).sin() * 0.7;
        self.r.render_add(&self.input, az, el, &mut self.l, &mut self.rr);
    }
    fn capture(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.l);
        out.extend_from_slice(&self.rr);
    }
}

// 5. Occlusion node: stereo, moving delay, all 8 sections live.
struct OccSc {
    node: AirAbsorptionOcclusionNode,
    noise: Noise,
    input: AudioBuffer,
    out: AudioBuffer,
    ch: u16,
}
impl OccSc {
    fn new(ch: u16) -> Self {
        Self {
            node: AirAbsorptionOcclusionNode::new(ch, SR, 1.0),
            noise: Noise(0x7777_1234),
            input: AudioBuffer::new(ch, BLOCK as u16),
            out: AudioBuffer::new(ch, BLOCK as u16),
            ch,
        }
    }
}
impl Scenario for OccSc {
    fn block(&mut self, b: usize) {
        for c in 0..self.ch {
            let mut tmp = [0.0f32; BLOCK];
            self.noise.fill(&mut tmp);
            self.input.channel_mut(c).copy_from_slice(&tmp);
        }
        let k = 1.0 + 0.2 * ((b % 5) as f32);
        let g = [0.9, 0.85, 0.7 / k, 0.5 / k, 0.4 / k, 0.25 / k, 0.15 / k, 0.08 / k];
        let delay = 300.0 + 20.0 * ((b as f32) * 0.3).sin();
        let p = coeffs(g, delay, Band8::splat(1.0), -60.0, Vec::new());
        self.node.process(&self.input, &mut self.out, &p);
    }
    fn capture(&self, out: &mut Vec<f32>) {
        for c in 0..self.ch {
            out.extend_from_slice(self.out.channel(c));
        }
    }
}

// 6. VBAP: gains() for a moving source + the ramped decode of one block into every speaker.
struct VbapSc {
    panner: VbapPanner,
    noise: Noise,
    input: Vec<f32>,
    out: AudioBuffer,
    prev: [f32; 32],
    n: usize,
}
impl VbapSc {
    fn new(hull: bool) -> Self {
        let panner = if hull { layout_7_1_4() } else { layout_5_1() };
        let n = panner.num_outputs();
        Self {
            panner,
            noise: Noise(0x5151_5151),
            input: vec![0.0; BLOCK],
            out: AudioBuffer::new(n as u16, BLOCK as u16),
            prev: [0.0; 32],
            n,
        }
    }
}
impl Scenario for VbapSc {
    fn block(&mut self, b: usize) {
        self.noise.fill(&mut self.input);
        let az = ((b as f32) * 0.31).sin() * 3.0;
        let el = ((b as f32) * 0.17).sin() * 0.8;
        let mut target = [0.0_f32; 32];
        self.panner.gains(az, el, &mut target[..self.n]);
        self.out.clear();
        for sp in 0..self.n {
            let (g0, g1) = (self.prev[sp], target[sp]);
            self.prev[sp] = g1;
            if g0 == 0.0 && g1 == 0.0 {
                continue;
            }
            quasar_dsp::vbap::ramp_add(self.out.channel_mut(sp as u16), &self.input, g0, g1);
        }
    }
    fn capture(&self, out: &mut Vec<f32>) {
        for c in 0..self.out.channels() {
            out.extend_from_slice(self.out.channel(c));
        }
    }
}

// ── golden files ─────────────────────────────────────────────────────────

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden").join(format!("{name}.f32"))
}

fn run_capture(sc: &mut dyn Scenario, blocks: usize) -> Vec<f32> {
    let mut all = Vec::new();
    for b in 0..blocks {
        sc.block(b);
        sc.capture(&mut all);
    }
    all
}

fn check_golden(name: &str, sc: &mut dyn Scenario, blocks: usize) {
    let got = run_capture(sc, blocks);
    let path = golden_path(name);
    if std::env::var("QUASAR_GEN_GOLDEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        return;
    }
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing golden {path:?}: {e}"));
    let want: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(got.len(), want.len(), "{name}: length");
    let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak > 1e-4, "{name}: golden signal is silent");
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let tol = 1e-5 * w.abs() + 1e-5 * peak;
        let err = (g - w).abs();
        worst = worst.max(err / peak);
        assert!(err <= tol, "{name}[{i}]: got {g}, golden {w}, |err| {err} > {tol}");
    }
    eprintln!("{name}: {} samples, worst error {:.2e} of peak", got.len(), worst);
}

#[test]
fn golden_fdn() {
    check_golden("fdn", &mut FdnSc::new(), 14);
}
#[test]
fn golden_early_reflections() {
    check_golden("early", &mut EarlySc::new(), 10);
}
#[test]
fn golden_reflection_decoder_speakers() {
    check_golden("decoder_spk", &mut DecoderSc::new(false, 16), 8);
}
#[test]
fn golden_reflection_decoder_hrtf() {
    check_golden("decoder_hrtf", &mut DecoderSc::new(true, 6), 6);
}
#[test]
fn golden_binaural() {
    check_golden("binaural", &mut BinauralSc::new(), 10);
}
#[test]
fn golden_occlusion() {
    check_golden("occlusion", &mut OccSc::new(2), 12);
}
#[test]
fn golden_vbap_planar() {
    check_golden("vbap_planar", &mut VbapSc::new(false), 10);
}
#[test]
fn golden_vbap_hull() {
    check_golden("vbap_hull", &mut VbapSc::new(true), 10);
}

// ── benchmarks ───────────────────────────────────────────────────────────

/// Best-of-`REPEATS` mean ns per `block` call.
fn time_block(sc: &mut dyn Scenario, iters: usize) -> f64 {
    const REPEATS: usize = 7;
    for b in 0..iters.min(200) {
        sc.block(b); // warm-up (also fills delay lines)
    }
    let mut best = f64::INFINITY;
    for r in 0..REPEATS {
        let t = Instant::now();
        for b in 0..iters {
            sc.block(black_box(b + r * iters));
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        best = best.min(ns);
    }
    black_box(&sc);
    best
}

fn report(name: &str, sc: &mut dyn Scenario, iters: usize, unit_div: usize, unit: &str) {
    let ns = time_block(sc, iters);
    let spb = sc.samples_per_block();
    println!(
        "BENCH {name:<34} {:>10.0} ns/block  {:>8.2} ns/sample  {:>9.1} ns/{unit}",
        ns,
        ns / spb as f64,
        ns / unit_div as f64
    );
}

#[test]
#[ignore]
fn bench_all() {
    println!("--- #91 DSP bench (block = {BLOCK} samples @ {SR} Hz) ---");
    report("fdn_bus_6out", &mut FdnSc::new(), 4000, 1, "block");
    report("early_reflections_16tap_mono", &mut EarlySc::new(), 3000, 16, "tap-block");
    report("reflection_decoder_5.1_16tap", &mut DecoderSc::new(false, 16), 2000, 16, "tap-block");
    report("reflection_decoder_hrtf_6tap", &mut DecoderSc::new(true, 6), 1000, 6, "tap-block");
    report("binaural_1tap", &mut BinauralSc::new(), 4000, 1, "tap-block");
    report("occlusion_1ch_8sec", &mut OccSc::new(1), 6000, 1, "ch-block");
    report("occlusion_2ch_8sec", &mut OccSc::new(2), 4000, 2, "ch-block");
    report("vbap_planar_5.1_gains+decode", &mut VbapSc::new(false), 20000, 1, "block");
    report("vbap_hull_7.1.4_gains+decode", &mut VbapSc::new(true), 20000, 1, "block");

    // gains() alone.
    for (name, p) in [("vbap_gains_only_5.1", layout_5_1()), ("vbap_gains_only_7.1.4", layout_7_1_4())] {
        let mut out = [0.0f32; 16];
        for i in 0..2000 {
            p.gains(black_box(i as f32 * 0.01), 0.2, &mut out);
        }
        let iters = 200_000;
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            for i in 0..iters {
                p.gains(black_box((i % 628) as f32 * 0.01 - 3.14), black_box(0.2), &mut out);
            }
            best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
        }
        black_box(out);
        println!("BENCH {name:<34} {best:>10.1} ns/call");
    }
}

/// A decaying FDN tail must not slow down (denormals): ns/block over a long decay stays within
/// 1.5x the steady cost. Run in release; timing based, hence ignored by default.
#[test]
#[ignore]
fn decay_has_no_denormal_spike() {
    let mut node = FdnReverbNode::new(1, SR);
    node.set_t60(&Band8::splat(0.4));
    node.set_wet(1.0);
    let mut noise = Noise(42);
    let mut input = vec![0.0f32; BLOCK];
    let mut out = AudioBuffer::new(2, BLOCK as u16);
    // Steady state: continuous noise.
    for _ in 0..200 {
        noise.fill(&mut input);
        node.process_bus(&input, &mut out, 2);
    }
    let t = Instant::now();
    for _ in 0..1000 {
        noise.fill(&mut input);
        node.process_bus(black_box(&input), &mut out, 2);
    }
    let steady = t.elapsed().as_nanos() as f64 / 1000.0;
    // Decay: silence in; the tail goes through -100 dB .. denormal range (0.4 s T60 -> ~0.7 s to
    // 1e-38 is not reached, so drive 1.5 s = 280 blocks in chunks and time each).
    input.iter_mut().for_each(|v| *v = 0.0);
    let mut worst: f64 = 0.0;
    let mut total = 0.0;
    let blocks = 2500; // 13 s
    for b in 0..blocks {
        let t = Instant::now();
        node.process_bus(black_box(&input), &mut out, 2);
        let ns = t.elapsed().as_nanos() as f64;
        total += ns;
        // Use 10-block windows to filter scheduler noise.
        if b % 10 == 9 {
            worst = worst.max(total / 10.0);
            total = 0.0;
        }
    }
    println!("BENCH decay: steady {steady:.0} ns/block, worst 10-block decay window {worst:.0} ns/block ({:.2}x)", worst / steady);
    assert!(worst <= 1.5 * steady, "decaying tail slowed down: {worst} vs steady {steady}");
}
