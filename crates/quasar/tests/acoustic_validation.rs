//! Objective acoustic validation harness (#95): scripted scenes rendered OFFLINE through the
//! real engine (no device, deterministic noise) with measurements asserted on the output.
//!
//! What is measured and why is documented in `docs/validation.md`. Run with
//!
//! ```text
//! cargo test -p quasar-audio --release --test acoustic_validation -- --nocapture
//! ```
//!
//! Sections:
//!  1. localisation: panning law and direction estimates for stereo / quad / 5.1 / 7.1 (velocity
//!     vector) and an HRTF listener (ITD by cross-correlation, ILD);
//!  2. propagation: impulse latency of the direct path = distance / c;
//!  3. room: RT60 / EDT / C50 / D50 of the rendered reverb (Schroeder integration) against the
//!     probe T60, and the analytic first-order image-source delays in a shoebox;
//!  4. loudness: an ITU-R BS.1770 K-weighted integrated loudness meter (self-checked against the
//!     standard's calibration point), distance law in LU and the output peak ceiling;
//!  5. real-time safety: the render path does not allocate (counting global allocator).
//!
//! The balance assertions of the demo's cathedral live in `direct_vs_room_levels.rs`.

mod common;

use quasar_audio::quasar_backends::cpu_simd::CpuSimdConfig;
use quasar_audio::quasar_backends::hw_stub::HardwareAcceleratorStub;
use quasar_audio::quasar_backends::CpuSimdComputeBackend;
use quasar_audio::quasar_core::bands::Band8;
use quasar_audio::quasar_core::hybrid::HybridSamplingStrategy;
use quasar_audio::quasar_core::probe_grid::{AcousticProbe, AcousticProbeGrid};
use quasar_audio::quasar_core::scene::{AcousticMesh, AcousticScene, Movability};
use quasar_audio::quasar_core::scene_output::{
    ChannelPull, ListenerConfig, PhysicalOutputLayout, SceneOutputConfig, SourceConfig,
};
use quasar_audio::quasar_dsp::audio_buffer::AudioBuffer;
use quasar_audio::quasar_dsp::binaural::woodworth_itd_seconds;
use quasar_audio::quasar_dsp::limiter::OutputSafetyConfig;
use quasar_audio::quasar_dsp::master_decoder::{layout_lfe, layout_positions, SpeakerLayout};
use quasar_audio::quasar_materials::instance::AcousticMaterialInstance;
use quasar_audio::quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_audio::SpatialAudioEngine;

const SR: f32 = 48_000.0;
const C: f32 = 343.0;
const BLOCK: usize = 256;

// ── scene + render helpers ────────────────────────────────────────────────

fn channels_of(layout: &PhysicalOutputLayout) -> u16 {
    match layout {
        PhysicalOutputLayout::Stereo | PhysicalOutputLayout::Hrtf => 2,
        PhysicalOutputLayout::Quad => 4,
        PhysicalOutputLayout::Surround51 => 6,
        PhysicalOutputLayout::Surround714 => 8,
        PhysicalOutputLayout::Custom { positions } => positions.len() as u16,
    }
}

/// Stub-backend engine (direct path = distance law, constant late field -10 dB / T60 0.5 s, no
/// geometry): the listener sits at the origin facing -Z; one emitter at azimuth `az_deg`
/// (+ = right), `dist` metres, ear height. The output limiter is off so levels are linear.
fn polar_engine(layout: PhysicalOutputLayout, az_deg: f32, dist: f32) -> SpatialAudioEngine {
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.set_backend(Box::new(HardwareAcceleratorStub::new()));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let src = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).expect("source");
    let a = az_deg.to_radians();
    let out = e.add_scene_output(SceneOutputConfig::new([dist * a.sin(), 0.0, -dist * a.cos()], Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    let l = e.add_listener(ListenerConfig { position: [0.0; 3], heading: [0.0, 0.0, -1.0], physical_layout: layout });
    e.set_output_safety(l, OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
    e.update_scene_spatial();
    e
}

struct Noise(u32);
impl Noise {
    /// Uniform white noise in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// Render `blocks` blocks of `input(sample_index)` (mono) at `stage`; returns one `Vec` per
/// output channel plus the input signal.
fn render(e: &mut SpatialAudioEngine, stage: u8, channels: u16, blocks: usize, mut input: impl FnMut(usize) -> f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    e.debug_audio_stage = stage;
    let mut out = AudioBuffer::new(channels, BLOCK as u16);
    let mut res = vec![Vec::with_capacity(blocks * BLOCK); channels as usize];
    let mut inp = Vec::with_capacity(blocks * BLOCK);
    let mut src = AudioBuffer::new(1, BLOCK as u16);
    for b in 0..blocks {
        for i in 0..BLOCK {
            let v = input(b * BLOCK + i);
            src.set(0, i as u16, v);
            inp.push(v);
        }
        out.clear();
        e.process_audio_scene(&[&src], std::slice::from_mut(&mut out));
        for c in 0..channels as usize {
            res[c].extend_from_slice(&out.channel(c as u16)[..BLOCK]);
        }
    }
    (res, inp)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len().max(1) as f64).sqrt() as f32
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

// ── 1. localisation ───────────────────────────────────────────────────────

/// Direction (degrees, + = right, 0 = ahead) of the VELOCITY vector `sum g_i u_i` of the
/// speaker amplitudes, and the total power (re the input power) of the output.
fn velocity_direction(layout: &SpeakerLayout, gains: &[f32]) -> f32 {
    let pos = layout_positions(layout);
    let (mut x, mut z) = (0.0_f32, 0.0_f32);
    for (g, p) in gains.iter().zip(&pos) {
        x += g * p[0];
        z += g * p[2];
    }
    x.atan2(-z).to_degrees()
}

/// Per-channel steady-state amplitude gains of the direct path of one emitter (stage 2, noise).
fn channel_gains(layout: PhysicalOutputLayout, az_deg: f32, dist: f32) -> Vec<f32> {
    let ch = channels_of(&layout);
    let mut e = polar_engine(layout, az_deg, dist);
    let mut rng = Noise(0x1357_9bdf);
    let (y, x) = render(&mut e, 2, ch, 80, |_| 0.5 * rng.next());
    let input = rms(&x[40 * BLOCK..]);
    // The distance law and air absorption scale every channel alike: normalise by the 1/d gain.
    y.iter().map(|c| rms(&c[40 * BLOCK..]) / input * dist).collect()
}

fn angle_error(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d < -180.0 {
        d += 360.0;
    }
    d.abs()
}

#[test]
fn panning_law_is_constant_power_and_points_at_the_source() {
    // (layout, engine layout, sweep of source azimuths, direction tolerance in degrees)
    let cases: [(&str, SpeakerLayout, PhysicalOutputLayout, Vec<f32>); 4] = [
        ("stereo", SpeakerLayout::Stereo, PhysicalOutputLayout::Stereo, (-6..=6).map(|k| k as f32 * 5.0).collect()),
        ("quad", SpeakerLayout::Quad, PhysicalOutputLayout::Quad, (-12..12).map(|k| k as f32 * 15.0).collect()),
        ("5.1", SpeakerLayout::Surround51, PhysicalOutputLayout::Surround51, (-12..12).map(|k| k as f32 * 15.0).collect()),
        ("7.1", SpeakerLayout::Surround714, PhysicalOutputLayout::Surround714, (-12..12).map(|k| k as f32 * 15.0).collect()),
    ];
    for (name, layout, phys, azimuths) in cases {
        let lfe = layout_lfe(&layout);
        let mut worst_dir = 0.0_f32;
        let (mut min_power, mut max_power) = (f32::MAX, f32::MIN);
        for &az in &azimuths {
            let g = channel_gains(phys.clone(), az, 2.0);
            // Sources outside the front pair of a stereo layout are clamped by the panner; only
            // the sweep inside +-30 degrees is a localisation claim there.
            assert!(lfe.iter().all(|&i| g[i] < 1e-6), "{name}: nothing is panned to the LFE slot");
            let power: f32 = g.iter().map(|v| v * v).sum();
            min_power = min_power.min(db(power.sqrt()));
            max_power = max_power.max(db(power.sqrt()));
            let est = velocity_direction(&layout, &g);
            worst_dir = worst_dir.max(angle_error(est, az));
        }
        // Total power over the channels is the same for every azimuth (constant-power panning).
        // It sits slightly below 0 dB re the 1/d gain because the direct chain also applies air
        // absorption (about -0.24 dB of broadband noise power at 2 m).
        println!("{name}: worst direction error {worst_dir:.2} deg, total power {min_power:.3} .. {max_power:.3} dB re 1/d");
        assert!(max_power - min_power < 0.1, "{name}: constant power across azimuth: {min_power:.3} .. {max_power:.3} dB");
        assert!(max_power < 0.1 && min_power > -0.6, "{name}: total power {min_power:.3} .. {max_power:.3} dB re the 1/d gain");
        assert!(worst_dir < 1.5, "{name}: velocity-vector direction error {worst_dir:.2} deg");
    }
}

/// Cross-correlation lag (samples, parabolic peak) of `b` against `a`: positive when `b` lags `a`.
fn xcorr_lag(a: &[f32], b: &[f32], max_lag: i32) -> f32 {
    let n = a.len().min(b.len()) - max_lag as usize - 1;
    let corr = |lag: i32| -> f64 {
        (max_lag as usize..n).map(|i| a[i] as f64 * b[(i as i32 + lag) as usize] as f64).sum::<f64>()
    };
    let (mut best, mut best_v) = (0, f64::MIN);
    for lag in -max_lag + 1..max_lag {
        let v = corr(lag);
        if v > best_v {
            best_v = v;
            best = lag;
        }
    }
    let (y0, y1, y2) = (corr(best - 1), corr(best), corr(best + 1));
    let den = y0 - 2.0 * y1 + y2;
    best as f32 + if den.abs() > 1e-12 { (0.5 * (y0 - y2) / den) as f32 } else { 0.0 }
}

#[test]
fn hrtf_listener_has_woodworth_itd_and_a_head_shadow() {
    let mut itd_err_worst = 0.0_f32;
    let mut last_ild = -1.0_f32;
    let mut first_ild = f32::MAX;
    for az in [15.0_f32, 30.0, 45.0, 60.0, 75.0, 90.0] {
        let mut e = polar_engine(PhysicalOutputLayout::Hrtf, az, 2.0);
        let mut rng = Noise(0x0bad_cafe);
        let (y, _) = render(&mut e, 2, 2, 120, |_| 0.5 * rng.next());
        let (l, r) = (&y[0][40 * BLOCK..], &y[1][40 * BLOCK..]);
        // Source on the right: the LEFT ear lags, so `l` against `r` has a positive lag.
        let itd = xcorr_lag(r, l, 60) / SR;
        let want = woodworth_itd_seconds(az.to_radians(), 0.0875, C);
        let ild = db(rms(r)) - db(rms(l));
        println!("az {az:>4}: ITD {:.3} ms (Woodworth {:.3} ms), ILD {ild:.1} dB", itd * 1e3, want * 1e3);
        itd_err_worst = itd_err_worst.max((itd - want).abs());
        // (Pinna EQ makes the ILD plateau / dip by about 1 dB beyond 60 degrees.)
        assert!(ild > last_ild - 1.0, "the head shadow must not shrink with azimuth: {ild} after {last_ild}");
        first_ild = first_ild.min(ild);
        last_ild = ild;
        assert!(ild > 0.5, "the ear facing the source must be louder: {ild} dB at {az} deg");
    }
    assert!(itd_err_worst < 60e-6, "ITD must follow the Woodworth model to within 60 us: worst {:.1} us", itd_err_worst * 1e6);
    assert!(last_ild > first_ild + 6.0, "head shadow at 90 deg ({last_ild:.1} dB) must clearly exceed 15 deg ({first_ild:.1} dB)");
    // And mirrored: a source on the left gives the mirrored ITD.
    let mut e = polar_engine(PhysicalOutputLayout::Hrtf, -60.0, 2.0);
    let mut rng = Noise(0x0bad_cafe);
    let (y, _) = render(&mut e, 2, 2, 120, |_| 0.5 * rng.next());
    let itd = xcorr_lag(&y[0][40 * BLOCK..], &y[1][40 * BLOCK..], 60) / SR;
    assert!((itd - woodworth_itd_seconds(60.0_f32.to_radians(), 0.0875, C)).abs() < 60e-6, "mirror ITD {itd}");
}

// ── 2. propagation latency ────────────────────────────────────────────────

#[test]
fn direct_impulse_arrives_after_distance_over_c() {
    for dist in [1.0_f32, 3.43, 10.0, 30.0] {
        let mut e = polar_engine(PhysicalOutputLayout::Stereo, 0.0, dist);
        let t0 = 3 * BLOCK;
        let (y, _) = render(&mut e, 2, 2, 3 + (dist * SR / C / BLOCK as f32).ceil() as usize + 3, |i| if i == t0 { 1.0 } else { 0.0 });
        let sum: Vec<f32> = (0..y[0].len()).map(|i| (y[0][i] * y[0][i] + y[1][i] * y[1][i]).sqrt()).collect();
        let peak = sum.iter().enumerate().fold((0, 0.0_f32), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
        let want = t0 as f32 + dist * SR / C;
        assert!((peak as f32 - want).abs() <= 1.5, "{dist} m: impulse peak at sample {peak}, expected {want:.1}");
        // 1/d amplitude, as total energy (a band-limited impulse spreads over a few samples). Air
        // absorption can only take energy away: -3 .. +0.5 dB of 1/d^2.
        let energy: f32 = sum.iter().map(|v| v * v).sum();
        let want_e = 1.0 / (dist * dist);
        assert!((-3.0..0.5).contains(&(10.0 * (energy / want_e).log10())), "{dist} m: energy {energy} vs 1/d^2 = {want_e}");
    }
}

// ── 3. room acoustics ─────────────────────────────────────────────────────

/// Constant probe grid (3x3x3, `spacing` m) + `BakedOnly`: the late field is the statistical
/// model with a KNOWN `t60` and volume `(2 spacing)^3`; one omni emitter 1 m in front of the
/// listener at the grid centre. No early reflections, so the response is direct + late field.
fn baked_engine(t60: f32, spacing: f32) -> SpatialAudioEngine {
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
    let c = spacing;
    let src = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).unwrap();
    let out = e.add_scene_output(SceneOutputConfig::new([c, c, c - 1.0], Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    let l = e.add_listener(ListenerConfig { position: [c, c, c], heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.set_output_safety(l, OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
    e.update_scene_spatial();
    e
}

/// Impulse response (total power over the channels, per sample) of an engine at `stage`.
fn impulse_power(mut e: SpatialAudioEngine, stage: u8, ch: u16, blocks: usize) -> Vec<f32> {
    let (y, _) = render(&mut e, stage, ch, blocks, |i| if i == BLOCK { 1.0 } else { 0.0 });
    (BLOCK..y[0].len()).map(|i| y.iter().map(|c| c[i] * c[i]).sum()).collect()
}

/// Power (per sample, total over the channels) of what stage `upper` adds on top of stage
/// `lower`: the DIFFERENCE of the two impulse responses (not of their powers, so a reflection that
/// overlaps the ringing of the direct sound has no cross term).
fn layer_power(mk: impl Fn() -> SpatialAudioEngine, upper: u8, lower: u8, ch: u16, blocks: usize) -> Vec<f32> {
    let imp = |i: usize| if i == BLOCK { 1.0 } else { 0.0 };
    let (hi, _) = render(&mut mk(), upper, ch, blocks, imp);
    let (lo, _) = render(&mut mk(), lower, ch, blocks, imp);
    (BLOCK..hi[0].len()).map(|i| (0..ch as usize).map(|c| (hi[c][i] - lo[c][i]).powi(2)).sum()).collect()
}

/// Schroeder backward integration: the decay curve in dB (0 dB at t = 0) of an energy response.
fn schroeder_db(p: &[f32]) -> Vec<f32> {
    let mut acc = 0.0_f64;
    let mut e: Vec<f64> = p.iter().rev().map(|&v| { acc += v as f64; acc }).collect();
    e.reverse();
    let total = e[0].max(1e-30);
    e.iter().map(|&v| 10.0 * (v / total).max(1e-12).log10() as f32).collect()
}

/// Reverberation time from the decay curve: linear regression between `hi` and `lo` dB
/// (e.g. -5 .. -25 = T20, 0 .. -10 = EDT), extrapolated to 60 dB. Seconds.
fn decay_time(curve: &[f32], hi: f32, lo: f32) -> f32 {
    let idx: Vec<usize> = (0..curve.len()).filter(|&i| curve[i] <= hi && curve[i] >= lo).collect();
    let n = idx.len() as f64;
    assert!(n > 10.0, "decay range {hi}..{lo} dB not covered");
    let (sx, sy) = (idx.iter().map(|&i| i as f64).sum::<f64>(), idx.iter().map(|&i| curve[i] as f64).sum::<f64>());
    let (sxx, sxy) = (idx.iter().map(|&i| (i * i) as f64).sum::<f64>(), idx.iter().map(|&i| i as f64 * curve[i] as f64).sum::<f64>());
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); // dB per sample (negative)
    (-60.0 / slope / SR as f64) as f32
}

/// Clarity C50 in dB: energy in the first 50 ms after the DIRECT arrival over the rest.
fn clarity_c50(p: &[f32], direct_at: usize) -> f32 {
    let split = direct_at + (0.05 * SR) as usize;
    let early: f64 = p[direct_at - 4..split].iter().map(|&v| v as f64).sum();
    let late: f64 = p[split..].iter().map(|&v| v as f64).sum();
    10.0 * (early / late).log10() as f32
}

#[test]
fn rendered_reverb_has_the_probe_rt60_edt_c50_and_d50() {
    // Direct sound 1 m ahead (arrives after ~140 samples), diffuse field from the probes.
    for (t60, spacing) in [(1.0_f32, 10.0_f32), (2.0, 10.0), (4.0, 12.0)] {
        let vol = (2.0 * spacing).powi(3);
        let blocks = ((t60 * 3.0 * SR) / BLOCK as f32).ceil() as usize + 8;
        let full = impulse_power(baked_engine(t60, spacing), 4, 2, blocks);
        let direct = impulse_power(baked_engine(t60, spacing), 2, 2, blocks);
        let tail = layer_power(|| baked_engine(t60, spacing), 4, 2, 2, blocks);
        // Start the integration where the late field starts (direct delay + 50 ms split).
        let onset = tail.iter().position(|&v| v > 1e-12).expect("late field present");
        let curve = schroeder_db(&tail[onset..]);
        let t20 = decay_time(&curve, -5.0, -25.0);
        let edt = decay_time(&curve, 0.0, -10.0);
        // C50 / D50 of the whole response (direct + late), t = 0 at the direct arrival.
        let direct_at = direct.iter().enumerate().fold((0, 0.0_f32), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
        let c50 = clarity_c50(&full, direct_at);
        let late_db = 10.0 * (312.2 * t60 / vol).log10();
        let d50 = 1.0 / (1.0 + 10f32.powf(-c50 / 10.0));
        println!("T60 {t60} s: T20-extrapolated {t20:.2} s, EDT {edt:.2} s, C50 {c50:.1} dB (late level {late_db:.1} dB), D50 {d50:.2}");
        assert!((t20 / t60 - 1.0).abs() < 0.10, "T60 {t60}: T20-extrapolated {t20:.2} s");
        assert!((edt / t60 - 1.0).abs() < 0.20, "T60 {t60}: EDT {edt:.2} s");
        // The late field starts after the 50 ms split, so everything before it is the direct
        // sound: C50 = direct energy / late energy = -(late level re direct at 1 m).
        assert!((c50 + late_db).abs() < 1.0, "T60 {t60}: C50 {c50:.2} dB, expected {:.2} dB", -late_db);
        assert!((d50 - 1.0 / (1.0 + 10f32.powf(late_db / 10.0))).abs() < 0.05, "D50 {d50}");
    }
}

fn shoebox() -> (SpatialAudioEngine, [f32; 3], [f32; 3], [f32; 3]) {
    let (lx, ly, lz) = (10.0_f32, 4.0_f32, 8.0_f32);
    let mut e = SpatialAudioEngine::new(0, SR, 15.0);
    e.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
    let wall = e.materials().add_instance(AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(Band8::splat(0.2), Band8::zeros(), Band8::zeros()),
    ));
    let quads: [[[f32; 3]; 4]; 6] = [
        [[0.0, 0.0, 0.0], [0.0, ly, 0.0], [0.0, ly, lz], [0.0, 0.0, lz]],
        [[lx, 0.0, 0.0], [lx, ly, 0.0], [lx, ly, lz], [lx, 0.0, lz]],
        [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, 0.0, lz], [0.0, 0.0, lz]],
        [[0.0, ly, 0.0], [lx, ly, 0.0], [lx, ly, lz], [0.0, ly, lz]],
        [[0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, ly, 0.0], [0.0, ly, 0.0]],
        [[0.0, 0.0, lz], [lx, 0.0, lz], [lx, ly, lz], [0.0, ly, lz]],
    ];
    let mut scene = AcousticScene::new();
    for (i, q) in quads.iter().enumerate() {
        scene.add_mesh(AcousticMesh::new(i as u64 + 1, q.to_vec(), vec![0, 1, 2, 0, 2, 3], wall));
    }
    let cfg = CpuSimdConfig { max_reflection_order: 1, ..CpuSimdConfig::default() };
    e.set_backend(Box::new(CpuSimdComputeBackend::new(scene, cfg)));
    e.set_strategy(HybridSamplingStrategy::RealTimeOnly);
    let (spk, lis) = ([2.0_f32, 1.2, 2.0], [6.0_f32, 1.6, 5.0]);
    let src = e.load_source(SourceConfig { path: "n.wav".into(), channels: 1 }).unwrap();
    let out = e.add_scene_output(SceneOutputConfig::new(spk, Movability::Static));
    e.connect_pull(out, ChannelPull::new(src, 0, 0.0));
    let l = e.add_listener(ListenerConfig { position: lis, heading: [0.0, 0.0, -1.0], physical_layout: PhysicalOutputLayout::Stereo });
    e.set_output_safety(l, OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
    e.update_scene_spatial();
    (e, spk, lis, [lx, ly, lz])
}

#[test]
fn first_order_reflection_delays_and_levels_match_the_image_sources() {
    let (_e, spk, lis, dims) = shoebox();
    let blocks = 40;
    let refl = layer_power(|| shoebox().0, 3, 2, 2, blocks);
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let d_dir = dist(spk, lis);
    let mut images = Vec::new();
    for axis in 0..3 {
        for wall in [0.0, dims[axis]] {
            let mut im = spk;
            im[axis] = 2.0 * wall - spk[axis];
            images.push(dist(im, lis));
        }
    }
    let t_direct = (d_dir * SR / C).round() as usize;
    let max_e = refl.iter().cloned().fold(0.0_f32, f32::max);
    // Every analytic first-order arrival has energy within -3 .. +0.5 dB of (1 - alpha) / d^2. The
    // backend's per-band gains are exact (its own tests pin them to 0.2 % including air
    // absorption); what the engine adds is the 4-point Hermite fractional delay, which low-passes
    // an IMPULSE by up to -1.9 dB of energy depending on the fractional part of the delay (weights
    // -1/16, 9/16, 9/16, -1/16 at 0.5), and air absorption of the HF-heavy impulse. Times the
    // engine's handover weight: discrete taps fade out over 10 ms ending `early_late_split` after
    // the direct sound, where the diffuse field takes over (this room: split 20 ms).
    let split_end = t_direct as f32 + 0.020 * SR;
    let fade = 0.010 * SR;
    let mut accounted = vec![false; refl.len()];
    for &d in &images {
        let n = (d * SR / C).round() as usize;
        let win: f32 = refl[n.saturating_sub(6)..n + 7].iter().sum();
        let w = ((split_end - n as f32) / fade).clamp(0.0, 1.0);
        let want = 0.8 / (d * d) * w * w;
        if w < 0.01 {
            assert!(win < 1e-3 / (d * d), "image at {d:.2} m is past the handover and must be silent: {win}");
            continue;
        }
        println!("image path {d:6.2} m at sample {n}: energy {win:.5}, expected {want:.5} ({:+.1} dB)", 10.0 * (win / want).log10());
        assert!((-3.0..0.5).contains(&(10.0 * (win / want).log10())), "image at {d:.2} m: energy {win} vs {want}");
        assert!(n > t_direct, "a reflection cannot precede the direct sound");
        for a in accounted[n.saturating_sub(6)..n + 7].iter_mut() {
            *a = true;
        }
    }
    // ... and there is nothing else (order 1 only): the energy outside the six windows is < 0.1% of the peak window.
    let stray: f32 = refl.iter().zip(&accounted).filter(|(_, &a)| !a).map(|(v, _)| *v).sum();
    assert!(stray < 1e-3 * max_e * 13.0, "unexplained energy between the image arrivals: {stray}");
}

// ── 4. loudness ───────────────────────────────────────────────────────────

/// ITU-R BS.1770-4 K-weighting (48 kHz coefficients) for one channel.
fn k_weight(x: &[f32]) -> Vec<f64> {
    // Stage 1: high shelf (head), stage 2: RLB high-pass. Direct form I, f64.
    let s1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [-1.69065929318241, 0.73248077421585]);
    let s2 = ([1.0, -2.0, 1.0], [-1.99004745483398, 0.99007225036621]);
    let mut y: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    for (b, a) in [s1, s2] {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for v in y.iter_mut() {
            let x0 = *v;
            let o = b[0] * x0 + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = o;
            *v = o;
        }
    }
    y
}

/// Integrated loudness in LUFS (BS.1770-4: 400 ms blocks, 75 % overlap, -70 LUFS absolute gate
/// and -10 LU relative gate) of channels that all have weight 1.0 (L / R / C: the stereo and 5.1
/// front channels used here; pass only those).
fn integrated_lufs(channels: &[&[f32]]) -> f32 {
    let w: Vec<Vec<f64>> = channels.iter().map(|c| k_weight(c)).collect();
    let (blk, hop) = ((0.4 * SR as f64) as usize, (0.1 * SR as f64) as usize);
    let n = w[0].len();
    let mut energies = Vec::new();
    let mut s = 0;
    while s + blk <= n {
        let z: f64 = w.iter().map(|c| c[s..s + blk].iter().map(|v| v * v).sum::<f64>() / blk as f64).sum();
        energies.push(z);
        s += hop;
    }
    let loud = |z: f64| -0.691 + 10.0 * z.max(1e-30).log10();
    let gated: Vec<f64> = energies.iter().cloned().filter(|&z| loud(z) > -70.0).collect();
    assert!(!gated.is_empty(), "silence: no block above the absolute gate");
    let rel = loud(gated.iter().sum::<f64>() / gated.len() as f64) - 10.0;
    let kept: Vec<f64> = gated.iter().cloned().filter(|&z| loud(z) > rel).collect();
    loud(kept.iter().sum::<f64>() / kept.len() as f64) as f32
}

#[test]
fn the_loudness_meter_matches_bs1770_calibration() {
    // 997 Hz, -20 dBFS peak (0.1) in both stereo channels: Z = 2 * 0.1^2 / 2 * |H(997)|^2 with
    // |H| = +0.691 dB, so L = -0.691 + 10 log10(0.01 * 1.1725) = -20.0 LUFS.
    let sine: Vec<f32> = (0..(8.0 * SR) as usize).map(|i| 0.1 * (std::f32::consts::TAU * 997.0 * i as f32 / SR).sin()).collect();
    let l = integrated_lufs(&[&sine, &sine]);
    assert!((l + 20.0).abs() < 0.05, "stereo 997 Hz at -20 dBFS: {l:.3} LUFS, expected -20.0");
    // Mono full scale is -3.01 LUFS.
    let full: Vec<f32> = sine.iter().map(|v| v * 10.0).collect();
    let l = integrated_lufs(&[&full]);
    assert!((l + 3.01).abs() < 0.05, "mono 997 Hz at 0 dBFS: {l:.3} LUFS, expected -3.01");
}

#[test]
fn loudness_follows_the_inverse_distance_law_and_the_ceiling_holds() {
    // Direct path only (stage 2): 6.02 LU per doubling of the distance (air absorption makes it
    // a touch more at 4 m).
    let loud = |dist: f32| {
        let mut e = polar_engine(PhysicalOutputLayout::Stereo, 0.0, dist);
        let mut rng = Noise(0x2222_1111);
        let (y, _) = render(&mut e, 2, 2, 700, |_| 0.4 * rng.next());
        integrated_lufs(&[&y[0][BLOCK * 20..], &y[1][BLOCK * 20..]])
    };
    let (l1, l2, l4) = (loud(1.0), loud(2.0), loud(4.0));
    println!("loudness at 1 / 2 / 4 m: {l1:.2} / {l2:.2} / {l4:.2} LUFS");
    // Air absorption (ISO 9613-1) only ever adds: it takes HF, which the K-weighting boosts.
    for (a, b, name) in [(l1, l2, "1 -> 2 m"), (l2, l4, "2 -> 4 m")] {
        assert!((6.0..7.0).contains(&(a - b)), "{name}: {:.2} LU per doubling", a - b);
    }

    // Output peak ceiling: a mix 20 dB too hot through the default output stage (-1 dBFS).
    let mut e = polar_engine(PhysicalOutputLayout::Stereo, 20.0, 1.0);
    e.set_output_safety(quasar_audio::quasar_core::scene_output::ListenerId(0), OutputSafetyConfig::default());
    let mut rng = Noise(0x3333_4444);
    let (y, _) = render(&mut e, 4, 2, 300, |_| 4.0 * rng.next());
    let peak = y.iter().flat_map(|c| c.iter()).fold(0.0_f32, |m, v| m.max(v.abs()));
    println!("overdriven peak {:.2} dBFS (ceiling -1)", db(peak));
    assert!(peak <= 10f32.powf(-1.0 / 20.0) + 1e-4, "peak {peak} exceeds the -1 dBFS ceiling");
    assert!(peak > 0.85, "the limiter must be working at the ceiling, not muting: {peak}");
}

// ── 5. real-time safety ───────────────────────────────────────────────────

#[test]
fn render_path_is_allocation_free_for_stereo_7_1_and_hrtf() {
    // Full chain (stage 4: direct, early taps, shared reverb bus, LFE, output stage), steady and
    // while the compute side keeps publishing new coefficients. The audio side is lock-free by
    // construction (`lockfree_engine_tests.rs` runs it against a held engine mutex).
    for layout in [PhysicalOutputLayout::Stereo, PhysicalOutputLayout::Surround714, PhysicalOutputLayout::Hrtf] {
        let ch = channels_of(&layout);
        let (mut e, ..) = shoebox_with(layout.clone());
        e.debug_audio_stage = 4;
        let mut input = AudioBuffer::new(1, BLOCK as u16);
        let mut rng = Noise(0x5555_6666);
        for i in 0..BLOCK {
            input.set(0, i as u16, 0.3 * rng.next());
        }
        let mut out = AudioBuffer::new(ch, BLOCK as u16);
        for _ in 0..20 {
            e.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
        }
        let mut total = 0;
        for round in 0..6 {
            e.set_scene_output_position(quasar_audio::quasar_core::scene_output::SceneOutputId(0), [2.0 + 0.3 * round as f32, 1.2, 2.0]);
            e.update_scene_spatial(); // compute side: not counted
            let (_, n) = common::count_allocs(|| {
                for _ in 0..8 {
                    e.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
                }
            });
            total += n;
        }
        assert_eq!(total, 0, "{layout:?}: the render path allocated {total} time(s)");
        assert!(out.peak() > 0.0);
    }
}

fn shoebox_with(layout: PhysicalOutputLayout) -> (SpatialAudioEngine, (), (), ()) {
    let (mut e, _, lis, _) = shoebox();
    // Replace the stereo listener by one with the requested layout.
    e.remove_listener(quasar_audio::quasar_core::scene_output::ListenerId(0));
    let l = e.add_listener(ListenerConfig { position: lis, heading: [0.0, 0.0, -1.0], physical_layout: layout });
    e.set_output_safety(l, OutputSafetyConfig::default());
    e.update_scene_spatial();
    (e, (), (), ())
}

