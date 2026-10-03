use quasar_core::bands::Band8;
use quasar_core::param_exchange::SpatialCoefficients;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::crossfader::EqualPowerCrossfader;
use quasar_dsp::directivity::DirectivityDspNode;
use quasar_dsp::directivity::DirectivityPattern;
use quasar_dsp::fractional_delay::HermiteInterpolatingDelayLine;
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::master_decoder::{DecoderMode, MasterSpatialDecoderNode, SpeakerLayout};
use quasar_dsp::node_graph::AudioNode;
use quasar_dsp::node_graph::AudioNodeGraph;
use quasar_dsp::biquad::BiquadFilter;

// ── audio_buffer_new ──────────────────────────────────────────────────

#[test]
fn audio_buffer_new() {
    let buf = AudioBuffer::new(2, 64);
    assert_eq!(buf.channels(), 2);
    assert_eq!(buf.samples(), 64);
    for ch in 0..2 {
        for s in 0..64 {
            assert_eq!(buf.get(ch, s), 0.0);
        }
    }
}

// ── audio_buffer_read_write ───────────────────────────────────────────

#[test]
fn audio_buffer_read_write() {
    let mut buf = AudioBuffer::new(2, 128);
    buf.set(0, 0, 0.5);
    buf.set(1, 63, -0.25);
    assert!((buf.get(0, 0) - 0.5).abs() < 1e-6);
    assert!((buf.get(1, 63) + 0.25).abs() < 1e-6);
}

// ── audio_buffer_clear ────────────────────────────────────────────────

#[test]
fn audio_buffer_clear() {
    let mut buf = AudioBuffer::new(4, 64);
    buf.set(0, 0, 1.0);
    buf.set(3, 31, 0.5);
    buf.clear();
    for ch in 0..4 {
        for s in 0..64 {
            assert_eq!(buf.get(ch, s), 0.0);
        }
    }
}

// ── audio_buffer_copy_from ────────────────────────────────────────────

#[test]
fn audio_buffer_copy_from() {
    let mut src = AudioBuffer::new(2, 32);
    src.set(0, 0, 1.0);
    src.set(1, 15, 0.75);

    let mut dst = AudioBuffer::new(2, 32);
    dst.copy_from(&src);
    assert!((dst.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((dst.get(1, 15) - 0.75).abs() < 1e-6);
}

// ── audio_buffer_add_from ─────────────────────────────────────────────

#[test]
fn audio_buffer_add_from() {
    let mut a = AudioBuffer::new(1, 64);
    let mut b = AudioBuffer::new(1, 64);
    a.set(0, 0, 0.3);
    b.set(0, 0, 0.7);
    a.add_from(&b);
    assert!((a.get(0, 0) - 1.0).abs() < 1e-6);
}

// ── audio_buffer_apply_gain ───────────────────────────────────────────

#[test]
fn audio_buffer_apply_gain() {
    let mut buf = AudioBuffer::new(2, 32);
    buf.set(0, 0, 1.0);
    buf.set(1, 16, 2.0);
    buf.apply_gain(0.5);
    assert!((buf.get(0, 0) - 0.5).abs() < 1e-6);
    assert!((buf.get(1, 16) - 1.0).abs() < 1e-6);
}

// ── audio_buffer_rms_peak ─────────────────────────────────────────────

#[test]
fn audio_buffer_rms_peak() {
    let mut buf = AudioBuffer::new(2, 128);
    for ch in 0..2 {
        for s in 0..128 {
            buf.set(ch, s, 0.5);
        }
    }
    assert!((buf.rms() - 0.5).abs() < 1e-6);
    assert!((buf.peak() - 0.5).abs() < 1e-6);

    buf.set(0, 0, 0.9);
    assert!((buf.peak() - 0.9).abs() < 1e-6);
}

/// Build coefficients with the given identity / direct-path fields; everything
/// else is neutral. Keeps the tests independent of struct-literal churn.
fn coeffs(source_id: u32, gain: f32, delay: f32, azimuth: f32, version: u64) -> SpatialCoefficients {
    SpatialCoefficients {
        source_id,
        direct_gain: Band8::splat(gain),
        direct_delay_samples: delay,
        direct_azimuth: azimuth,
        direct_elevation: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        early_late_split_secs: 0.0,
        directivity_gain: quasar_core::bands::Band8::splat(1.0),
        version,
    }
}

/// 10 ms fade at 48 kHz = 480 frames.
const FADE_FRAMES: usize = 480;

// ── equal_power_crossfader_snap ───────────────────────────────────────

#[test]
fn equal_power_crossfader_snap() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 0.0, 0.0, 0.0, 0));

    let mut target = coeffs(1, 0.9, 10.0, 0.0, 1);
    target.late_t60 = Band8::splat(2.0);
    target.late_gain_db = -3.0;
    xfader.snap_to(target);

    let c = xfader.current_coefficients();
    assert_eq!(c.source_id, 1);
    assert!((c.direct_gain.0[0] - 0.9).abs() < 1e-6);
    assert!((c.direct_delay_samples - 10.0).abs() < 1e-6);
    assert!((c.late_t60.0[3] - 2.0).abs() < 1e-6);
    assert!((c.late_gain_db + 3.0).abs() < 1e-6);
    assert!(xfader.is_complete());
}

// ── equal_power_crossfader_transition ─────────────────────────────────

#[test]
fn equal_power_crossfader_transition() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 0.0, 0.0, 0.0, 0));
    xfader.set_target(&coeffs(0, 1.0, 0.0, 0.0, 1));
    assert!(!xfader.is_complete());

    // A whole fade worth of frames in one block completes it.
    let t = xfader.advance(FADE_FRAMES);
    assert!((t - 1.0).abs() < 1e-6);
    assert!(xfader.is_complete());
    assert!((xfader.current_coefficients().direct_gain.0[0] - 1.0).abs() < 1e-3);
}

// ── crossfader parameter blend (replaces the old cos^2 + sin^2 identity) ─

/// Parameters are blended linearly with t = frames / fade_frames. Advancing in
/// 256-frame blocks must hit exactly t = 256/480 and then t = 1 (target).
#[test]
fn crossfader_blend_is_linear_in_frames() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 0.0, 0.0, 0.0, 0));
    xfader.set_target(&coeffs(0, 1.0, 100.0, 0.0, 1));

    let t1 = xfader.advance(256);
    assert!((t1 - 256.0 / 480.0).abs() < 1e-6);
    let c = xfader.current_coefficients();
    assert!((c.direct_gain.0[0] - t1).abs() < 1e-5, "gain {} vs t {t1}", c.direct_gain.0[0]);
    assert!((c.direct_delay_samples - 100.0 * t1).abs() < 1e-3);
    assert!(!xfader.is_complete());

    let t2 = xfader.advance(256);
    assert!((t2 - 1.0).abs() < 1e-6);
    assert!(xfader.is_complete());
    assert!((xfader.current_coefficients().direct_gain.0[0] - 1.0).abs() < 1e-6);
}

/// A parameter that does not change must stay constant through a fade (an
/// equal-power sin/cos weighting would overshoot it by up to sqrt(2)); a
/// changing one must be monotonic.
#[test]
fn crossfader_constant_parameter_does_not_overshoot() {
    let mut a = coeffs(0, 0.0, 0.0, 0.0, 0);
    a.late_t60 = Band8::splat(1.5);
    let mut b = coeffs(0, 1.0, 0.0, 0.0, 1);
    b.late_t60 = Band8::splat(1.5);

    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, a);
    xfader.set_target(&b);

    let mut prev = 0.0_f32;
    for _ in 0..FADE_FRAMES / 16 {
        xfader.advance(16);
        let c = xfader.current_coefficients();
        assert!((c.late_t60.0[0] - 1.5).abs() < 1e-5, "constant parameter drifted: {}", c.late_t60.0[0]);
        let g = c.direct_gain.0[0];
        assert!(g >= prev - 1e-6 && g <= 1.0 + 1e-6, "gain not monotonic within [0,1]: {g}");
        prev = g;
    }
    assert!(xfader.is_complete());
}

/// A different source_id means a different emitter: no fade, snap.
#[test]
fn crossfader_source_change_snaps() {
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, coeffs(0, 0.0, 0.0, 0.0, 0));
    xfader.set_target(&coeffs(7, 1.0, 0.0, 0.0, 1));
    assert!(xfader.is_complete());
    assert_eq!(xfader.current_coefficients().source_id, 7);
    assert!((xfader.current_coefficients().direct_gain.0[0] - 1.0).abs() < 1e-6);
}

// ── hermite_delay_impulse_response ────────────────────────────────────

#[test]
fn hermite_delay_impulse_response() {
    let mut dl = HermiteInterpolatingDelayLine::new(0.1, 48000.0);
    let delay = 100.0;

    // Impulse followed by 100 zeros: the impulse is exactly 100 pushes old.
    dl.push(1.0);
    for _ in 1..=delay as usize {
        dl.push(0.0);
    }

    assert!((dl.tap(delay) - 1.0).abs() < 1e-6, "impulse at integer delay");
    assert!(dl.tap(delay - 1.0).abs() < 1e-6);
    assert!(dl.tap(delay + 1.0).abs() < 1e-6);
}

// ── hermite_delay_fractional ──────────────────────────────────────────

#[test]
fn hermite_delay_fractional() {
    let mut dl = HermiteInterpolatingDelayLine::new(0.1, 48000.0);

    // Impulse that is 101 pushes old, read at 100.5 (halfway to it).
    dl.push(1.0);
    for _ in 1..=101 {
        dl.push(0.0);
    }

    // Catmull-Rom of a unit impulse at the next-older neighbour, t = 0.5:
    // h01(0.5) + h10(0.5) * 0.5 = 0.5 + 0.125 * 0.5.
    let output = dl.tap(100.5);
    assert!((output - 0.5625).abs() < 1e-5, "expected 0.5625, got {output}");
}

// ── hermite_delay_accuracy ────────────────────────────────────────────

#[test]
fn hermite_delay_accuracy() {
    let sample_rate = 48000.0;
    let mut dl = HermiteInterpolatingDelayLine::new(0.05, sample_rate);
    let delay_samples = 50.0;
    let freq = 200.0;
    let n = 256;

    // Generate input sine wave
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
        .collect();

    // Process through delay line
    let mut output = vec![0.0; n];
    dl.process_channel(&input, &mut output, delay_samples);

    // Output must be the input delayed by exactly 50 samples.
    let delay_int = delay_samples as usize;
    let mut error_power = 0.0;
    let mut signal_power = 0.0;
    for i in delay_int..n {
        let err = output[i] - input[i - delay_int];
        error_power += err * err;
        signal_power += input[i - delay_int] * input[i - delay_int];
    }
    let db = if signal_power > 0.0 {
        10.0 * (error_power / signal_power).log10()
    } else {
        -200.0
    };
    assert!(db < -100.0, "delay accuracy too low: {db:.1} dB");
}

// ── biquad_filter_lowpass ─────────────────────────────────────────────

#[test]
fn biquad_filter_lowpass() {
    let mut filter = BiquadFilter::new();
    filter.set_lowpass(500.0, 48000.0);

    let low_freq_in: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 48000.0).sin())
        .collect();
    let high_freq_in: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 8000.0 * i as f32 / 48000.0).sin())
        .collect();

    let low_out: Vec<f32> = low_freq_in.iter().map(|&s| filter.process(s)).collect();
    let low_rms = (low_out.iter().map(|s| s * s).sum::<f32>() / low_out.len() as f32).sqrt();

    let mut filter2 = BiquadFilter::new();
    filter2.set_lowpass(500.0, 48000.0);
    let high_out: Vec<f32> = high_freq_in.iter().map(|&s| filter2.process(s)).collect();
    let high_rms = (high_out.iter().map(|s| s * s).sum::<f32>() / high_out.len() as f32).sqrt();

    assert!(
        high_rms < low_rms * 0.3,
        "lowpass should attenuate high frequencies more: low_rms={low_rms}, high_rms={high_rms}"
    );
}

// ── biquad_filter_response ────────────────────────────────────────────

#[test]
fn biquad_filter_response() {
    let mut filter = BiquadFilter::new();
    filter.set_lowpass(1000.0, 48000.0);

    let dc_in = vec![1.0; 256];
    let dc_out: Vec<f32> = dc_in.iter().map(|&s| filter.process(s)).collect();
    let dc_gain = dc_out[dc_out.len() - 1];
    assert!((dc_gain - 1.0).abs() < 0.05, "DC gain should be ~1.0, got {dc_gain}");
}


// ── fdn_reverb_stability ──────────────────────────────────────────────

#[test]
fn fdn_reverb_stability() {
    let mut reverb = FdnReverbNode::new(1, 48000.0);
    reverb.set_t60(&Band8::splat(2.0));

    let mut output = AudioBuffer::new(1, 256);

    let mut params = coeffs(0, 0.0, 0.0, 0.0, 0);
    params.late_t60 = Band8::splat(2.0);
    params.late_gain_db = 0.0;

    // First block: impulse
    let mut impulse_buf = AudioBuffer::new(1, 256);
    impulse_buf.set(0, 0, 1.0);
    reverb.process(&impulse_buf, &mut output, &params);

    let first_peak = output.peak();

    // Process silence to see if energy grows unbounded
    let silence = AudioBuffer::new(1, 256);
    for _ in 0..100 {
        reverb.process(&silence, &mut output, &params);
    }

    let later_peak = output.peak();
    assert!(
        later_peak < first_peak * 2.0,
        "reverb should not amplify over time: first={first_peak}, later={later_peak}"
    );
}

// ── fdn_t60_approximation ─────────────────────────────────────────────

#[test]
fn fdn_t60_approximation() {
    let mut reverb = FdnReverbNode::new(1, 48000.0);
    let target_t60 = 1.0;
    reverb.set_t60(&Band8::splat(target_t60));

    let mut params = coeffs(0, 0.0, 0.0, 0.0, 0);
    params.late_t60 = Band8::splat(target_t60);
    params.late_gain_db = 0.0;

    // Inject impulse and measure decay
    let mut impulse_buf = AudioBuffer::new(1, 256);
    impulse_buf.set(0, 0, 1.0);
    let mut output = AudioBuffer::new(1, 256);
    reverb.process(&impulse_buf, &mut output, &params);

    let initial_rms = output.rms().max(1e-10);

    // Let it ring for ~1 second
    let silence = AudioBuffer::new(1, 256);
    let num_blocks = (48000 / 256) as usize;
    let mut final_rms = initial_rms;
    for _ in 0..num_blocks {
        reverb.process(&silence, &mut output, &params);
        final_rms = output.rms().max(1e-10);
    }

    let db_drop = 20.0 * (final_rms / initial_rms).log10();
    // The first block includes the dry impulse, so this is a coarse bound (the
    // current FDN measures ~-80 dB here); exact T60 accuracy belongs to the
    // reverb work, not this test.
    assert!(db_drop < -40.0, "energy should decay by 1 s (drop={db_drop:.1} dB)");
}

// ── directivity_omni_uniform ──────────────────────────────────────────

#[test]
fn directivity_omni_uniform() {
    let mut node = DirectivityDspNode::new(DirectivityPattern::Omnidirectional, 1);
    let mut input = AudioBuffer::new(1, 64);
    for i in 0..64 {
        input.set(0, i, (i as f32 * 0.1).sin());
    }
    let mut output = AudioBuffer::new(1, 64);
    // Omni radiates equally everywhere, whatever the emission angle (and source id) is.
    for (id, az, el) in [(0u32, 0.0f32, 0.0f32), (5, 1.0, 0.3), (31, std::f32::consts::PI, -0.7)] {
        node.set_angle(az, el);
        node.process(&input, &mut output, &coeffs(id, 1.0, 0.0, 0.0, 0));
        for i in 0..64 {
            assert!((output.get(0, i) - input.get(0, i)).abs() < 1e-6, "omni must pass through");
        }
    }
}

// ── directivity_cardioid_null ─────────────────────────────────────────

/// The node's angle is an explicit input (no more `source_id * 0.1` placeholder): the cardioid is
/// 1 on axis, 1/2 at 90 degrees and a null at 180, and the source id has no influence.
#[test]
fn directivity_cardioid_null() {
    let mut cardioid = DirectivityDspNode::new(DirectivityPattern::Cardioid, 1);
    let mut input = AudioBuffer::new(1, 64);
    for i in 0..64 {
        input.set(0, i, 1.0);
    }
    let mut output = AudioBuffer::new(1, 64);
    let pi = std::f32::consts::PI;

    cardioid.set_angle(0.0, 0.0);
    cardioid.process(&input, &mut output, &coeffs(0, 1.0, 0.0, 0.0, 0));
    assert!((output.get(0, 10) - 1.0).abs() < 1e-6, "cardioid on-axis gain is 1");

    cardioid.set_angle(pi, 0.0);
    cardioid.process(&input, &mut output, &coeffs(0, 1.0, 0.0, 0.0, 0));
    assert!(output.get(0, 10).abs() < 1e-6, "cardioid rear is a null, got {}", output.get(0, 10));

    cardioid.set_angle(pi / 2.0, 0.0);
    cardioid.process(&input, &mut output, &coeffs(31, 1.0, 0.0, 0.0, 0));
    assert!((output.get(0, 10) - 0.5).abs() < 1e-6, "side is -6 dB, whatever the source id");

    // Elevation counts: 90 degrees up is also 90 degrees off axis.
    cardioid.set_angle(0.0, pi / 2.0);
    cardioid.process(&input, &mut output, &coeffs(0, 1.0, 0.0, 0.0, 0));
    assert!((output.get(0, 10) - 0.5).abs() < 1e-6);

    // Figure-8 and SH patterns.
    let mut f8 = DirectivityDspNode::new(DirectivityPattern::Figure8, 1);
    f8.set_angle(pi, 0.0);
    assert!((f8.current_gain() - 1.0).abs() < 1e-6, "figure-8 radiates equally to the rear");
    f8.set_angle(pi / 2.0, 0.0);
    assert!(f8.current_gain().abs() < 1e-6, "figure-8 null at the sides");
    let sh = DirectivityPattern::SphericalHarmonics { weights: vec![0.5, 0.0, 0.0, 0.5], order: 1 };
    assert!((DirectivityDspNode::compute_gain(&sh, 0.0, 0.0) - 1.0).abs() < 1e-6);
    assert!((DirectivityDspNode::compute_gain(&sh, pi, 0.0)).abs() < 1e-6);
    // Non-finite angles read as on axis.
    cardioid.set_angle(f32::NAN, f32::INFINITY);
    assert_eq!(cardioid.current_gain(), 1.0);
}

// ── master_decoder_stereo_pan ─────────────────────────────────────────

fn stereo_decoder() -> MasterSpatialDecoderNode {
    MasterSpatialDecoderNode::new(DecoderMode::Vbap { layout: SpeakerLayout::Stereo }, 48000.0)
}

fn render_dc(azimuth: f32) -> (f32, f32) {
    let mut node = stereo_decoder();
    let mut input = AudioBuffer::new(1, 64);
    for i in 0..64 {
        input.set(0, i, 1.0);
    }
    let mut output = AudioBuffer::new(2, 64);
    node.process(&input, &mut output, &coeffs(0, 1.0, 0.0, azimuth, 0));
    (output.get(0, 63), output.get(1, 63))
}

#[test]
fn master_decoder_stereo_pan() {
    let node = stereo_decoder();
    assert_eq!(node.input_channels(), 2);
    assert_eq!(node.output_channels(), 2);

    // Centre: equal L/R, constant power (1/sqrt(2) each).
    let (l, r) = render_dc(0.0);
    assert!((l - r).abs() < 1e-4, "centre should be balanced: {l} vs {r}");
    assert!((l * l + r * r - 1.0).abs() < 1e-3, "constant power at centre: {}", l * l + r * r);

    // +azimuth is toward +X = right; -azimuth left; power stays constant.
    for az in [0.2_f32, 0.4] {
        let (l, r) = render_dc(az);
        assert!(r > l, "az {az}: right should dominate ({l}, {r})");
        assert!((l * l + r * r - 1.0).abs() < 1e-3);
        let (l2, r2) = render_dc(-az);
        assert!(l2 > r2, "az {}: left should dominate ({l2}, {r2})", -az);
        assert!((l - r2).abs() < 1e-4 && (r - l2).abs() < 1e-4, "pan must be symmetric");
    }

    // Hard right at the speaker: left is silent.
    let (l, r) = render_dc(30.0_f32.to_radians());
    assert!(l.abs() < 1e-3 && (r - 1.0).abs() < 1e-3, "at the right speaker: ({l}, {r})");
}

// ── audio_node_graph_process ──────────────────────────────────────────

#[test]
fn audio_node_graph_process() {
    struct GainNode {
        gain: f32,
        ch: u16,
    }

    impl AudioNode for GainNode {
        fn process(
            &mut self,
            input: &AudioBuffer,
            output: &mut AudioBuffer,
            _params: &SpatialCoefficients,
        ) {
            for ch in 0..self.ch.min(input.channels()).min(output.channels()) as usize {
                for i in 0..input.samples() as usize {
                    output.channel_mut(ch as u16)[i] = input.channel(ch as u16)[i] * self.gain;
                }
            }
        }

        fn reset(&mut self) {}
        fn input_channels(&self) -> u16 { self.ch }
        fn output_channels(&self) -> u16 { self.ch }
    }

    let mut graph = AudioNodeGraph::new();
    let n0 = graph.add_node(Box::new(GainNode { gain: 0.5, ch: 1 }));
    let n1 = graph.add_node(Box::new(GainNode { gain: 2.0, ch: 1 }));

    graph.connect_direct(n0, 0, n1, 0);

    let mut src = AudioBuffer::new(1, 64);
    src.set(0, 0, 1.0);
    src.set(0, 5, -0.5);

    let mut output = AudioBuffer::new(1, 64);
    let params = coeffs(0, 1.0, 0.0, 0.0, 0);

    graph.process(&[&src], &[params], &mut output);
    // 0.5 * 2.0 = unity through the chain; only the leaf node is mixed out.
    assert!((output.get(0, 0) - 1.0).abs() < 1e-3);
    assert!((output.get(0, 5) + 0.5).abs() < 1e-3);
    assert!(output.get(0, 1).abs() < 1e-6);
}
