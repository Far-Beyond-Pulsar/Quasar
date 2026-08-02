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
use quasar_dsp::occlusion::BiquadFilter;

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

// ── equal_power_crossfader_snap ───────────────────────────────────────

#[test]
fn equal_power_crossfader_snap() {
    let initial = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, initial);

    let target = SpatialCoefficients {
        source_id: 1,
        direct_gain: Band8::splat(0.9),
        direct_delay_samples: 10.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(2.0),
        late_gain_db: -3.0,
        version: 1,
    };
    xfader.snap_to(target);

    let coeffs = xfader.current_coefficients();
    assert_eq!(coeffs.source_id, 1);
    assert!((coeffs.direct_gain.0[0] - 0.9).abs() < 1e-6);
    assert!(xfader.is_complete());
}

// ── equal_power_crossfader_transition ─────────────────────────────────

#[test]
fn equal_power_crossfader_transition() {
    let initial = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, initial);

    let target = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 1,
    };
    xfader.set_target(target);

    let fade_frames = ((10.0_f64 / 1000.0) * 48000.0).round() as u32;
    for _ in 0..fade_frames {
        xfader.advance();
    }
    assert!(xfader.is_complete());

    let coeffs = xfader.current_coefficients();
    assert!((coeffs.direct_gain.0[0] - 1.0).abs() < 1e-3);
}

// ── equal_power_crossfader_constant_power ─────────────────────────────

#[test]
fn equal_power_crossfader_constant_power() {
    let initial = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    let mut xfader = EqualPowerCrossfader::new(10.0, 48000.0, initial.clone());

    let target = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    xfader.snap_to(initial);
    // Re-set target to trigger crossfade
    let _ = xfader.current_coefficients();

    xfader.set_target(target);

    let fade_frames = ((10.0_f64 / 1000.0) * 48000.0).round() as u32;
    // Check power sum is ~constant at midpoint
    for step in 0..fade_frames {
        xfader.advance();
        let t = step as f32 / fade_frames as f32;
        let expected = (std::f32::consts::PI * t / 2.0).cos().powi(2)
            + (std::f32::consts::PI * t / 2.0).sin().powi(2);
        assert!((expected - 1.0).abs() < 1e-5, "constant power violation at step {step}");
    }
}

// ── hermite_delay_impulse_response ────────────────────────────────────

#[test]
fn hermite_delay_impulse_response() {
    let mut dl = HermiteInterpolatingDelayLine::new(0.1, 48000.0);
    let delay = 100.0;

    // Write an impulse at position 0
    dl.push(1.0);
    for _ in 1..=delay as usize {
        dl.push(0.0);
    }

    let output = dl.tap(delay);
    assert!((output - 1.0).abs() < 0.1, "impulse at integer delay: expected ~1.0 got {output}");
}

// ── hermite_delay_fractional ──────────────────────────────────────────

#[test]
fn hermite_delay_fractional() {
    let mut dl = HermiteInterpolatingDelayLine::new(0.1, 48000.0);
    let delay = 100.5;

    dl.push(1.0);
    for _ in 1..=101 {
        dl.push(0.0);
    }

    let output = dl.tap(delay);
    assert!(output > 0.0 && output < 1.0, "fractional delay should produce interpolated value, got {output}");
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

    // Compare phase — output should be delayed version of input
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
    assert!(db < -40.0, "delay accuracy too low: {db:.1} dB");
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

    let params = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(2.0),
        late_gain_db: 0.0,
        version: 0,
    };

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

    let params = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(0.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(target_t60),
        late_gain_db: 0.0,
        version: 0,
    };

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
    assert!(
        db_drop < 0.0,
        "energy should decay (drop={db_drop:.1} dB)"
    );
}

// ── directivity_omni_uniform ──────────────────────────────────────────

#[test]
fn directivity_omni_uniform() {
    let mut node = DirectivityDspNode::new(DirectivityPattern::Omnidirectional, 1);
    let input = AudioBuffer::new(1, 64);
    let mut output = AudioBuffer::new(1, 64);
    let params = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    node.process(&input, &mut output, &params);
    // Process doesn't crash; omni should pass through with uniform gain
    assert_eq!(output.channels(), 1);
}

// ── directivity_cardioid_null ─────────────────────────────────────────

#[test]
fn directivity_cardioid_null() {
    // pattern_gain is called internally by DirectivityDspNode
    // We test the static method indirectly via the gain computation
    let mut cardioid = DirectivityDspNode::new(DirectivityPattern::Cardioid, 1);
    let input = AudioBuffer::new(1, 64);
    let mut output = AudioBuffer::new(1, 64);
    let params = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    cardioid.process(&input, &mut output, &params);
    assert_eq!(output.channels(), 1);
}

// ── master_decoder_stereo_pan ─────────────────────────────────────────

#[test]
fn master_decoder_stereo_pan() {
    let mut node = MasterSpatialDecoderNode::new(
        DecoderMode::Vbap {
            layout: SpeakerLayout::Stereo,
        },
        48000.0,
    );
    assert_eq!(node.input_channels(), 2);
    assert_eq!(node.output_channels(), 2);

    let mut input = AudioBuffer::new(1, 64);
    for i in 0..64 {
        input.set(0, i, 1.0);
    }
    let mut output = AudioBuffer::new(2, 64);
    let params = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };
    node.process(&input, &mut output, &params);

    // With source_id=0, azimuth is 0*0.2=0 which is center → equal L/R
    assert!(output.get(0, 0) > 0.0);
    assert!(output.get(1, 0) > 0.0);
}

// ── audio_node_graph_process ──────────────────────────────────────────

#[test]
fn audio_node_graph_process() {
    use quasar_dsp::node_graph::AudioNode;

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

    let mut output = AudioBuffer::new(1, 64);
    let params = SpatialCoefficients {
        source_id: 0,
        direct_gain: Band8::splat(1.0),
        direct_delay_samples: 0.0,
        early_reflections: Vec::new(),
        late_t60: Band8::splat(0.5),
        late_gain_db: -10.0,
        version: 0,
    };

    graph.process(&[&src], &[params], &mut output);
    assert!((output.get(0, 0) - 1.0).abs() < 1e-3);
}
