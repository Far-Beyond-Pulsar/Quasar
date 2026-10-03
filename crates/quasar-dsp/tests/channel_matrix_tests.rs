//! #83: output format-conversion matrix. Published BS.775 coefficients, power checks, ramped
//! changes, errors instead of dropped channels.

use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::channel_matrix::{downmix_gains, layout_channel_count, ChannelMatrix, MatrixError, BS775_K};
use quasar_dsp::master_decoder::SpeakerLayout;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;
const K: f32 = std::f32::consts::FRAC_1_SQRT_2;

fn row(g: &[f32], cols: usize, o: usize) -> &[f32] {
    &g[o * cols..(o + 1) * cols]
}

#[test]
fn bs775_five_one_to_stereo_coefficients() {
    // Lo = L + 0.7071 C + 0.7071 Ls, Ro = R + 0.7071 C + 0.7071 Rs; no LFE.
    let g = downmix_gains(&SpeakerLayout::Surround51, &SpeakerLayout::Stereo).unwrap();
    assert_eq!(g.len(), 2 * 6);
    assert_eq!(row(&g, 6, 0), &[1.0, 0.0, K, 0.0, K, 0.0]);
    assert_eq!(row(&g, 6, 1), &[0.0, 1.0, K, 0.0, 0.0, K]);
    assert!((BS775_K - 0.707_106_78).abs() < 1e-7);
}

#[test]
fn quad_to_stereo_and_seven_one_to_five_one_coefficients() {
    let g = downmix_gains(&SpeakerLayout::Quad, &SpeakerLayout::Stereo).unwrap();
    assert_eq!(row(&g, 4, 0), &[1.0, 0.0, K, 0.0]);
    assert_eq!(row(&g, 4, 1), &[0.0, 1.0, 0.0, K]);

    // 7.1 (FL FR C LFE BL BR SL SR) -> 5.1 (FL FR C LFE BL BR)
    let g = downmix_gains(&SpeakerLayout::Surround714, &SpeakerLayout::Surround51).unwrap();
    assert_eq!(g.len(), 6 * 8);
    for o in 0..4 {
        let mut want = [0.0_f32; 8];
        want[o] = 1.0;
        assert_eq!(row(&g, 8, o), &want, "front / centre / LFE pass through");
    }
    assert_eq!(row(&g, 8, 4), &[0.0, 0.0, 0.0, 0.0, K, 0.0, K, 0.0]);
    assert_eq!(row(&g, 8, 5), &[0.0, 0.0, 0.0, 0.0, 0.0, K, 0.0, K]);

    // 7.1 -> stereo is the product: FL 1, C K, BL K * K + SL K * K = 1/2 ...
    let g = downmix_gains(&SpeakerLayout::Surround714, &SpeakerLayout::Stereo).unwrap();
    assert_eq!(g.len(), 2 * 8);
    let want_l = [1.0, 0.0, K, 0.0, 0.5, 0.0, 0.5, 0.0];
    for (a, b) in row(&g, 8, 0).iter().zip(&want_l) {
        assert!((a - b).abs() < 1e-6, "{:?}", row(&g, 8, 0));
    }
}

#[test]
fn identity_for_identical_layouts_and_errors_for_unsupported_conversions() {
    for (l, n) in [(SpeakerLayout::Stereo, 2), (SpeakerLayout::Quad, 4), (SpeakerLayout::Surround51, 6), (SpeakerLayout::Surround714, 8)] {
        assert_eq!(layout_channel_count(&l), n);
        let g = downmix_gains(&l, &l).unwrap();
        for o in 0..n {
            for i in 0..n {
                assert_eq!(g[o * n + i], if o == i { 1.0 } else { 0.0 });
            }
        }
    }
    // Upmix, 5.1 -> quad, custom layouts: an error, never a silent channel drop.
    let custom = SpeakerLayout::Custom { positions: vec![[0.0, 0.0, -1.0]; 3] };
    for (a, b) in [
        (SpeakerLayout::Stereo, SpeakerLayout::Surround51),
        (SpeakerLayout::Surround51, SpeakerLayout::Quad),
        (SpeakerLayout::Surround714, SpeakerLayout::Quad),
        (SpeakerLayout::Quad, SpeakerLayout::Surround51),
        (custom.clone(), SpeakerLayout::Stereo),
        (SpeakerLayout::Stereo, custom.clone()),
    ] {
        match downmix_gains(&a, &b) {
            Err(MatrixError::UnsupportedConversion { from_channels, to_channels }) => {
                assert_eq!((from_channels, to_channels), (layout_channel_count(&a), layout_channel_count(&b)));
            }
            other => panic!("{a:?} -> {b:?} must be an error, got {other:?}"),
        }
    }
    assert!(ChannelMatrix::new(0, 2, SR, 10.0).is_err());
    assert!(ChannelMatrix::new(2, 33, SR, 10.0).is_err());
    let mut m = ChannelMatrix::new(2, 2, SR, 10.0).unwrap();
    assert_eq!(m.set_matrix(&[1.0; 3]), Err(MatrixError::BadShape { given: 3, expected: 4 }));
    assert!(format!("{}", MatrixError::BadShape { given: 3, expected: 4 }).contains("expected 4"));
}

#[test]
fn power_of_the_standard_downmixes() {
    // Per-input power gain `sum_o M^2`: front and centre keep unit power (centre = 2 K^2),
    // a surround that goes to one side loses 3 dB by design, the LFE is dropped.
    let m = ChannelMatrix::from_downmix(&SpeakerLayout::Surround51, &SpeakerLayout::Stereo, SR, 10.0).unwrap();
    let want = [1.0, 1.0, 1.0, 0.0, 0.5, 0.5];
    for (i, w) in want.iter().enumerate() {
        assert!((m.input_power(i) - w).abs() < 1e-6, "5.1 input {i}: {}", m.input_power(i));
    }
    let m = ChannelMatrix::from_downmix(&SpeakerLayout::Quad, &SpeakerLayout::Stereo, SR, 10.0).unwrap();
    for (i, w) in [1.0, 1.0, 0.5, 0.5].iter().enumerate() {
        assert!((m.input_power(i) - w).abs() < 1e-6);
    }
    // 7.1 -> 5.1: a rear / side surround alone loses 3 dB, the uncorrelated pair sums to unit power.
    let m = ChannelMatrix::from_downmix(&SpeakerLayout::Surround714, &SpeakerLayout::Surround51, SR, 10.0).unwrap();
    for i in 0..8 {
        let want = if i >= 4 { 0.5 } else { 1.0 };
        assert!((m.input_power(i) - want).abs() < 1e-6, "7.1 input {i}");
    }
    let ls_row: f32 = (0..8).map(|i| m.gain(4, i).powi(2)).sum();
    assert!((ls_row - 1.0).abs() < 1e-6, "an uncorrelated BL + SL pair feeds Ls with unit power: {ls_row}");
}

struct Noise(u32);
impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

fn run(m: &mut ChannelMatrix, blocks: usize, mut fill: impl FnMut(usize, usize) -> f32) -> Vec<Vec<f32>> {
    let (nin, nout) = (m.input_channels(), m.output_channels());
    let mut out = vec![Vec::new(); nout];
    let (mut input, mut o) = (AudioBuffer::new(nin as u16, BLOCK as u16), AudioBuffer::new(nout as u16, BLOCK as u16));
    for b in 0..blocks {
        for c in 0..nin {
            for i in 0..BLOCK {
                input.set(c as u16, i as u16, fill(c, b * BLOCK + i));
            }
        }
        m.process(&input, &mut o);
        for c in 0..nout {
            out[c].extend_from_slice(&o.channel(c as u16)[..BLOCK]);
        }
    }
    out
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

#[test]
fn rendered_power_matches_the_matrix() {
    // Uncorrelated unit-variance noise in every channel: output power per channel = sum of
    // squared gains of its row.
    let mut m = ChannelMatrix::from_downmix(&SpeakerLayout::Surround714, &SpeakerLayout::Stereo, SR, 10.0).unwrap();
    let mut rngs: Vec<Noise> = (0..8).map(|c| Noise(0x1234_5678 + 977 * c as u32)).collect();
    let mut cache = vec![Vec::new(); 8];
    let y = run(&mut m, 200, |c, t| {
        while cache[c].len() <= t {
            let v = rngs[c].next();
            cache[c].push(v);
        }
        cache[c][t]
    });
    let in_rms = rms(&cache[0]);
    for o in 0..2 {
        let want: f32 = (0..8).map(|i| m.gain(o, i).powi(2)).sum::<f32>().sqrt() * in_rms;
        let got = rms(&y[o]);
        assert!((got / want - 1.0).abs() < 0.03, "output {o}: rms {got} vs {want}");
    }
    // The LFE (slot 3) is not in the stereo downmix at all.
    let mut only_lfe = ChannelMatrix::from_downmix(&SpeakerLayout::Surround51, &SpeakerLayout::Stereo, SR, 10.0).unwrap();
    let y = run(&mut only_lfe, 4, |c, t| if c == 3 { (t as f32 * 0.1).sin() } else { 0.0 });
    assert!(y.iter().all(|c| c.iter().all(|&v| v == 0.0)), "the LFE is not mixed into the stereo downmix");
}

#[test]
fn centre_channel_is_split_at_minus_3_db_with_unit_total_power() {
    let mut m = ChannelMatrix::from_downmix(&SpeakerLayout::Surround51, &SpeakerLayout::Stereo, SR, 10.0).unwrap();
    let y = run(&mut m, 4, |c, t| if c == 2 { (t as f32 * 0.05).sin() } else { 0.0 });
    for i in 0..BLOCK * 4 {
        assert!((y[0][i] - K * (i as f32 * 0.05).sin()).abs() < 1e-6 && y[0][i] == y[1][i]);
    }
}

#[test]
fn matrix_changes_are_ramped_per_sample() {
    // Switch from the identity-like (front pair only) to the full downmix with a DC surround
    // present: the output must glide over the 10 ms ramp, sample by sample, no step.
    let mut m = ChannelMatrix::new(6, 2, SR, 10.0).unwrap();
    let mut front = vec![0.0_f32; 12];
    front[0] = 1.0; // L <- FL
    front[7] = 1.0; // R <- FR
    m.set_matrix_immediate(&front).unwrap();
    // Block 0: steady. Then the downmix target is set before block 1.
    let (mut input, mut out) = (AudioBuffer::new(6, BLOCK as u16), AudioBuffer::new(2, BLOCK as u16));
    for c in [0u16, 1, 2, 4, 5] {
        for i in 0..BLOCK {
            input.set(c, i as u16, 0.5);
        }
    }
    m.process(&input, &mut out);
    assert_eq!(out.channel(0)[BLOCK - 1], 0.5);
    m.switch_downmix(&SpeakerLayout::Surround51, &SpeakerLayout::Stereo).unwrap();
    let mut left = Vec::new();
    for _ in 0..4 {
        m.process(&input, &mut out);
        left.extend_from_slice(&out.channel(0)[..BLOCK]);
    }
    // Final left = 0.5 (1 + K + K); ramp length 480 samples.
    let end = 0.5 * (1.0 + 2.0 * K);
    assert!((left[3 * BLOCK] - end).abs() < 1e-5, "ramp complete: {} vs {end}", left[3 * BLOCK]);
    let max_step = left.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
    let slope = (end - 0.5) / 480.0;
    assert!(max_step <= slope * 1.01 + 1e-7, "max step {max_step} vs ramp slope {slope}");
    assert!(max_step > 0.5 * slope, "it really ramps: {max_step}");
    // Chunking independence: the same switch rendered in odd blocks gives the same samples.
    let mut m2 = ChannelMatrix::new(6, 2, SR, 10.0).unwrap();
    m2.set_matrix_immediate(&front).unwrap();
    m2.process(&input, &mut out);
    m2.switch_downmix(&SpeakerLayout::Surround51, &SpeakerLayout::Stereo).unwrap();
    let mut small_in = AudioBuffer::new(6, 100);
    for c in [0u16, 1, 2, 4, 5] {
        for i in 0..100 {
            small_in.set(c, i, 0.5);
        }
    }
    let mut small_out = AudioBuffer::new(2, 100);
    let mut left2 = Vec::new();
    for _ in 0..(4 * BLOCK / 100) {
        m2.process(&small_in, &mut small_out);
        left2.extend_from_slice(&small_out.channel(0)[..100]);
    }
    for i in 0..left2.len() {
        assert!((left2[i] - left[i]).abs() < 1e-5, "chunking changes sample {i}: {} vs {}", left2[i], left[i]);
    }
}
