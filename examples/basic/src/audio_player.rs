use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use quasar_audio::SpatialAudioEngine;
use quasar_dsp::audio_buffer::{AudioBuffer, DEFAULT_BLOCK_SIZE};
use quasar_dsp::late_reverb::FdnReverbNode;
use quasar_dsp::master_decoder::{DecoderMode, MasterSpatialDecoderNode, SpeakerLayout};
use quasar_dsp::occlusion::AirAbsorptionOcclusionNode;

pub struct WavData {
    pub sample_rate: u32,
    pub num_channels: u16,
    pub samples: Vec<f32>,
}

pub fn load_wav(path: impl AsRef<std::path::Path>) -> Result<WavData, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let bits = spec.bits_per_sample;
    let sample_rate = spec.sample_rate;
    let num_channels = spec.channels as u16;

    let samples: Vec<f32> = match bits {
        16 => reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect(),
        24 => reader
            .into_samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 8_388_608.0)
            .collect(),
        32 => reader
            .into_samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i32::MAX as f32)
            .collect(),
        _ => return Err(format!("unsupported bit depth: {}", bits).into()),
    };

    Ok(WavData {
        sample_rate,
        num_channels,
        samples,
    })
}

const NUM_SOURCES: usize = 4;

pub struct QuasarAudioPlayer {
    engine: Arc<Mutex<SpatialAudioEngine>>,
    wav: Arc<Mutex<WavPlaybackState>>,
    _stream: cpal::Stream,
}

struct WavPlaybackState {
    data: WavData,
    read_pos: f64,
    rate_ratio: f64,
    playing: bool,
}

impl QuasarAudioPlayer {
    pub fn new(wav_path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let wav_data = load_wav(&wav_path)?;

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no audio output device found")?;
        let out_config = device.default_output_config()?;
        let out_sample_rate = out_config.sample_rate().0;
        let out_channels = out_config.channels() as usize;
        let sr = out_sample_rate as f32;

        let mut engine = SpatialAudioEngine::new(NUM_SOURCES, sr, 15.0);
        let rate_ratio = wav_data.sample_rate as f64 / out_sample_rate as f64;

        // Build per-source DSP chains: occlusion → reverb → decoder
        {
            let graph = engine.dsp_graph();
            for src in 0..NUM_SOURCES {
                let occ = AirAbsorptionOcclusionNode::new(1, sr, 0.1);
                let occ_idx = graph.add_node(Box::new(occ));

                let rev = FdnReverbNode::new(1, sr);
                let rev_idx = graph.add_node(Box::new(rev));

                let dec = MasterSpatialDecoderNode::new(
                    DecoderMode::Vbap { layout: SpeakerLayout::Stereo },
                    sr,
                );
                let dec_idx = graph.add_node(Box::new(dec));

                // Each chain's connection carries its source_id so Phase 2
                // passes the correct spatial params.
                graph.connect_with_source(occ_idx, 0, rev_idx, 0, 1.0, src);
                graph.connect_with_source(rev_idx, 0, dec_idx, 0, 1.0, src);
            }
        }

        let engine = Arc::new(Mutex::new(engine));

        let wav_state = Arc::new(Mutex::new(WavPlaybackState {
            data: wav_data,
            read_pos: 0.0,
            rate_ratio,
            playing: true,
        }));

        let engine_clone = engine.clone();
        let wav_clone = wav_state.clone();
        let err_fn = |err: cpal::StreamError| eprintln!("Audio stream error: {}", err);

        let stream = device.build_output_stream(
            &out_config.config(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let total_frames = data.len() / out_channels;
                data.fill(0.0);
                if total_frames == 0 {
                    return;
                }

                let mut wav = wav_clone.lock().unwrap();
                if !wav.playing {
                    return;
                }

                let total_raw = wav.data.samples.len();
                let num_wav_ch = wav.data.num_channels as usize;
                let ratio = wav.rate_ratio;
                let mut frames_remaining = total_frames;
                let mut frame_offset = 0;

                while frames_remaining > 0 {
                    let block = (DEFAULT_BLOCK_SIZE as usize).min(frames_remaining);

                    // Build one mono input buffer per WAV channel
                    let mut inputs: Vec<AudioBuffer> = Vec::with_capacity(NUM_SOURCES);

                    for src in 0..NUM_SOURCES {
                        let wav_ch = src.min(num_wav_ch - 1);
                        let mut buf = AudioBuffer::new(1, block as u16);
                        {
                            let ch = buf.channel_mut(0);
                            for i in 0..block {
                                let wav_pos = wav.read_pos;
                                let frame_a = wav_pos.floor() as usize;
                                let frame_b = frame_a + 1;
                                let frac = (wav_pos - frame_a as f64) as f32;

                                let get_sample = |frame: usize| -> f32 {
                                    if frame < total_raw / num_wav_ch {
                                        wav.data.samples[frame * num_wav_ch + wav_ch]
                                    } else {
                                        0.0
                                    }
                                };

                                let sa = get_sample(frame_a);
                                let sb = get_sample(frame_b);
                                ch[i] = sa + (sb - sa) * frac;
                            }
                        }
                        inputs.push(buf);
                    }
                    let input_refs: Vec<&AudioBuffer> = inputs.iter().collect();

                    // Advance WAV read position
                    wav.read_pos += ratio * block as f64;
                    let total_frames_f = (total_raw / num_wav_ch) as f64;
                    if wav.read_pos >= total_frames_f {
                        wav.read_pos = 0.0;
                    }

                    // Process through Quasar
                    let mut output_buf = AudioBuffer::new(out_channels as u16, block as u16);
                    if let Ok(mut eng) = engine_clone.lock() {
                        eng.process_audio(&input_refs, &mut output_buf);
                    }

                    // Write to cpal
                    let play_ch = out_channels.min(output_buf.channels() as usize);
                    for i in 0..block {
                        let dst = frame_offset + i;
                        for ch in 0..play_ch {
                            data[dst * out_channels + ch] = output_buf.channel(ch as u16)[i];
                        }
                    }

                    frames_remaining -= block;
                    frame_offset += block;
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;

        Ok(Self {
            engine,
            wav: wav_state,
            _stream: stream,
        })
    }

    pub fn update_spatial(
        &self,
        source_positions: &[[f32; 3]],
        listener_position: [f32; 3],
    ) {
        if let Ok(engine) = self.engine.lock() {
            for (i, &pos) in source_positions.iter().enumerate().take(NUM_SOURCES) {
                use quasar_core::backend::SpatialQuery;
                engine.update_spatial(&SpatialQuery {
                    source_position: pos,
                    listener_position,
                    source_id: i as u32,
                });
            }
        }
    }

    pub fn is_playing(&self) -> bool {
        self.wav.lock().unwrap().playing
    }

    pub fn set_playing(&self, playing: bool) {
        self.wav.lock().unwrap().playing = playing;
    }
}
