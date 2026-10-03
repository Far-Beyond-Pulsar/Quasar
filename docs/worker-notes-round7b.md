# Worker notes, round 7b

## #76 resampling wiring: DONE
- New `crates/quasar/src/source_resampler.rs`: `FrameSource` trait (impl for `BufferedStream`, feature `streaming`), `ResampledSource` (preallocated; `render` / `render_block`; zero-fills + counts underruns), `DriftController` (PI on 1 s low-passed ring fill, tau 6 s / tau_i 24 s, +-5000 ppm clamp; resampler slews the trim).
- `BufferedStream` got additive `sample_block`, `written_frames`, `read_cursor` (one try_lock per block instead of per sample).
- Example (`examples/basic/src/audio_demo.rs`): linear interpolation replaced by `ResampledSource`; src/out `AudioBuffer`s are preallocated outside the callback. Drift control OFF for the file source (a file has no clock: the I/O thread refills as fast as drained, so fill carries no drift info).
- Tests: `crates/quasar/tests/source_resampler_tests.rs` (9): drift +-100/+-1000 ppm over 400 s, -300 ppm with 44.1->48 conversion, control experiment without controller underruns, controller bounds, looping seam (Common and Once policies) vs ideal tone, alloc-free render/render_block.
- Example `cargo check --release` OK with CARGO_TARGET_DIR=examples/basic/target-check.

## #83 / #149 output conversion: DONE
- `crates/quasar/src/output_stage.rs` (new), `Command::SetConversion`, `Garbage::Conv`, hooks in `render.rs` (render into physical scratch, matrix, THEN limiter), engine method `set_listener_output_layout(id, Option<PhysicalOutputLayout>) -> Result<(), MatrixError>` (separate impl block in lib.rs before `patch_bay_entry`).
- `ChannelMatrix` additions: `process_add`, `fade_to_zero`, `is_settled`, `is_silent`.
- `crates/quasar-dsp/src/channel_order.rs` (new): `DeviceChannelOrder`, `ChannelRemap` (WASAPI identity; ALSA FL FR RL RR C LFE (SL SR); CoreAudio 7.1 side/rear swap). ALSA/CoreAudio tables are assumptions (see module docs), unverified on hardware. Applied in the example's cpal callback only when not identity.
- Tests: `crates/quasar/tests/output_conversion_tests.rs` (9), 4 unit tests in channel_order.rs.
