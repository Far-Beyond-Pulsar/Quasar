# Objective acoustic validation

Quasar's rendered output is measured offline and deterministically (no audio device, seeded
noise, fixed block size) by two integration-test files of `crates/quasar`:

| file | what it covers |
|---|---|
| `tests/acoustic_validation.rs` | localisation, propagation latency, room acoustics (RT60 / EDT / C50 / D50, image sources), loudness (BS.1770) and peak ceiling, allocation-free render path |
| `tests/direct_vs_room_levels.rs` | direct / early / reverb balance of the demo cathedral, the reverb calibration formula, the demo mix targets |

Both are part of `cargo test` and of CI (see `.github/workflows/rust.yml`). Everything runs in a
second or two in release; in debug allow up to about 30 s.

```text
cargo test -p quasar-audio --release --test acoustic_validation -- --nocapture
cargo test -p quasar-audio --release --test direct_vs_room_levels -- --nocapture
cargo test -p quasar-audio --release --test direct_vs_room_levels -- --ignored --nocapture   # directivity sweep
```

`--nocapture` prints the measured numbers next to every assertion; the printed tables are the
reference for "what does the engine currently do".

## How a scene is rendered

The engine is run with `process_audio_scene` in 256-frame blocks. `debug_audio_stage` isolates the
layers: 2 = direct path, 3 = + early reflections, 4 = full (+ shared reverb bus). The difference of
two stages (of the SIGNALS, not the powers) is the layer's contribution. The output limiter is
switched off where linear levels are measured (`OutputSafetyConfig { enabled: false, .. }`).

Geometry-free scenes use the stub backend (distance law only, constant late field of -10 dB and
T60 0.5 s). Room scenes use `CpuSimdComputeBackend` (image sources) or `BakedOnly` sampling of a
constant probe grid (known T60 and volume).

## Metrics

### Localisation

* **Velocity vector** `sum_i g_i u_i` of the per-channel amplitude gains `g_i` (RMS of each output
  channel over the RMS of the input, divided by the `1/d` gain) and speaker unit vectors `u_i`.
  For amplitude panning between an adjacent speaker pair the velocity vector points exactly at
  the source, so the direction error is an assertion at `1.5 deg` (measured: `0.00 .. 0.01 deg`
  for stereo, quad, 5.1 and 7.1 over a full 360 degree sweep in 15 degree steps; stereo within
  +-30 degrees).
* **Panning law**: the total power over the channels (`sum g_i^2`) is the same for every
  azimuth within 0.1 dB (constant-power panning) and equals `-0.24 dB` re the `1/d` gain, which
  is the air absorption of broadband noise at 2 m. The LFE slot of 5.1 / 7.1 receives nothing.
* **HRTF listener** (parametric binaural renderer): **ITD** = lag of the cross-correlation peak of
  the two ears (parabolic interpolation) against the Woodworth model
  `(a / c)(theta + sin theta)`, `a = 0.0875 m`: tolerance 60 us (measured within 16 us from 15 to
  90 degrees, mirrored for the left side). **ILD** = ear level difference in dB: positive toward
  the source ear, growing with azimuth (non-decreasing within 1 dB: the pinna EQ makes it plateau
  beyond 60 degrees), 3.8 dB at 15 degrees to 14.5 dB at 60 degrees.

### Latency

An impulse at 1 / 3.43 / 10 / 30 m must peak at `t0 + d * fs / c` within 1.5 samples (zero added
latency in the default output stage; a configured limiter look-ahead is covered by
`output_safety_tests.rs::look_ahead_latency_reported_equals_measured`), with total energy within
-3 .. +0.5 dB of `1 / d^2` (air absorption can only lower it).

### Room acoustics (Schroeder integration)

The late field of a `BakedOnly` engine with a constant probe grid of known `T60` and volume `V`
is measured from the impulse response `h(t)` (total over the output channels):

* **Schroeder curve** `E(t) = 10 log10( int_t^inf h^2 / int_0^inf h^2 )`.
* **RT60 (T20)**: linear regression of `E(t)` between -5 and -25 dB, extrapolated to -60 dB.
  Tolerance 10 % of the probe T60 (measured 0.99 / 2.00 / 4.02 s for 1 / 2 / 4 s).
* **EDT**: regression between 0 and -10 dB, times 6. Tolerance 20 % (measured 0.91 / 1.90 / 3.93 s;
  the FDN builds up its echo density over the first tens of milliseconds, so EDT is a little
  short, as it is in real rooms).
* **C50 / D50**: clarity `C50 = 10 log10(E[0, 50 ms] / E[50 ms, inf))` and definition
  `D50 = E[0, 50 ms] / E[0, inf)`, `t = 0` at the DIRECT arrival. The engine starts the diffuse
  field at the early / late split (50 ms for these probes), so the first 50 ms are the direct
  sound only and `C50 = -(late level re the 1 m direct sound)`: the test asserts the measured C50
  against `-10 log10(312.2 T60 / V)` within 1 dB (measured +13.8 / +11.0 / +10.6 dB).
* **Image sources**: in a 10 x 4 x 8 m shoebox with one emitter and one listener the six
  first-order reflection delays must match the mirror-image geometry within the rendering
  accuracy, each arrival must carry `(1 - alpha) / d^2` of energy within -3 .. +0.5 dB (the
  engine's 4-point Hermite fractional delay low-passes an impulse by up to -1.9 dB of energy
  depending on the fractional delay; the backend's own tests pin its per-band gains to 0.2 %),
  taps past the early / late handover are faded to the late field, and nothing else arrives.
  Directions of reflections are covered by `reflection_spatial_tests.rs`.

### Reverb level calibration

`rev/direct` power at a point = `312.2 T60 / (V Q)` re the direct sound of the same emitter at 1 m
(`quasar_core::reverb_model`, Q = directivity factor). `direct_vs_room_levels.rs` asserts the
engine's measured reverb against it within 1 dB for six (T60, V) combinations (measured within
0.6 dB, engine slightly below) and for a cardioid emitter (Q = `1 / diffuse_send_gain^2`).

### Loudness and peak

* **K-weighted integrated loudness** implemented in the test (no dependency): ITU-R BS.1770-4,
  two biquads at 48 kHz (high shelf + RLB high-pass), 400 ms blocks with 75 % overlap, absolute
  gate -70 LUFS, relative gate -10 LU, channel weights 1.0 (the stereo / front channels used).
  Self-check against the standard's calibration point: a 997 Hz sine at -20 dBFS peak in both
  stereo channels is `-20.0 LUFS` (0 dBFS in one channel: `-3.01 LUFS`).
* **Distance law in LU**: the direct path loses 6.02 LU per doubling of the distance (measured
  6.2 LU for 1 -> 2 m, 6.6 LU for 2 -> 4 m; the excess is air absorption, which the K-weighting's
  HF shelf emphasises). Accepted range 6.0 .. 7.0.
* **Peak ceiling**: noise 20 dB hotter than full scale through the DEFAULT output stage never
  exceeds -1 dBFS (sample peak) and does reach it (the limiter works, it does not mute).

### Real-time safety

* `render_path_is_allocation_free_for_stereo_7_1_and_hrtf`: full chain (direct, early taps, shared
  reverb bus, LFE, output stage) with the counting global allocator of `tests/common`, steady and
  while the compute side keeps publishing coefficients: zero allocations on the audio thread.
* The renderer shares no lock with the compute side: `lockfree_engine_tests.rs` holds the engine
  mutex / keeps the compute thread busy while the audio thread renders (continuity + worst block
  time asserted); `no_alloc_tests.rs` covers the legacy path; `live_edit_tests.rs` the edits.

## Regression map

| audit issue | measurement |
|---|---|
| #72 per-sample ramps | `ramping_tests.rs` |
| #73 in-place patch-bay edits | `live_edit_tests.rs`, `patch_bay_ramp_tests.rs` |
| #74 directivity | `source_directivity_engine_tests.rs` |
| #75 lock-free audio thread | `lockfree_engine_tests.rs`, `spsc_tests.rs` |
| #76 resampler | `resampler_tests.rs` |
| #77 per-listener chain | `multi_listener_tests.rs` |
| #79 no allocation | `no_alloc_tests.rs`, the allocation test here |
| #80 output safety | `output_safety_tests.rs`, `limiter_tests.rs`, the ceiling test here |
| any block size | `block_size_tests.rs` |
| reverb calibration / balance | `direct_vs_room_levels.rs` |
| mix trims | `mix_trim_tests.rs` |

DSP-level golden files (`crates/quasar-dsp/tests/golden`, `dsp_perf.rs`) pin the unit outputs of the
binaural renderer, early taps, FDN, occlusion node, VBAP and the reflection decoder to 1e-5 of
peak; regenerate them deliberately with `QUASAR_GEN_GOLDEN=1 cargo test -p quasar-dsp --test dsp_perf`.

## Limits

These measurements say whether the engine does what its model says, not whether it sounds right.
Listening validation is a separate process: see `docs/listening-checklist.md`.
