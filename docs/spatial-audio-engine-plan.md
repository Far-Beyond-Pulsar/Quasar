# Quasar Spatial Audio Engine — Locked Implementation Plan

Status: **LOCKED.** Sub-agent implementation proceeds strictly per this document.
Any deviation requires returning to this file and amending it first.

---

## 1. Vision (non-negotiable)

Quasar is a **room-driven spatial audio engine**. Every effect applied to audio is
*derived from the scene* — geometry, materials, probes, and ray-tracing (bouncing
audio around the world). No effect parameter is authored per-emitter; the room is
the single source of truth and the DSP graph is a renderer of resolved
coefficients.

**Content model — channel pulling is the ONLY way audio enters the mix.**
There is no "play a whole file on an emitter" convenience path. A `Source` is raw
multi-channel content. A `SceneOutput` is a positioned world emitter whose audible
content is the sum of explicit `(source_id, channel, gain)` taps. Any number of
taps, from any number of sources, per output.

**Three strictly separated layers:**

| Layer | Meaning | Owned by |
|---|---|---|
| Source | An audio file/stream with N channels | Game loads it |
| SceneOutput | A position (emitter/speaker) in the game world | Game places it |
| Listener | A world position + heading + physical device layout | Game places it; Quasar renders to it |

Scene outputs and listeners' physical layouts are **independent**. The engine
spatializes every scene output against every listener and renders the result onto
that listener's physical device.

---

## 2. Core types (exact signatures)

Place in `crates/quasar-core/src/` (new modules) and re-export.

```rust
// scene_output.rs
pub struct SourceId(pub u32);
pub struct SceneOutputId(pub u32);
pub struct ListenerId(pub u32);

/// ONE tap in the patch bay. Content for a scene output = Σ over its pulls.
pub struct ChannelPull {
    pub source_id: SourceId,
    pub channel: u32,          // 0 .. source.channels-1
    pub gain_db: f32,          // API surface is dB; converted to linear in DSP
}

pub struct SceneOutputConfig {
    pub position: [f32; 3],
    pub orientation: Option<[f32; 3]>,   // heading for directivity (reserved)
    pub directivity: f32,                // 0.0 = omnidirectional, 1.0 = max cone
    pub pulls: Vec<ChannelPull>,
    pub movability: crate::scene::Movability,
}

pub enum PhysicalOutputLayout {
    Stereo,
    Surround51,
    Surround714,
    Quad,
    Custom { positions: Vec<[f32; 3]> }, // real-world speaker directions
    Hrtf,
}

pub struct ListenerConfig {
    pub position: [f32; 3],
    pub heading: [f32; 3],               // normalized forward vector
    pub physical_layout: PhysicalOutputLayout,
}
```

`Movability` must exist in `crates/quasar-core/src/scene.rs` (already does, verify).

---

## 3. Data flow per audio block (zero-alloc, audio thread)

```
 ① RESOLVE   (compute thread, per (SceneOutput, Listener))
    HybridSampler.resolve → SpatialCoefficients
    (direct gain, occlusion, early reflections, late T60/energy)

 ② PATCH BAY (audio thread)
    for each SceneOutput:
        buf[output] = Σ over pulls: src[source_id].channel[channel] × lin_gain

 ③ SPATIAL RENDER (audio thread)
    for each SceneOutput (computed ONCE per output, NOT per listener):
        occ  — occlusion filter (per-band lowpass from traced absorption)
        erf  — early reflection taps (delayed/gained bounces, per band)
        rev  — send level derived from room late reverb estimate
    for each Listener:
        decode each SceneOutput's rendered buffer onto listener.physical_layout
        (VBAP nearest-speaker pan from azimuth/elevation, or HRTF)
        sum → listener output bus

 ④ DEVICE
    format-convert (matrix) each listener bus → physical device
    (arbitrary source format → arbitrary device format; engine-owned)
```

---

## 4. Runtime API (on `SpatialAudioEngine`)

```rust
// Sources
pub fn load_source(&mut self, cfg: SourceConfig) -> Result<SourceId, SpatialAudioError>;
pub fn unload_source(&mut self, id: SourceId);

// Scene outputs
pub fn add_scene_output(&mut self, cfg: SceneOutputConfig) -> SceneOutputId;
pub fn remove_scene_output(&mut self, id: SceneOutputId);
pub fn set_scene_output_position(&mut self, id: SceneOutputId, pos: [f32; 3]);

// Patch bay (all gain-smoothed, click-free)
pub fn connect_pull(&mut self, output: SceneOutputId, pull: ChannelPull);
pub fn disconnect_pull(&mut self, output: SceneOutputId, source: SourceId, channel: u32);
pub fn set_pull_gain(&mut self, output: SceneOutputId, source: SourceId, channel: u32, gain_db: f32);

// Listeners
pub fn add_listener(&mut self, cfg: ListenerConfig) -> ListenerId;
pub fn remove_listener(&mut self, id: ListenerId);
pub fn update_listener(&mut self, id: ListenerId, position: [f32; 3], heading: [f32; 3]);

// Threads
pub fn update_spatial(&mut self, listener_id: ListenerId); // compute thread
pub fn process_audio(&mut self, sources: &[&AudioBuffer], outputs: &mut [AudioBuffer]); // audio thread
```

`process_audio` signature changes: inputs are one `AudioBuffer` per **Source**
(with `source.channels` channels); outputs are one buffer per **Listener**.
A source buffer carries its channel count; the patch bay indexes into it.

---

## 5. DSP pipeline changes (crates/quasar-dsp)

1. **Patch-bay mixer node** — `PatchBayNode`: input = all source buffers, output =
   per-scene-output mixed buffers. Sums `channel × linear_gain` per pull. Zero-alloc.
2. **Early-reflection render stage** — new node (or extend existing graph) that
   renders `SpatialCoefficients.early_reflections` as delayed/gained taps. This data
   is computed today (`CpuSimdComputeBackend::trace_early_reflections`) but NEVER
   rendered. Must be wired into the graph.
3. **Working occlusion filter** — `AirAbsorptionOcclusionNode::update_occlusion`
   currently exists but is NEVER called (`occlusion.rs:128`). Wire it so per-band
   lowpass cutoffs are driven by the traced occlusion factor + material absorption.
4. **Crossfader units bug** — `fade_frames` counts samples (`crossfader.rs:24`) but
   `advance()` is called once per block (`quasar/src/lib.rs:191`). A 15 ms fade
   takes ~3.8 s and restarts every compute update. Fix: advance must be scaled by
   block size (or fade_frames expressed in blocks) so fades complete in ~15 ms and
   converge. This is the root cause of source-0-asymmetry and frozen panning.
5. **Emitter-once / decode-per-listener** — per-SceneOutput occlude/reverb once;
   only VBAP/HRTF decode is per-listener. No N× duplicate DSP.
6. **Format conversion matrix** — final stage converts listener bus format → device
   format. Arbitrary custom → stereo/5.1/quad/HRTF. Engine-owned, matrix-based.

---

## 6. Mandatory bug fixes (prereqs for the demo to sound correct)

| Bug | File | Fix |
|---|---|---|
| Crossfade never converges / source-0 asymmetry | `quasar-dsp/src/crossfader.rs`, `quasar/src/lib.rs` | Scale advance by block size; express fade in blocks; ensure per-output params (not per-WAV-channel) |
| Early reflections computed but not rendered | `crates/quasar-dsp`, graph wiring | Add early-reflection tap render stage |
| `update_occlusion` dead code | `crates/quasar-dsp/src/occlusion.rs` | Wire per-band filter updates from coefficients |
| Reverb dry/wet split hardcoded | `crates/quasar-dsp/src/late_reverb.rs` | Split derived from room resolution (`early_late_split_secs`), not constant 0.5 |
| Output channels silently dropped | `examples/basic/src/main.rs` | Engine downmix via format matrix; no silent drop |
| Backend strategy | demo | Use `HybridBlend` (probes + real-time), not `RealTimeOnly` |

---

## 7. AAA design decisions (locked)

1. **Buses + sends.** Reverb is a shared bus with per-output sends (room property,
   not emitter property). No per-output FDN as the shipped architecture.
2. **Pluggable acoustic model.** `HybridProbeSampler` (real-time ray + baked probe)
   is the resolution interface. Baked probes used where available. SceneOutput has
   directivity/spread.
3. **Emitter-once, decode-per-listener.** Voice budget + relevance/priority in the
   data model from day one (implementation may be phased).
4. **Format conversion matrix + object path.** Listener output and content have
   explicit channel formats; final converter is a matrix. Custom-N-ch path kept;
   Dolby-Atmos-style object path reserved.
5. **dB API, linear DSP, ramps, pan laws.** All gain APIs are dB. Every connect/gain
   change ramps (attack/release). Pan law selectable per format.

---

## 8. Implementation phases (deliverables + acceptance)

### P1 — Core types + API surface
- [ ] Add `SourceId`, `SceneOutputId`, `ListenerId`, `ChannelPull`, `SceneOutputConfig`,
      `PhysicalOutputLayout`, `ListenerConfig` in `crates/quasar-core`.
- [ ] Re-export from `crates/quasar` (lib.rs `pub use`).
- [ ] Add engine methods (Section 4) to `SpatialAudioEngine`.
- [ ] Fix crossfader units bug (Section 6).
- [ ] `cargo check` + `cargo test` pass. No `todo!()`, no unwrap in audio path.

### P2 — Patch-bay mixer + DSP render
- [ ] `PatchBayNode` + graph wiring (source buffers → per-output mixed buffers).
- [ ] Early-reflection render stage wired.
- [ ] Occlusion filter wired (`update_occlusion` called with real coefficients).
- [ ] Per-SceneOutput occ/rev; decode-per-listener.
- [ ] `cargo check` + `cargo test` pass. Zero allocations in `process_audio`.

### P3 — Multi-listener, dynamic remap, format matrix
- [ ] N listeners; decode-per-listener stage.
- [ ] Click-free `connect_pull`/`disconnect_pull`/`set_pull_gain` (ramps).
- [ ] Format conversion matrix (custom → stereo/5.1/quad/HRTF).
- [ ] Virtualization/voice-budget data model (priority on SceneOutput).
- [ ] `cargo check` + `cargo test` pass.

### P4 — Demo rewrite (`examples/basic`)
- [ ] Load `8_Channel_ID.wav` as ONE source (8 channels).
- [ ] 8 scene outputs at cathedral positions; output *i* pulls `(source, ch i, 0 dB)`.
- [ ] Listener = camera (position + heading); physical layout = real device.
- [ ] Delete the `source_positions` array + the fused `positions` array pattern.
- [ ] Backend strategy = `HybridBlend` (probes + real-time).
- [ ] Acceptance: moving a scene output behind a column changes material-filtered
      occlusion + reflection; entering a baked zone swaps reverb to probe data;
      remapping a speaker is a single runtime `connect_pull` call.

---

## 9. Anti-shortcut rules (ALL agents MUST comply)

1. **No hardcoded effect parameters.** Every gain/decay/filter value in DSP comes
   from resolved `SpatialCoefficients`. No magic constants for reverb/occlusion
   levels.
2. **No `todo!()`, `unimplemented!()`, or silent `unwrap()` on the audio thread.**
   Panic paths are only on the compute/init threads.
3. **Zero allocation in `process_audio`.** All buffers preallocated at construction.
   The patch bay and all nodes allocate at `new()`, never in `process()`.
4. **No shortcuts in the patch bay.** It must support any number of pulls per output,
   any combination of sources/channels, arbitrary gain. No "one channel per output"
   convenience fallback.
5. **No removal of room-driven computation.** Distance attenuation, occlusion,
   reflections, reverb must all still flow from the acoustic scene/materials/probes.
6. **Preserve existing public API where possible.** `SpatialCoefficients`,
   `SpatialQuery`, backends, and the DSP graph remain. Do not delete modules that
   other crates depend on without a migration.
7. **Verification is mandatory.** Every phase ends with `cargo check` and `cargo test`
   green for the workspace AND the example crate. Report exact commands run.
8. **Follow existing code conventions.** Match surrounding style; no new dependencies
   unless approved in this file.

---

## 10. Verification commands

```
# workspace
cargo check -p quasar-core -p quasar-dsp -p quasar-backends -p quasar-audio
cargo test -p quasar-core -p quasar-backends   # dsp_tests compile errors are pre-existing; do not expand scope

# example (separate workspace)
cargo check   (in examples/basic)

# any new types/nodes need unit tests in the matching crate's tests/ dir
```
