<p align="center">
  <img width="300" height="300" alt="Gemini_Generated_Image_zaovfjzaovfjzaov" src="https://github.com/user-attachments/assets/9c1496bd-3859-4466-93b1-2f796c96f2e9" />
</p>

# Quasar — WIP Spatial Audio Engine

A **high-performance, ultra-modular spatial audio engine** designed for deep integration into custom game engine ecosystems. Quasar bridges offline acoustic baking with a flexible real-time spatial computation pipeline, enabling everything from static baked environments to fully dynamic real-time acoustics.

<img width="2660" height="1154" alt="image" src="https://github.com/user-attachments/assets/cb406346-4bb3-4048-8203-61cdec81b74f" />

---

## Core Concept

Quasar decouples three concerns through a **channel-pulling** architecture:

```
  Source (multi-channel audio)
     │
     │ ChannelPull (source_id, channel, gain_db)
     ▼
  SceneOutput (positioned world emitter)
     │
     │ SpatialQuery (source_position → listener_position)
     ▼
  Listener (world position + heading + physical device layout)
     │
     ▼
  VBAP decode onto physical speakers / HRTF
```

- **Sources** are raw multi-channel audio files or streams (mono, stereo, 7.1.4, etc.)
- **SceneOutputs** are positioned world-space emitters whose audible content is the sum of explicit `ChannelPull` taps onto loaded sources
- **Listeners** have a world position, heading, and a physical output layout (Stereo, 5.1, 7.1.4, Quad, Custom, HRTF)

Audio flows from sources through the **patch bay** into scene outputs, then through occlusion / early reflections / late reverb, and finally VBAP-decoded onto each listener's physical speaker layout — all **zero-allocation** on the audio thread.

---

## Crate Layout

```
Quasar/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── quasar/                   # Facade crate (re-exports everything + SpatialAudioEngine)
│   ├── quasar-core/              # Core traits, types, triple-buffer, probe grid, scene output model
│   ├── quasar-materials/         # Dynamic acoustic material system
│   ├── quasar-dsp/               # Zero-alloc audio DSP graph
│   └── quasar-backends/          # Acoustic compute backends (CPU SIMD / WGPU / HW stub)
```

---

## Quick Start

```rust
use quasar::prelude::*;
use quasar::SpatialAudioEngine;
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_materials::tabular::{Tabular8BandEvaluator, TABULAR_MODEL_ID};
use quasar_materials::instance::AcousticMaterialInstance;
use quasar_core::scene::AcousticScene;
use quasar_core::scene_output::*;
use quasar_core::bands::Band8;
use quasar_dsp::audio_buffer::AudioBuffer;

let sr = 48000.0;
let mut engine = SpatialAudioEngine::new(0, sr, 15.0);

// 1. Set up acoustic materials
engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator::new()));
let mat = engine.materials().add_instance(AcousticMaterialInstance::new(
    TABULAR_MODEL_ID,
    Tabular8BandEvaluator::create_params(Band8::splat(0.35), Band8::zeros(), Band8::zeros()),
));

// 2. Set up real-time acoustic compute backend
let mut scene = AcousticScene::new();
// add meshes with material handles...
let backend = CpuSimdComputeBackend::new(scene, CpuSimdConfig::default());
engine.set_backend(Box::new(backend));

// 3. Load an audio source
let src = engine.load_source(SourceConfig {
    path: "audio.wav".into(),
    channels: 2,
})?;

// 4. Create a positioned scene output (world-space speaker)
let speaker = engine.add_scene_output(SceneOutputConfig::new(
    [5.0, 1.5, 0.0],
    Movability::Static,
));

// 5. Pull a channel from the source into the speaker
engine.connect_pull(speaker, ChannelPull::new(src, 0, 0.0)); // L
engine.connect_pull(speaker, ChannelPull::new(src, 1, 0.0)); // R

// 6. Add a listener
let listener = engine.add_listener(ListenerConfig {
    position: [0.0, 1.6, 0.0],
    heading: [0.0, 0.0, -1.0],
    physical_layout: PhysicalOutputLayout::Stereo,
});

// Compute thread (15-30 Hz):
engine.update_scene_spatial();

// Audio thread (48 kHz):
let src_buf = AudioBuffer::new(2, 256);
let mut out = AudioBuffer::new(2, 256);
engine.process_audio_scene(&[&src_buf], &mut [&mut out]);
```

---

## Architecture

### Content Model (P1)

Three registries define the audio scene:

| Type | ID | Purpose |
|------|----|---------|
| `SourceConfig` | `SourceId` | Multi-channel audio content |
| `SceneOutputConfig` | `SceneOutputId` | Positioned world emitter with pull list |
| `ListenerConfig` | `ListenerId` | World position + heading + physical layout |

**Patches** (`ChannelPull`) define the content of each scene output — a `(source_id, channel, gain_db)` tuple. The sum of all pulls on an output is its audible content.

### Scene Pipeline (P2)

`process_audio_scene()` runs in one zero-alloc pass:

1. **Publish** latest triple-buffer coefficients from the compute thread
2. **Smooth** per-pair (listener × output) coefficients through equal-power crossfaders
3. **Patch bay** sums configured pulls into one mono buffer per scene output
4. **Spatial render** per output: occlusion/air absorption → early reflections → late reverb
5. **Listener decode**: VBAP-decode each output's mono onto the listener's physical speaker layout

### Threading Model

```
┌──────────────────────┐
│ GAME / MAIN THREAD   │
│ load_source()        │
│ add_scene_output()   │
│ add_listener()       │
│ connect_pull()       │
│ update_listener()    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ COMPUTE THREAD       │ (15–30 Hz)
│ update_scene_spatial()│
│ → hybrid probe / ray │
│ → triple-buffer write│
└──────────┬───────────┘
           │ lock-free atomic
           ▼
┌──────────────────────┐
│ AUDIO THREAD         │ (48 kHz)
│ process_audio_scene()│
│ → patch bay          │
│ → occlusion / early  │
│   / late reverb      │
│ → VBAP decode        │
│ NEVER allocates      │
└──────────────────────┘
```

### Lock-Free Parameter Exchange

`ParameterTripleBuffer` uses three atomic-indexed slots so the compute thread and audio thread never access the same slot simultaneously. No mutexes, no blocking, no allocation.

```
Compute: begin_write() → modify → end_write()  [atomic swap write↔staging]
Audio:   update() → read()                      [atomic swap staging↔read]
```

### Scene Output Config

```rust
pub struct SceneOutputConfig {
    pub position: [f32; 3],
    pub orientation: Option<[f32; 3]>,
    pub directivity: f32,           // 0 = omni, 1 = max cone
    pub pulls: Vec<ChannelPull>,    // patch-bay taps
    pub movability: Movability,
}
```

### Listener Config

```rust
pub struct ListenerConfig {
    pub position: [f32; 3],
    pub heading: [f32; 3],          // normalized forward vector
    pub physical_layout: PhysicalOutputLayout,
}

pub enum PhysicalOutputLayout {
    Stereo,
    Surround51,
    Surround714,
    Quad,
    Custom { positions: Vec<[f32; 3]> },
    Hrtf,
}
```

### Live Patch-Bay Remapping

Pulls can be added, removed, or re-gained at runtime — no audio glitch, no DSP rebuild:

```rust
engine.connect_pull(output_id, ChannelPull::new(src_id, 3, -6.0));
engine.disconnect_pull(output_id, src_id, 3);
engine.set_pull_gain(output_id, src_id, 3, -3.0);
```

---

## Crate Details

### `quasar-core` — Foundation Layer

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `bands` | `Band8` | 8-octave band model (62.5 Hz–8 kHz) |
| `rays` | `Ray`, `RayHit`, `RayInteractionContext` | Ray tracing primitives |
| `scene` | `AcousticScene`, `AcousticMesh` | Runtime proxy geometry |
| `backend` | `IAcousticComputeBackend`, `MaterialProvider`, `SpatialQuery`, `SpatialQueryResult` | Compute backend abstraction |
| `scene_output` | `SourceId`, `SourceConfig`, `SceneOutputId`, `SceneOutputConfig`, `ListenerId`, `ListenerConfig`, `ChannelPull`, `PhysicalOutputLayout` | Content model |
| `param_exchange` | `SpatialCoefficients`, `ParameterTripleBuffer` | Lock-free compute ↔ audio handoff |
| `probe_grid` | `AcousticProbeGrid`, `AcousticProbe`, `AcousticProbeSample` | Baked probe data |
| `hybrid` | `HybridProbeSampler`, `HybridSamplingStrategy` | Baked vs real-time strategy |
| `error` | `SpatialAudioError` | Unified error type |

### `quasar-materials` — Dynamic Material System

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `evaluator` | `IAcousticMaterialEvaluator`, `AcousticResponse8Band` | Material evaluation trait |
| `instance` | `MaterialModelId`, `MaterialParameterBuffer`, `AcousticMaterialInstance` | Material data |
| `registry` | `AcousticMaterialRegistry` | Thread-safe registry |
| `tabular` | `Tabular8BandEvaluator` | Static 8-band lookup |
| `delany_bazley` | `PorousDelanyBazleyEvaluator` | Continuous porous absorber |
| `resonant_panel` | `ResonantPanelEvaluator` | Low-frequency membrane absorber |
| `gpu_pipeline` | `GpuMaterialLayout` | GPU bindless pipeline helpers |

### `quasar-dsp` — Zero-Allocation Audio Graph

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `audio_buffer` | `AudioBuffer` | Fixed-capacity deinterleaved buffer |
| `crossfader` | `EqualPowerCrossfader` | Cosine/sine equal-power crossfade |
| `fractional_delay` | `HermiteInterpolatingDelayLine` | 4-point Hermite interpolation |
| `patch_bay` | `PatchBayNode`, `PatchEntry` | Multi-input/multi-output mixer |
| `occlusion` | `AirAbsorptionOcclusionNode` | 8-band occlusion + distance delay |
| `early_reflections` | `EarlyReflectionDelayNode` | Multi-tap fractional delay |
| `late_reverb` | `FdnReverbNode` | 16-line FDN with Hadamard matrix |
| `master_decoder` | `MasterSpatialDecoderNode` | VBAP speaker decoding |

### `quasar-backends` — Compute Backends

| Backend | Feature | Description |
|---------|---------|-------------|
| `CpuSimdComputeBackend` | `cpu-simd` (default) | BVH + rayon-parallel ray tracing |
| `WgpuComputeBackend` | `wgpu-compute` | WGSL compute shaders on GPU |
| `HardwareAcceleratorStub` | always | Future hardware placeholder |

---

## Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| `quasar-core` | `nebula-import` | Nebula serialization bridge |
| `quasar-backends` | `cpu-simd` | CPU SIMD backend (default) |
| `quasar-backends` | `wgpu-compute` | WGPU GPU compute backend |

---

## Integration with Nebula

[Nebula](https://github.com/Far-Beyond-Pulsar/Nebula) is the offline baking toolchain. Quasar consumes baked probe grids at runtime:

```rust
engine.set_probe_grid(nebula_bytes_to_probe_grid(&baked_bytes)?);
engine.set_strategy(HybridSamplingStrategy::HybridBlend);
```

Hybrid strategies:

| Strategy | Direct Path | Early Reflections | Late Reverb |
|----------|-------------|-------------------|-------------|
| `BakedOnly` | Probe grid | Probe grid | Probe grid |
| `RealTimeOnly` | Ray trace | Ray trace | Statistical |
| `HybridBlend` | Ray trace | Ray trace | Probe grid |

---

## Development

```bash
cargo build
cargo test --no-run
cargo test -p quasar-core
cargo test -p quasar-materials
cargo test -p quasar-dsp
cargo test -p quasar-backends --no-default-features --features cpu-simd
cargo check --workspace
```

### Crate Dependencies

```
quasar-audio
  ├── quasar-core        (glam, bytemuck, serde)
  ├── quasar-materials   (quasar-core)
  ├── quasar-dsp         (quasar-core, quasar-materials)
  └── quasar-backends    (quasar-core, quasar-materials, rayon, wgpu)
```

---

## License

MIT OR Apache-2.0
