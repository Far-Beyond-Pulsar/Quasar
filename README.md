# Quasar — Spatial Audio Engine

A **high-performance, ultra-modular spatial audio engine** designed for deep integration into custom game engine ecosystems. Quasar bridges offline acoustic baking with a flexible real-time spatial computation pipeline, enabling everything from static baked environments to fully dynamic real-time acoustics.

---

## Architecture

```
                              ┌──────────────────────────────────────────────┐
                              │              Audio Sources                   │
                              │  (Mono, Stereo, Ambisonic, 7.1.4, Custom)    │
                              └──────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                Spatial Audio Engine Core                                 │
│                                                                                          │
│  ┌─────────────────────────┐   ┌─────────────────────────┐   ┌────────────────────────┐  │
│  │    Acoustic Scene DB    │   │  Nebula Probe Grid      │   │  Lock-Free Audio Graph │  │
│  │  (BVH / Proxy Geometry) │   │  (8-Band RIR / T60 /    │   │  (DSP Nodes / Buses)   │  │
│  │                         │   │   Reverb Zones)         │   │                        │  │
│  └───────────┬─────────────┘   └───────────┬─────────────┘   └───────────┬────────────┘  │
└──────────────┼─────────────────────────────┼─────────────────────────────┼───────────────┘
               │                             │                             │
               ▼                             ▼                             │
┌──────────────────────────────────────────────────────────────┐           │
│                 Spatial Computing Backend                    │           │
│  (CPU SIMD / WGPU Ray Query / HW Accelerator Stub)           │           │
│  → Direct path, early reflections, late reverb estimate      │           │
└──────────────────────────────┬───────────────────────────────┘           │
                               │                                           │
                               ▼                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                           Real-Time Audio Thread (48 kHz)                                 │
│                                                                                           │
│  ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │  Lock-Free Coeff Ramp   │──►│  FDN Reverb / Part.     │──►│  HRTF / VBAP Decoder    │  │
│  │  (Zero Alloc / 10-20ms) │   │  FFT Convolution        │   │  (Speakers / Headphone) │  │
│  └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Crate Layout

```
Quasar/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── quasar/                   # Runtime binary
│   ├── quasar-core/              # Core traits, types, triple-buffer, probe grid
│   ├── quasar-materials/         # Dynamic acoustic material system
│   ├── quasar-dsp/               # Zero-alloc audio DSP graph
│   ├── quasar-backends/          # Acoustic compute backends
│   └── quasar-audio/             # Facade crate + SpatialAudioEngine
```

### `quasar-core` — Foundation Layer

Core traits, data types, and the lock-free parameter exchange system that connects the compute thread to the audio thread.

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `bands` | `Band8` | 8-octave band model (62.5 Hz - 8 kHz), per-band arithmetic |
| `rays` | `Ray`, `RayHit`, `RayInteractionContext` | Ray tracing primitives |
| `scene` | `AcousticScene`, `AcousticMesh` | Runtime proxy geometry |
| `backend` | `IAcousticComputeBackend`, `MaterialProvider`, `SpatialQuery`, `SpatialQueryResult` | Compute backend abstraction |
| `param_exchange` | `SpatialCoefficients`, `ParameterTripleBuffer` | Lock-free compute → audio handoff |
| `probe_grid` | `AcousticProbeGrid`, `AcousticProbe`, `AcousticProbeSample` | Baked probe data with trilinear interpolation |
| `hybrid` | `HybridProbeSampler`, `HybridSamplingStrategy` | Baked vs real-time resolution strategy |
| `error` | `SpatialAudioError` | Unified error type |
| `nebula_import` | `nebula_bytes_to_probe_grid()` | *(feature-gated)* Nebula serialization bridge |

**Key trait — `IAcousticComputeBackend`:**

```rust
pub trait IAcousticComputeBackend: Send + Sync {
    fn query_spatial(
        &self,
        queries: &[SpatialQuery],
        materials: &dyn MaterialProvider,
    ) -> Vec<SpatialQueryResult>;

    fn supports_dynamic_geometry(&self) -> bool;
    fn update_scene(&mut self, scene: &AcousticScene) -> Result<(), SpatialAudioError>;
    fn trace_ray(&self, ray: &Ray) -> Vec<RayHit>;
}
```

**Lock-Free Parameter Exchange — `ParameterTripleBuffer`:**

The triple buffer uses three atomic-indexed slots such that the compute thread (producer) and audio thread (consumer) never access the same slot simultaneously. No mutexes, no blocking, no allocation.

```
Compute Thread:  begin_write() → modify → end_write()  [atomic swap write↔staging]
Audio Thread:    update() → read()                      [atomic swap staging↔read]
```

### `quasar-materials` — Dynamic Material System

Materials are NOT hardcoded structs. They are dynamic physical transfer functions composed from a `MaterialModelId` and a raw byte-aligned `MaterialParameterBuffer` that can be blitted directly to GPU storage buffers.

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `evaluator` | `IAcousticMaterialEvaluator`, `AcousticResponse8Band` | Material evaluation trait |
| `instance` | `MaterialModelId`, `MaterialParameterBuffer`, `AcousticMaterialInstance` | Material data representation |
| `registry` | `AcousticMaterialRegistry` | Thread-safe material registry |
| `tabular` | `Tabular8BandEvaluator` | Static 8-band lookup (model 1) |
| `delany_bazley` | `PorousDelanyBazleyEvaluator` | Continuous porous absorber (model 2) |
| `resonant_panel` | `ResonantPanelEvaluator` | Low-frequency membrane absorber (model 3) |
| `gpu_pipeline` | `GpuMaterialLayout`, `wgsl_material_eval_source()` | GPU bindless pipeline helpers |

**Built-in Material Evaluators:**

| Model ID | Name | Parameters | Formula |
|----------|------|------------|---------|
| 1 | Tabular | 24 × f32 (absorption, scattering, transmission × 8 bands) | Direct lookup |
| 2 | Delany-Bazley | flow_resistivity, thickness_m | Complex impedance + propagation constant |
| 3 | Resonant Panel | panel_mass_kgm2, cavity_depth_m | Lorentzian peak at resonant frequency |

Hot-swappable: updating a material's `MaterialParameterBuffer` takes effect immediately — no acceleration structure rebuild required.

### `quasar-dsp` — Zero-Allocation Audio Graph

The audio thread (48 kHz, 128-256 sample blocks) **NEVER allocates, locks, or blocks**. All memory is pre-allocated at initialization.

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `audio_buffer` | `AudioBuffer` | Fixed-capacity deinterleaved buffer (stack-allocated) |
| `crossfader` | `EqualPowerCrossfader` | Cosine/sine equal-power crossfade (10-20 ms) |
| `fractional_delay` | `HermiteInterpolatingDelayLine` | 4-point Hermite interpolation for fractional sample delays |
| `node_graph` | `AudioNode` trait, `AudioNodeGraph` | Topological DSP graph execution |
| `directivity` | `DirectivityDspNode` | Source radiation pattern (omni, cardioid, figure-8, SH) |
| `occlusion` | `AirAbsorptionOcclusionNode`, `BiquadFilter` | 8-band occlusion filtering + distance delay |
| `early_reflections` | `EarlyReflectionDelayNode` | Multi-tap fractional delay for specular paths |
| `late_reverb` | `FdnReverbNode` | 16-line Feedback Delay Network with Hadamard matrix |
| `master_decoder` | `MasterSpatialDecoderNode` | Binaural HRTF / VBAP / Ambisonic output |

**Crossfade curve:** `g₀(t) = cos(πt/2)`, `g₁(t) = sin(πt/2)` for `t ∈ [0,1]` → constant power (`g₀² + g₁² = 1`).

**FDN Reverb Architecture:**
- 16 mutually-coupled delay lines with prime-based coprime lengths (15-80 ms range)
- 16×16 Hadamard (Householder) orthogonal feedback matrix
- Per-line one-pole lowpass filters for frequency-dependent T60 control
- Modulated delay offsets for chorusing (smooths metallic artifacts)

### `quasar-backends` — Compute Backends

Three interchangeable backends implementing `IAcousticComputeBackend`:

| Backend | Feature Flag | Description |
|---------|-------------|-------------|
| `CpuSimdComputeBackend` | `cpu-simd` (default) | BVH-accelerated, `rayon`-parallel ray tracing |
| `WgpuComputeBackend` | `wgpu-compute` | WGSL compute shader dispatch on GPU |
| `HardwareAcceleratorStub` | always | Placeholder for future DSP/NPU hardware |

**CpuSimdComputeBackend features:**
- SAH (Surface Area Heuristic) BVH acceleration structure
- Möller-Trumbore ray-triangle intersection
- Specular path tracing configurable up to order 5
- Sabine/Eyring statistical late reverberation estimation
- ISO 9613-1 air absorption model

**WgpuComputeBackend features:**
- Double-buffered staging buffers for non-blocking readback
- WGSL switch-based material evaluation dispatch
- Configurable rays-per-query (default: 256)

### `quasar-audio` — Facade Crate

```rust
pub struct SpatialAudioEngine {
    // Owns all subsystems
    hybrid_sampler: HybridProbeSampler,
    triple_buffer: ParameterTripleBuffer,
    material_registry: AcousticMaterialRegistry,
    dsp_graph: AudioNodeGraph,
    crossfader: EqualPowerCrossfader,
}

impl SpatialAudioEngine {
    pub fn new(sample_rate: f32, fade_ms: f32) -> Self;
    pub fn set_backend(&mut self, backend: Box<dyn IAcousticComputeBackend>);
    pub fn set_probe_grid(&mut self, grid: AcousticProbeGrid);
    pub fn set_strategy(&mut self, strategy: HybridSamplingStrategy);
    pub fn materials(&self) -> &AcousticMaterialRegistry;
    pub fn materials_mut(&mut self) -> &mut AcousticMaterialRegistry;
    pub fn dsp_graph(&mut self) -> &mut AudioNodeGraph;
    pub fn update_spatial(&self, queries: &[SpatialQuery]);
    pub fn process_audio(&mut self, inputs: &[&AudioBuffer], output: &mut AudioBuffer);
    pub fn reset(&mut self);
}
```

---

## Threading Model

```
┌──────────────────────────────────────────────────────────────┐
│ GAME / MAIN THREAD                                           │
│ - Updates source/listener positions, scene geometry          │
│ - Calls SpatialAudioEngine::update_spatial() per frame       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ COMPUTE THREAD (15-30 Hz)                                    │
│ - HybridProbeSampler::resolve() per source                   │
│   → baked probe grid OR real-time IAcousticComputeBackend    │
│ - Writes SpatialCoefficients → ParameterTripleBuffer         │
│   (begin_write / end_write — atomic, lock-free)              │
└──────────────────────┬───────────────────────────────────────┘
                       │ atomic triple buffer
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ AUDIO THREAD (48 kHz, 128-256 sample blocks)                 │
│ NEVER allocates, NEVER locks, NEVER blocks                   │
│ 1. ParameterTripleBuffer::update() + read()                  │
│ 2. EqualPowerCrossfader::set_target() → advances blend       │
│ 3. AudioNodeGraph::process():                                │
│    Directivity → Occlusion → EarlyReflections →              │
│    LateReverb → MasterDecoder                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Integration with Nebula

[Nebula](https://github.com/Far-Beyond-Pulsar/Nebula) is the offline baking toolchain that produces spatial audio data. Quasar consumes this data at runtime — no compile-time dependency required.

### Bridge (feature: `nebula-import`)

```rust
use quasar_core::nebula_import::nebula_bytes_to_probe_grid;

// Load baked data from Nebula's serialized AcousticOutput
let baked_bytes = std::fs::read("scene.acoustic")?;
let probe_grid = nebula_bytes_to_probe_grid(&baked_bytes)?;

// Feed into the engine
let mut engine = SpatialAudioEngine::new(48000.0, 15.0);
engine.set_probe_grid(probe_grid);
engine.set_strategy(HybridSamplingStrategy::HybridBlend);
```

The bridge:
1. Deserializes nebula-audio's bincode format using mirrored serde structs
2. Converts each `ImpulseResponse` → `AcousticProbe` (transposing `[Vec<f32>; 8]` → `Vec<Band8>`)
3. Constructs a runtime `AcousticProbeGrid` with the probe data

### Hybrid Sampling Strategies

| Strategy | Direct Path | Early Reflections | Late Reverb |
|----------|------------|-------------------|-------------|
| `BakedOnly` | Probe grid | Probe grid | Probe grid |
| `RealTimeOnly` | Ray trace | Ray trace | Statistical estimate |
| `HybridBlend` | Ray trace | Ray trace | Probe grid |

---

## Usage Example

```rust
use quasar_audio::*;
use quasar_core::probe_grid::AcousticProbeGrid;

fn main() -> Result<(), SpatialAudioError> {
    // Create the engine
    let mut engine = SpatialAudioEngine::new(48000.0, 15.0);

    // Register materials
    let concrete = AcousticMaterialInstance::new(
        TABULAR_MODEL_ID,
        Tabular8BandEvaluator::create_params(
            Band8::new([0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05]),  // absorption
            Band8::splat(0.1),  // scattering
            Band8::zeros(),     // transmission
        ),
    );
    engine.materials().register_evaluator(Box::new(Tabular8BandEvaluator));
    engine.materials().add_instance(concrete);

    // Set up real-time CPU backend
    let scene = AcousticScene::new();
    let backend = CpuSimdComputeBackend::new(scene, CpuSimdConfig::default());
    engine.set_backend(Box::new(backend));

    // Spatial query per frame (game thread)
    let queries = vec![SpatialQuery {
        source_position: [0.0, 1.0, 0.0],
        listener_position: [5.0, 1.5, 3.0],
        source_id: 0,
    }];
    engine.update_spatial(&queries);

    // Audio processing (audio thread)
    let input = AudioBuffer::new(1, 256);  // mono source
    let mut output = AudioBuffer::new(2, 256);  // stereo output
    engine.process_audio(&[&input], &mut output);

    Ok(())
}
```

---

## Configuration Reference

### `CpuSimdConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_reflection_order` | `u32` | `3` | Max specular bounces for early reflections |
| `diffuse_rays_per_query` | `u32` | `64` | Stochastic rays for reverb estimation |
| `max_reflection_distance` | `f32` | `50.0` | Max reflection path length (world units) |
| `speed_of_sound` | `f32` | `343.0` | Speed of sound (m/s) |
| `temperature_celsius` | `f32` | `20.0` | Ambient temperature for air absorption |
| `humidity_percent` | `f32` | `50.0` | Relative humidity for air absorption |

### `WgpuComputeConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rays_per_query` | `u32` | `256` | Stochastic rays per source-listener pair |
| `max_bounces` | `u32` | `16` | Max ray bounces |
| `max_duration_secs` | `f32` | `1.0` | Max ray travel time |
| `speed_of_sound` | `f32` | `343.0` | Speed of sound (m/s) |

### `Band8` — 8-Octave Band Centers

| Band | Center Frequency |
|------|-----------------|
| 0 | 62.5 Hz |
| 1 | 125 Hz |
| 2 | 250 Hz |
| 3 | 500 Hz |
| 4 | 1 kHz |
| 5 | 2 kHz |
| 6 | 4 kHz |
| 7 | 8 kHz |

---

## Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| `quasar-core` | `nebula-import` | Enable Nebula serialization bridge (adds serde + bincode deps) |
| `quasar-backends` | `cpu-simd` | CPU SIMD backend with rayon (default) |
| `quasar-backends` | `wgpu-compute` | WGPU GPU compute backend |

---

## Material Model Reference

### Tabular (Model ID: 1)
Direct 8-band lookup table. Parameter buffer: 24 × f32 = 96 bytes.
```
[absorption_0..7, scattering_0..7, transmission_0..7]
```

### Delany-Bazley Porous Absorber (Model ID: 2)
Empirical model for fibrous porous materials. Parameter buffer: 8 bytes.
```
[flow_resistivity: f32]  — Rayls/m (1000–100000, typical: 5000–50000)
[thickness_m: f32]       — material thickness in meters
```

Uses the Delany-Bazley model:
- Characteristic impedance: Zc = ρ₀c₀[1 + 0.0571(ρ₀f/Rs)^(-0.754) - j·0.087(ρ₀f/Rs)^(-0.732)]
- Propagation constant: k = (ω/c₀)[1 + 0.0978(ρ₀f/Rs)^(-0.700) - j·0.189(ρ₀f/Rs)^(-0.595)]
- Surface impedance: Zs = -j·Zc·cot(k·d)
- Absorption: α = 1 - |(Zs - ρ₀c₀)/(Zs + ρ₀c₀)|²

### Resonant Panel Absorber (Model ID: 3)
Mass-spring membrane absorber for low-frequency control. Parameter buffer: 8 bytes.
```
[panel_mass_kgm2: f32]   — surface density in kg/m² (1–20)
[cavity_depth_m: f32]    — air gap in meters (0.02–0.5)
```

Absorption peaks at the resonant frequency: f₀ ≈ 60 / √(m·d)

---

## Test Coverage

| Crate | Test File | Test Count |
|-------|-----------|------------|
| `quasar-core` | `tests/core_tests.rs` | 12 |
| `quasar-core` | `tests/nebula_import_tests.rs` | 7 |
| `quasar-materials` | `tests/materials_tests.rs` | 13 |
| `quasar-dsp` | `tests/dsp_tests.rs` | 21 |
| `quasar-backends` | `tests/backends_tests.rs` | 16 |
| **Total** | | **69** |

Key test categories:
- **Band8 math**: lerp accuracy, per-band operations, dB conversion
- **Triple buffer**: single/multi-cycle write/read, thread safety
- **Probe grid**: trilinear interpolation, bounds checking
- **Material evaluators**: absorption range, frequency-dependent behavior, resonant peaks
- **Material registry**: register, evaluate, hot-swap, remove, error handling
- **DSP nodes**: crossfader constant power, fractional delay accuracy (< -80 dB), FDN stability
- **Backends**: BVH build, ray intersection, direct path occlusion, distance attenuation, air absorption
- **Serialization**: round-trip with nebula-compatible bincode format

---

## Development

```bash
# Build all crates
cargo build

# Run all tests (compile only — some require GPU)
cargo test --no-run

# Run CPU-compatible tests
cargo test -p quasar-core
cargo test -p quasar-materials
cargo test -p quasar-dsp
cargo test -p quasar-backends --no-default-features --features cpu-simd

# Check with Nebula import feature
cargo check -p quasar-core --features nebula-import

# Full workspace check
cargo check --workspace
```

### Crate Dependencies

```
quasar-audio
  ├── quasar-core        (glam, bytemuck, optionally serde + bincode)
  ├── quasar-materials   (quasar-core)
  ├── quasar-dsp         (quasar-core, quasar-materials)
  └── quasar-backends    (quasar-core, quasar-materials, optionally rayon + wgpu)
```

---

## License

This project is licensed under the terms of the [MIT](LICENSE) license.
