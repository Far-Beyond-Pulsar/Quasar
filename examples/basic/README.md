# Cathedral audio demo (Helio large HLFS cathedral + Quasar)

Quasar's spatial audio engine in a complex scene: a duplicate of Helio's `indoor_cathedral_hlfs` example in its
**large** mode (a Gothic interior of about 145 x 45 x 43 m: ribbed vaults, clustered limestone piers, marble paving,
carved oak pews, bronze chandeliers, leaded stained glass, one shadowed sun and an incense medium), running on a
newer Helio (`05c2f7d7`, SceneDB-authoritative scene). The acoustic geometry is **read back from the SceneDB world**:
every renderable object is also a Quasar acoustic mesh, so the audio engine evaluates the real scene (about 415 000
triangles), not a hand-built proxy.

Run from `examples/basic` (or from the repository root):

```
cargo run --release --manifest-path examples/basic/Cargo.toml
```

The first build of the Helio dependency tree takes a while. The demo opens a window and plays audio on the default
output device (8-channel test WAV on 8 speakers, looping).

## Controls

Scene (Helio): **WASD** move, **Space / Shift** up / down, mouse drag look (click to grab the cursor), **Esc** release
the cursor / exit, **F1** shadow debug, **F2** performance overlay, **F3** debug overlay.

Audio (Quasar):

| Key | Action |
| --- | --- |
| **V** (or **R**) | start / pause acoustic trace capture (the last nonempty trace stays visible) |
| **C** / **B** / **N** / **M** | cycle emitter (nearest, then each, then nearest) / all emitters / rejected reflection candidates / the solver's probe rays |
| **T** | probe grid overlay (on by default; the lowest two layers are drawn) |
| **Y** | acoustic scene bounds box; prints the scene statistics and the acoustic material table |
| **G** | swap the Aux Left / Right channels (live patch-bay remap) |
| **[** / **]** | master volume down / up, 3 dB steps (pre-limiter gain; the limiter ceiling stays at -1 dBFS) |
| **-** / **=** | reverb trim down / up, 2 dB steps |
| **;** / **'** | early-reflection trim down / up, 2 dB steps |
| **1** / **2** / **3** | cycle DSP stage (0 silence .. 4 full) / print audio and spatial-worker timing / reset timing |

The overlay is drawn with Helio's world-space debug lines (they show through walls). Direct segment: green clear,
yellow partial, red blocked. Reflection paths: green / cyan / violet by order, brighter = stronger, the 16 strongest with
bounce markers and normals. The window title shows the rays traced, stored / dropped, valid / selected paths, the active
emitter(s) and, only for a watertight scene, whether the listener is inside or outside (this scene is not watertight, so the title says the side is unknown, see below).
Speakers are Helio billboards (a speaker icon that grows and brightens with the speaker's signal) plus debug spheres and
aim cones.

## Environment variables

Helio scene: `HLFS_NO_FOG`, `HLFS_FOG_DENSITY`, `HLFS_FOG_MODE`, `HLFS_FOG_BLEND`, `HLFS_SUN`, `HLFS_RT` (ray-traced
shadows), `HLFS_PRESAMPLED`, `HLFS_NO_HAZE_LIGHTS`, `HLFS_LEGACY_CATHEDRAL_LIGHTS`, `HLFS_GLASS_ALPHA`,
`HLFS_CLEAR_GLASS`, `HLFS_NO_STONE_TEXTURES`, `HLFS_LIVE_MOTION_TEST` (camera and listener follow Helio's scripted path).

Quasar: `RUST_LOG`, and the diagnostics below. `QUASAR_OPAQUE=1` makes every acoustic surface fully opaque (the old demo
shell's setting) to compare the inside balance with and without transmission.

## Headless check (no window, no audio device)

`QUASAR_HEADLESS_CHECK=1 cargo run --release` (or `-- --check`) builds the SceneDB world on a surface-less GPU device,
reads the acoustic geometry, builds the engine and prints: object / triangle counts (authored, SceneDB draw ranges,
audio scene), scene AABB, material usage and fallbacks, BVH build time, the 32 mirror planes chosen, the speaker
clearance, rays and milliseconds per spatial update at several listener positions, and a direct / early / late level
table. `QUASAR_SWEEP=1` prints the tracer cost for several image-source orders / plane budgets. The same checks are
tests: `cargo test --release -- --include-ignored` (the GPU ones are `#[ignore]` because they need an adapter).

## Geometry and materials

* **Geometry.** `src/acoustic_geometry.rs` walks the SceneDB `World`: each `StaticObjectComponent` row (mesh slot,
  material slot, column-major transform) becomes one `AcousticMesh`, with the CPU vertex / index payload of the row's
  `MeshComponent` and the row's material class. The cathedral batches everything into 9 opaque material meshes plus 6
  glass pane meshes (15 object rows, identity transforms); anything spawned into the world, with any transform, is
  included. The conversion is tested for transforms, tags, fallbacks and skipped rows on a CPU-only `World`.
* **Materials.** Every material row carries an `AcousticSurface` tag (set where the render material is made, in
  `cathedral_large.rs`). A material without a tag is classified from its Helio data (alpha blend -> glass, metallic ->
  bronze, emissive -> flame, else rough stone) and counted as a *fallback*; the check and tests require zero fallbacks.
* **Acoustic tables.** Per octave band 62.5 Hz .. 8 kHz, with `Tabular8BandEvaluator`, scattering 0 for now. The values are
  plausible engineering estimates from typical published tables (stone masonry, polished marble, oak, window glass, metal)
  and mass-law reasoning for the transmission, **not measurements**:

  | class | used for | absorption 62.5 .. 8k | transmission (amplitude) 62.5 .. 8k |
  | --- | --- | --- | --- |
  | stone | walls, clerestory, vault web | .03 .03 .04 .06 .07 .08 .10 .12 | .025 .016 .009 .005 .0028 .0016 .001 .001 |
  | carved stone | ribs, piers, arches, mullions | .04 .04 .05 .07 .08 .10 .12 .14 | as stone |
  | basalt | floor strips | .01 .01 .01 .015 .02 .02 .025 .03 | as stone |
  | marble paving | nave floor | .01 .01 .015 .02 .02 .025 .03 .04 | as stone |
  | altar stone | sanctuary, altar | .01 .01 .015 .02 .02 .025 .03 .04 | as stone |
  | oak | pews | .20 .15 .11 .09 .08 .08 .08 .09 | .30 .25 .18 .12 .08 .05 .03 .02 |
  | bronze | chandeliers, candle holders, saddle bars | .02 .02 .02 .03 .03 .04 .05 .05 | .98 .96 .93 .85 .70 .50 .35 .30 |
  | wax | candles | .02 .02 .03 .04 .05 .06 .07 .08 | as bronze |
  | flame | candle flames | 0 | 1 1 1 .99 .98 .97 .95 .95 |
  | stained glass | window panes (6 colour meshes) | .35 .35 .25 .18 .12 .07 .04 .04 | .17 .12 .07 .04 .025 .035 .04 .03 |

  Stone transmission is the proposal of GitHub issue #142 (heavy shell, about -32 dB at 62 Hz down to -60 dB). Thin
  rods and small bodies (radius 2-7 cm) are nearly transparent: the wavelength is far larger than the object.
* **Tracer settings** (`audio_demo::tracer_config`): image-source order **2** (not the old demo's 3), 32 mirror planes,
  128 diffuse rays, 150 m maximum reflection distance (the hall is 144 m long). Measured on the full scene with 8 emitters
  on a release build: order 3 with 32 planes costs about 100 ms and 265 000 - 310 000 rays per update, order 2 about
  10 - 14 ms and 50 000 - 66 000 rays. The backend's tracer chooses its mirror planes by merged coplanar area, so on this
  scene they are the floor, the side and end walls, the nave arcade walls, the aisle vaults and the window planes (see
  the headless output). The nave vault web is 64 narrow facets of about 43 m^2 each, below the 32nd plane (159 m^2), so
  there is **no ceiling mirror plane for the nave**: vault energy reaches the listener only through the statistical late
  field. Detail geometry (pews, chandeliers, tracery) is kept for occlusion and diffraction.
* **Is the cathedral watertight?** No: it is built from overlapping blocks, rods and single-sided panes, so
  `room_is_closed()` is false. The backend therefore uses the bounding-box volume for the statistical late field and
  warns once, and the closed-room shortcuts (skip reflections / diffraction when one endpoint is outside) never apply:
  listener and speakers outside the building are traced exhaustively through the real geometry (still attenuated by the
  transmission of the stone and glass), the backend's `room_side` is always `Unknown`, and the title says "inside/outside unknown".
* **Probe grid and T60.** Probes cover the scene AABB, about 12 m apart horizontally and 11 m vertically (5 x 5 x 14).
  Every probe carries the backend's own statistical (Eyring with air absorption) T60 for this geometry and these
  materials; the HybridBlend strategy is kept. Because the backend's room statistics count every triangle (including
  buried faces of overlapping blocks) over the bounding-box volume, the estimate is probably shorter than the building's
  true reverberation.
* **Spatial updates** run on a dedicated thread (`SpatialWorker`, ~30 Hz): the render thread only publishes the camera
  pose. The audio callback owns the lock-free `AudioRenderer` and shares only atomics.

## Stage layout and mix

The 8 stage speakers keep the old demo's layout relative to the audience point, translated to
`audio_demo::AUDIENCE = (0, 2.3, 56)` in the nave (the camera starts there, looking down the nave toward the altar at
-z). Heights are metres above the floor (as before). The audience point is 16 m from the entrance wall so the rear
speakers stay inside the building; the side speakers fall between two pew rows.

| device ch | speaker | WAV ch | position |
| --- | --- | --- | --- |
| 0 | Front Left | 0 | (-7, 5.5, 44) |
| 1 | Front Right | 1 | (7, 5.5, 44) |
| 2 | Center | 2 | (0, 3.0, 44) |
| 3 | Sub / LFE | 5 | (0, 0.3, 49) |
| 4 | Back Left | 3 | (-7, 2.0, 68) |
| 5 | Back Right | 4 | (7, 2.0, 68) |
| 6 | Side Left | 6 | (-7, 0.5, 44) |
| 7 | Side Right | 7 | (7, 0.5, 44) |

Every speaker is aimed at the audience point with directivity 0.7, the reverb bus is trimmed by -5 dB and the Sub output is
also sent to the LFE channel. Those are artistic defaults (the engine's own defaults stay physical); the direct-sound
distances of the old demo are preserved, but the late field of this hall is much weaker relative to the direct sound than
in the old 22 x 56 m hall (see the headless level table), so `=` raises it if you want more hall.
