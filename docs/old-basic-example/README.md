# Cathedral audio demo

Run `cargo run --release --manifest-path examples/basic/Cargo.toml` from the repository root.

Press **V** to start, pause, or resume acoustic trace capture (**R** is an alias). Pausing freezes the last nonempty trace on screen. Empty updates also keep the previous snapshot, so the lines remain visible when the listener and scene stop changing. Key repeat will not toggle capture repeatedly.

By default the overlay shows one emitter at a time (the one nearest to the listener):

- **Direct segment** (speaker to listener): **green** clear, **yellow** partially occluded, **red** blocked (mean occlusion amplitude at or below 0.3).
- **Reflection paths** of that emitter, from speaker through each bounce to listener: **green** first order, **cyan** second order, **violet** third order. Brightness follows the path gain (down to 40 dB below the strongest). The 16 strongest are emphasised with **yellow** bounce markers and surface normals; weaker valid paths are drawn faint.

Keys (all ignore key repeat):

- **C** cycles the emitter: nearest (auto), then each emitter in id order, then back to nearest.
- **B** shows all emitters at once.
- **N** shows rejected reflection candidates (capped): **orange** blocked, **grey** bounce outside the surface (partial, listener side only), **yellow** edge fade near zero, **blue** below the energy cutoff.
- **M** shows the solver's probe rays: occlusion probes (red hit / blue clear), diffraction detour probes (orange) and path-validation segments (lavender). Stored only while on and capped at 4096 per update.

The window title shows the total rays traced (what the solver really fired), how many were stored for drawing and how many were dropped by the cap, valid and selected reflection paths, the active emitter(s), whether the listener is inside or outside the room ("outside: reflections skipped" when the listener or the emitter is outside the closed room shell with opaque walls, where no reflected path can exist), and the N / M states.

Lines use Helio's world-space debug drawing and show through walls. Capture can be expensive in dense scenes.

The drawing refreshes every display frame. Captured geometry refreshes with the engine's existing roughly 30 Hz spatial update, across all speaker/listener pairs, so it does not restart audio crossfades on every display frame. Pausing capture freezes the displayed geometry while audio spatial processing continues as before.

The demo's acoustic scene contains the room shell and columns, rather than every decorative visual mesh. Early reflections use image-source geometry and BVH visibility queries. Late reverb is statistical and has no traced reflection paths.

## Audio mix

The demo mix is artistic; the engine's own defaults are physical. Every stage speaker is aimed at the audience (directivity 0.7) and the reverb bus is trimmed by -5 dB, so near a speaker the direct sound is about 4 dB above the reverb and at the start position the hall is a few dB above the direct sound instead of 12 dB (numbers and the test that pins them: `DEMO_*` in `src/main.rs`, `crates/quasar/tests/direct_vs_room_levels.rs`).

- **-** / **=** lower / raise the reverb trim by 2 dB (printed in the console).
- **;** / **'** lower / raise the early-reflection trim by 2 dB.
- **[** / **]** master volume in 3 dB steps (pre-limiter gain; the limiter ceiling stays at -1 dBFS).
