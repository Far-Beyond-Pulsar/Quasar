# Cathedral audio demo

Run `cargo run --release --manifest-path examples/basic/Cargo.toml` from the repository root.

Press **V** to toggle the acoustic trace overlay (**R** is an alias). The capture starts disabled and key repeat will not toggle it repeatedly.

- **Blue:** ray tests that reached the end of the query interval without a hit.
- **Red:** ray tests that hit acoustic geometry, ending at the actual intersection, with short surface normals.
- **Purple:** valid reflection candidates before strongest-path selection.
- **Green:** paths selected by the backend for audio, from speaker through each bounce to listener.
- **Yellow:** selected bounce points and surface normals.

Lines show through walls. The window title shows ray and path counts. The overlay grows its own GPU line buffer to fit all captured rays. Capture can be expensive in dense scenes.

The drawing refreshes every display frame. Captured geometry refreshes with the engine's existing roughly 30 Hz spatial update, across all speaker/listener pairs, so it does not restart audio crossfades on every display frame.

The demo's acoustic scene contains the room shell and columns, rather than every decorative visual mesh. Early reflections use image-source geometry and BVH visibility queries. Late reverb is statistical and has no traced reflection paths.
