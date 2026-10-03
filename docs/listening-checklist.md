# Listening validation checklist

The objective harness (`docs/validation.md`) says whether the engine does what its model says.
It cannot say whether the result is convincing. This checklist is the repeatable listening
process that complements it. Nothing here has been signed off by a listener yet: the "expected"
columns are what the harness numbers predict you should perceive, and a mismatch between the
two is a finding either way (a bug in the model, or a number that does not mean what we think).

## Setup

* Headphones for the HRTF listener, a calibrated speaker layout (stereo, 5.1 or 7.1) for the
  others; level-match the rig with the pink-noise reference below before judging anything.
* Run the demo with `cargo run --release --manifest-path examples/basic/Cargo.toml` (cathedral,
  8 stage speakers, `assets/8_Channel_ID.wav` announces each speaker in turn) or listen to a
  rendered file of the scenes below.
* Note the engine commit, the output device, the layout and the master gain. Master gain `[` / `]`
  changes the level in 3 dB steps (the output limiter holds the -1 dBFS ceiling); the demo
  default is quiet for this asset (see `docs/worker-notes-round5.md`), raise it first.
* Blind where possible: have a second person switch between A and B (ABX) so that you do not
  know which is which. Three or more listeners; record `correct / trials` per question.

## 1. Localisation

Reference scene (harness: `panning_law_is_constant_power_and_points_at_the_source`,
`hrtf_listener_has_woodworth_itd_and_a_head_shadow`): one emitter 2 m from the listener, listener
facing -Z, noise or speech, azimuth swept from -90 to +90 degrees in 15 degree steps, in random
order.

| question | expected | measured by the harness |
|---|---|---|
| Is the image where the azimuth says (eyes closed, point at it)? | Within about +-10 degrees for loudspeakers between the front speakers (amplitude panning, no head tracking); sources between two speakers 110 degrees apart (5.1 surround pair) are less stable | velocity-vector error <= 0.01 deg |
| Does the level stay constant while it moves? | Yes, no loudness dip between speakers | total power constant within 0.1 dB |
| Does it jump when it crosses a speaker? | No | gains are continuous (see `vbap_tests.rs`) |
| HRTF: left / right correct, front / back and elevation? | Left / right clear, front / back weak (a parametric model: no pinna cues beyond a notch), distance externalisation modest | ITD within 16 us of Woodworth, ILD 4 .. 14 dB |
| 5.1 / 7.1: nothing in the subwoofer except the LFE send | Yes | the LFE slot receives no panned signal |

## 2. Distance

Reference scene (`loudness_follows_the_inverse_distance_law_and_the_ceiling_holds`,
`direct_impulse_arrives_after_distance_over_c`): the same emitter at 1, 2, 4 and 12 m straight ahead.

* Level drops about 6 dB per doubling (6.0 .. 7.0 LU); judge with eyes closed which of two
  distances is farther (should be unanimous from a factor of two).
* Click or impulse train: the arrival is delayed by 2.9 ms per metre (a 12 m emitter is 35 ms
  late). Move the emitter: the pitch glides (Doppler) instead of clicking.
* Reverberation does NOT fall with distance (diffuse field): far away the direct-to-reverberant
  ratio falls, which is the distance cue. Critical distance of a large hall is several metres
  (`rc = 0.057 sqrt(Q V / T60)`).

## 3. Occlusion and diffraction

Reference scene (`crates/quasar-backends/tests/occlusion_model_tests.rs`, demo): emitter behind a
column, then behind a closed wall, listener walking out of sight.

* Behind a column: level drops and the sound gets duller smoothly (HF first), no click when the
  line of sight is lost or regained; diffraction keeps a soft path around the edge.
* Behind a closed wall: much quieter and dark; with an opaque closed shell the early reflections
  are skipped ("outside: reflections skipped" in the window title), the late field remains
  strongly attenuated (-20 dB per outside party).
* Expected defect to watch for: zipper noise while moving across the shadow boundary.

## 4. Reverb and balance

Reference scenes (`direct_vs_room_levels.rs`, `rendered_reverb_has_the_probe_rt60_edt_c50_and_d50`):

* **Decay**: an impulse or a staccato note in the cathedral at the start position; the tail
  decays like a 4 - 7 s hall (probe T60). Clap test: does the length match the room size?
* **Balance near a speaker** (2 m from the centre speaker, all 8 playing): the direct sound must
  be clearly in front, not "only echoes". Harness (demo mix): direct 4.3 dB over the reverb, early
  reflections 5 dB under the direct sound.
* **Balance far away** (start position, 12 m): the hall is audible and the sources remain
  locatable; the reverb is about 4 dB over the direct sound (demo mix, was 12 dB before).
* **Trims**: `-` / `=` (reverb) and `;` / `'` (early reflections) in 2 dB steps must change only
  their component, smoothly. At -30 dB reverb the room should sound dry and the speakers close.
* Physical defaults (`set_reverb_gain_db(.., 0.0)`) sound much more reverberant than the demo mix;
  that is the correct level for omnidirectional speakers and it is expected to feel "too wet".

## 5. Artefacts

Listen for, on every scene above and while moving through the room: clicks at block edges (the
direct sound once had dropouts with 224-frame callbacks, `block_size_tests.rs`), zipper noise when
pulls, trims or gains change, pitch steps while a source moves, a metallic ring in the reverb
tail (FDN), and level pumping from the output limiter on hot material.

## Recording results

| date | commit | device / layout | scene | question | listeners | result | note |
|---|---|---|---|---|---|---|---|

A finding that contradicts a harness assertion gets a regression test in
`crates/quasar/tests/acoustic_validation.rs` once it is understood.
