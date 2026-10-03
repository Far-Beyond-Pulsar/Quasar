# Round 7a worker notes

## #151 batched pair resolution (done)
- `HybridProbeSampler::resolve_batch` (quasar-core/src/hybrid.rs): one `query_spatial` for all pairs; `resolve` = batch of one.
- `update_scene_spatial` (quasar/src/lib.rs): phase 1 collect dirty pairs (same pose test), phase 2 one batch, phase 3 publish (body unchanged).
- Tests: quasar-backends/tests/batch_resolve_tests.rs (bitwise, grid in/out, empty/missing config, ignored timing), quasar/tests/batch_update_tests.rs (engine audio bit-identical batched vs per-pair).
- Timing (release, 32 cores, 400k-tri hall with ribs, 8 emitters): raw query_spatial 214.3 ms, resolve_batch 218.3 ms, per-pair loop 1618.2 ms (7.4x).

## #152 plane selection (code + tests done; measurements pending)
- build_planes(triangles,&CpuSimdConfig,probe) in cpu_simd.rs: dedupe -> coplanar patches -> buried probe -> merge tilted adjacent patches (5deg/0.25m) -> importance rank. ReflectPlane gets fitted/spread; locate_on_plane snaps bounce onto member triangle for merged groups. wgpu build_scene uses same fn + flat BVH probe. New config: plane_merge_angle_deg, plane_merge_offset, plane_buried_distance. New API: reflection_planes() -> ReflectionPlaneInfo.
- tests: crates/quasar-backends/tests/plane_selection_tests.rs. Existing backend tests all pass unchanged.
- other worker's failing test seen: quasar-audio source_resampler_tests::drift_plus_minus_100_ppm_same_rate (resampler, not mine).

- GPU: shader ray_trace.wgsl locate_on_plane now handles merged groups (bmin.w=spread, bmax.w=fitted flag); new test wgpu_parity_tests::faceted_vault_matches_cpu_backend passes on real adapter. All 7 parity tests ok.
- headless check running in bg: CARGO_TARGET_DIR=examples/basic/target-check, log /tmp/headless_r7a.log
