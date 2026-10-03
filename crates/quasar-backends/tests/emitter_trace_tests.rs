//! #156 in the CPU backend: emitter patterns prune / rank image-source paths, emitter shapes
//! spread the soft-occlusion probes, and the default (no table) is bit-identical.

mod common;

use common::{box_mesh, hall, Opaque, INSIDE, SRC};
use quasar_backends::cpu_simd::{CpuSimdComputeBackend, CpuSimdConfig};
use quasar_core::backend::{EarlyReflection, IAcousticComputeBackend, SpatialQuery, SpatialQueryResult};
use quasar_core::emitter_pattern::{EmitterPattern, EmitterShape, EmitterTrace};
use quasar_core::scene::AcousticScene;

fn query() -> SpatialQuery {
    SpatialQuery { source_position: SRC, listener_position: INSIDE, source_id: 0 }
}

fn run(backend: &CpuSimdComputeBackend) -> SpatialQueryResult {
    backend.query_spatial(&[query()], &Opaque).remove(0)
}

fn room_only() -> AcousticScene {
    let mut s = AcousticScene::new();
    s.add_mesh(box_mesh(1, [0.0, 0.0, 0.0], [20.0, 12.0, 30.0], true));
    s
}

fn first_order(scene: AcousticScene, max_reflections: usize) -> CpuSimdComputeBackend {
    CpuSimdComputeBackend::new(
        scene,
        CpuSimdConfig { max_reflection_order: 1, max_reflections, ..CpuSimdConfig::default() },
    )
}

fn trace(pattern: Option<EmitterPattern>, forward: [f32; 3], shape: EmitterShape) -> Vec<(u32, EmitterTrace)> {
    vec![(0, EmitterTrace { pattern, forward, shape })]
}

fn cone() -> EmitterPattern {
    // Full level inside 15 deg (half of 30), -60 dB beyond 25 deg (half of 50): every band is
    // far below the -40 dB prune threshold outside it.
    EmitterPattern::SoundCone { inner_deg: 30.0, outer_deg: 50.0, outer_gain_db: -60.0 }
}

fn has_direction(list: &[EarlyReflection], dir: [f32; 3]) -> bool {
    list.iter().any(|r| {
        let d = r.direction;
        d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2] > 0.999
    })
}

#[test]
fn an_empty_table_and_an_omni_default_entry_are_bit_identical() {
    let plain = CpuSimdComputeBackend::new(hall(), CpuSimdConfig::default());
    let mut with_entry = CpuSimdComputeBackend::new(hall(), CpuSimdConfig::default());
    with_entry.set_emitters(&trace(None, [0.0, 0.0, 1.0], EmitterShape::Default));
    let (a, b) = (run(&plain), run(&with_entry));
    assert_eq!(format!("{:?}", a.direct_path), format!("{:?}", b.direct_path));
    assert_eq!(format!("{:?}", a.early_reflections), format!("{:?}", b.early_reflections));
    assert_eq!(format!("{:?}", a.late_reverb), format!("{:?}", b.late_reverb));
    // An id that is not in the table is also omnidirectional.
    let mut other = CpuSimdComputeBackend::new(hall(), CpuSimdConfig::default());
    other.set_emitters(&[(7, EmitterTrace { pattern: Some(cone()), forward: [0.0, 0.0, 1.0], shape: EmitterShape::Point })]);
    assert_eq!(format!("{:?}", run(&other).early_reflections), format!("{:?}", a.early_reflections));
}

#[test]
fn a_cone_prunes_the_paths_that_leave_the_back_and_sides_and_keeps_the_rest_unchanged() {
    // Empty box, emitter at SRC (z = 5) facing +Z toward the listener (z = 25). First-order
    // image sources and the angle their FIRST leg leaves the emitter at (from +Z): far wall 0,
    // floor ~10, both side walls 45, ceiling ~46, rear wall 180. The cone (half-angles 15 / 25):
    // far wall and floor survive; sides, ceiling and rear are at -60 dB in every band: pruned.
    let forward = [0.0, 0.0, 1.0];
    let plain = first_order(room_only(), 64);
    let mut patterned = first_order(room_only(), 64);
    patterned.set_emitters(&trace(Some(cone()), forward, EmitterShape::Default));

    plain.reset_ray_counter();
    let a = run(&plain);
    let rays_plain = plain.rays_traced();
    patterned.reset_ray_counter();
    let b = run(&patterned);
    let rays_patterned = patterned.rays_traced();

    assert_eq!(a.early_reflections.len(), 6, "an empty box has six first-order paths");
    assert_eq!(b.early_reflections.len(), 2, "far wall and floor survive: {:?}", b.early_reflections.iter().map(|r| r.direction).collect::<Vec<_>>());
    // The far wall (+Z) survives; the rear (-Z), the sides (+-X) and the ceiling (+Y) do not.
    assert!(has_direction(&b.early_reflections, [0.0, 0.0, 1.0]));
    assert!(!has_direction(&b.early_reflections, [0.0, 0.0, -1.0]));
    assert!(!has_direction(&b.early_reflections, [0.70710677, 0.0, -0.70710677]));
    // Every surviving path is exactly the unpruned path: gains are never scaled by the backend.
    for r in &b.early_reflections {
        let same = a.early_reflections.iter().any(|p| {
            p.order == r.order && p.delay_samples == r.delay_samples && p.gain.0 == r.gain.0 && p.direction == r.direction
        });
        assert!(same, "surviving path differs from the unpruned one: {r:?}");
    }
    // Pruned paths cast no visibility rays.
    assert!(rays_patterned < rays_plain, "rays {rays_patterned} vs {rays_plain}");
    // The direct path is untouched (the engine applies the pattern to it).
    assert_eq!(format!("{:?}", a.direct_path), format!("{:?}", b.direct_path));
}

#[test]
fn ray_cost_follows_the_solid_angle_of_the_pattern() {
    // The same emitter in the hall (columns: more candidates): narrower cone => fewer rays.
    let cfg = || CpuSimdConfig { max_reflection_order: 2, max_reflections: 64, ..CpuSimdConfig::default() };
    let count = |pattern: Option<EmitterPattern>| {
        let mut b = CpuSimdComputeBackend::new(hall(), cfg());
        b.set_emitters(&trace(pattern, [0.0, 0.0, 1.0], EmitterShape::Default));
        b.reset_ray_counter();
        let _ = run(&b);
        b.rays_traced()
    };
    let omni = count(None);
    let wide = count(Some(EmitterPattern::SoundCone { inner_deg: 120.0, outer_deg: 240.0, outer_gain_db: -60.0 }));
    let narrow = count(Some(EmitterPattern::SoundCone { inner_deg: 30.0, outer_deg: 60.0, outer_gain_db: -60.0 }));
    assert!(wide <= omni && narrow < wide, "omni {omni}, wide {wide}, narrow {narrow}");
    assert!((narrow as f32) < 0.8 * omni as f32, "a narrow cone should save a clear share of the rays: {narrow} vs {omni}");
}

#[test]
fn the_reflection_cap_keeps_what_the_speaker_actually_radiates() {
    // Cap of two first-order paths. Unpatterned, the strongest two are the shortest ones (floor
    // and a side wall). A cone toward the listener (no pruning: outer gain -20 dB) re-weights
    // the ranking: the far-wall path (on axis) must now be among the two.
    let forward = [0.0, 0.0, 1.0];
    let plain = first_order(room_only(), 2);
    let mut patterned = first_order(room_only(), 2);
    patterned.set_emitters(&trace(
        Some(EmitterPattern::SoundCone { inner_deg: 60.0, outer_deg: 120.0, outer_gain_db: -20.0 }),
        forward,
        EmitterShape::Default,
    ));
    let a = run(&plain);
    let b = run(&patterned);
    assert_eq!(a.early_reflections.len(), 2);
    assert_eq!(b.early_reflections.len(), 2);
    assert!(!has_direction(&a.early_reflections, [0.0, 0.0, 1.0]), "unpatterned top-2 should not contain the far wall");
    assert!(has_direction(&b.early_reflections, [0.0, 0.0, 1.0]), "pattern-weighted top-2 must contain the on-axis far wall");
}

#[test]
fn the_emitter_shape_spreads_the_occlusion_probes() {
    // A 0.2 m pole exactly on the line of sight. A point source is fully behind it; the legacy
    // 0.35 m disc sees around it with most probes; a 3 m line array sees around it with nearly
    // all of them.
    let mut scene = room_only();
    scene.add_mesh(box_mesh(100, [9.9, 0.0, 14.9], [10.1, 12.0, 15.1], false));
    let factor = |shape: Option<EmitterShape>| {
        let mut b = CpuSimdComputeBackend::new(scene.clone(), CpuSimdConfig::default());
        if let Some(shape) = shape {
            b.set_emitters(&trace(None, [0.0, 0.0, 1.0], shape));
        }
        run(&b).direct_path.occlusion_factor
    };
    let point = factor(Some(EmitterShape::Point));
    let legacy = factor(None);
    let default_entry = factor(Some(EmitterShape::Default));
    let disc = factor(Some(EmitterShape::Disc { radius: 0.35 }));
    let line = factor(Some(EmitterShape::Line { half_length: 1.5, axis: [1.0, 0.0, 0.0] }));
    let sphere = factor(Some(EmitterShape::Sphere { radius: 0.35 }));
    assert_eq!(legacy, default_entry, "Default shape is the legacy behaviour");
    assert!(point < legacy, "point {point} vs legacy disc {legacy}");
    assert!(line > legacy, "line {line} vs legacy disc {legacy}");
    assert!(disc > point && sphere > point, "disc {disc}, sphere {sphere}, point {point}");
    // A bigger aperture can only see more of the source around a thin occluder.
    let big = factor(Some(EmitterShape::Disc { radius: 1.5 }));
    assert!(big >= disc - 1e-6, "big disc {big} vs disc {disc}");
}
