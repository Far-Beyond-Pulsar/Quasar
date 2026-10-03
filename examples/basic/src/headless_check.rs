//! `QUASAR_HEADLESS_CHECK=1` (or `--check`): no window, no audio device.
//!
//! Builds the SceneDB world on a surface-less wgpu device, reads the acoustic geometry out of it,
//! builds the engine, prints geometry / material / plane / timing statistics, runs
//! `update_scene_spatial` from several listener positions and prints a direct / early / late level
//! table (offline `process_audio_scene` with noise, like `crates/quasar/tests/direct_vs_room_levels.rs`).

use crate::acoustic_geometry::{self, AcousticClass, ExtractedScene};
use crate::audio_demo::{self, AUDIENCE, NUM_SPEAKERS};
use crate::v3_demo_common;
use helio_pass_gbuffer::{MeshComponent, StaticObjectComponent};
use pulsar_scenedb::World;
use quasar_backends::cpu_simd::CpuSimdComputeBackend;
use quasar_core::backend::{IAcousticComputeBackend, SpatialQuery};
use quasar_core::rays::Ray;
use quasar_core::scene_output::PhysicalOutputLayout;
use quasar_dsp::audio_buffer::AudioBuffer;
use quasar_dsp::limiter::OutputSafetyConfig;
use std::collections::HashMap;
use std::time::Instant;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Build the large cathedral into a fresh SceneDB world on a surface-less GPU device. The world
/// keeps its `Arc`s to the device alive through its GPU mirror.
pub fn build_world() -> pulsar_scenedb::SceneDb {
    let (device, queue) = v3_demo_common::headless_gpu();
    let mut scene_db = v3_demo_common::new_scene_db_with_gpu_mirror(&device, &queue);
    crate::populate_large_cathedral(&mut scene_db.world);
    scene_db
}

/// Counts of what the SceneDB would render, independent of the acoustic extraction.
#[derive(Debug)]
pub struct RenderCounts {
    pub objects: usize,
    pub mesh_rows: usize,
    pub draw_triangles: usize,
    pub authored_triangles: usize,
}

pub fn render_counts(world: &World) -> RenderCounts {
    let mut objects = 0;
    let mut draw_triangles = 0;
    for (_, (o,)) in world.query::<(&StaticObjectComponent,)>() {
        objects += 1;
        draw_triangles += o.index_count as usize / 3;
    }
    RenderCounts {
        objects,
        mesh_rows: world.query::<(&MeshComponent,)>().count(),
        draw_triangles,
        authored_triangles: crate::cathedral_large::AUTHORED_TRIANGLES.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Closest distance from `p` to triangle `abc` (Ericson, Real-Time Collision Detection 5.1.5).
fn point_triangle_distance(p: glam::Vec3, a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// World-space triangles of an extracted scene with their class.
fn world_triangles(ex: &ExtractedScene) -> Vec<([glam::Vec3; 3], AcousticClass)> {
    let mut out = Vec::new();
    for (mesh, &class) in ex.scene.meshes.iter().zip(&ex.classes) {
        let m = glam::Mat4::from_cols_array(&mesh.transform);
        let p: Vec<glam::Vec3> = mesh.positions.iter().map(|v| m.transform_point3(glam::Vec3::from_array(*v))).collect();
        for t in mesh.indices.chunks_exact(3) {
            out.push(([p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]], class));
        }
    }
    out
}

/// A replica of the backend's mirror-plane selection (`build_planes`: coplanar triangles merged
/// by plane equation, largest total area first, `max_planes` kept) for DIAGNOSTICS. The tolerances
/// below are copied from the crate (normal cos / offset tolerance) so the ranking matches.
pub struct PlaneInfo {
    pub normal: [f32; 3],
    pub offset: f32,
    pub area: f32,
    pub triangles: usize,
    pub dominant_class: AcousticClass,
}

pub fn dominant_planes(tris: &[([glam::Vec3; 3], AcousticClass)], max_planes: usize) -> (Vec<PlaneInfo>, usize) {
    struct P {
        n: glam::Vec3,
        d: f32,
        area: f32,
        tris: usize,
        by_class: HashMap<AcousticClass, f32>,
    }
    // Quantised key: normal cells of 1/50 and offset cells of 2 cm (the crate merges by tolerance
    // instead of by grid; the grid keeps this replica O(n)).
    let mut map: HashMap<(i32, i32, i32, i32), P> = HashMap::new();
    for (t, class) in tris {
        let cross = (t[1] - t[0]).cross(t[2] - t[0]);
        let len = cross.length();
        if len < 1e-9 {
            continue;
        }
        let mut n = cross / len;
        let axis = if n.x.abs() >= n.y.abs() && n.x.abs() >= n.z.abs() { 0 } else if n.y.abs() >= n.z.abs() { 1 } else { 2 };
        if n[axis] < 0.0 {
            n = -n;
        }
        let d = n.dot(t[0]);
        let key = ((n.x * 50.0).round() as i32, (n.y * 50.0).round() as i32, (n.z * 50.0).round() as i32, (d * 50.0).round() as i32);
        let area = 0.5 * len;
        let p = map.entry(key).or_insert_with(|| P { n, d, area: 0.0, tris: 0, by_class: HashMap::new() });
        p.area += area;
        p.tris += 1;
        *p.by_class.entry(*class).or_default() += area;
    }
    let total = map.len();
    let mut planes: Vec<P> = map.into_values().collect();
    planes.sort_by(|a, b| b.area.total_cmp(&a.area));
    planes.truncate(max_planes);
    (
        planes
            .into_iter()
            .map(|p| PlaneInfo {
                normal: p.n.to_array(),
                offset: p.d,
                area: p.area,
                triangles: p.tris,
                dominant_class: p.by_class.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(c, _)| *c).unwrap(),
            })
            .collect(),
        total,
    )
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// RMS of the TOTAL power over all output channels (last 100 of `blocks`) and of one input channel.
/// `active` lists the WAV channels fed with noise.
fn render_rms(e: &mut quasar_audio::SpatialAudioEngine, stage: u8, channels: u16, active: &[usize], blocks: usize) -> (f32, f32) {
    e.debug_audio_stage = stage;
    let mut seeds: Vec<u32> = (0..8u32).map(|i| 0x1234_5678 ^ (i + 1).wrapping_mul(0x9E37_79B9)).collect();
    let mut lps = [0.0_f32; 8];
    let mut out = AudioBuffer::new(channels, BLOCK as u16);
    let (mut sum, mut n, mut sum_in) = (0.0_f64, 0usize, 0.0_f64);
    for b in 0..blocks {
        let mut input = AudioBuffer::new(8, BLOCK as u16);
        for &s in active {
            for i in 0..BLOCK {
                seeds[s] = seeds[s].wrapping_mul(1664525).wrapping_add(1013904223);
                let white = (seeds[s] >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                lps[s] += 0.2 * (white - lps[s]);
                input.set(s as u16, i as u16, 1.5 * lps[s]);
            }
        }
        e.process_audio_scene(&[&input], std::slice::from_mut(&mut out));
        if b >= blocks - 100 {
            for i in 0..BLOCK {
                let mut p = 0.0_f64;
                for c in 0..channels {
                    let v = out.channel(c)[i] as f64;
                    p += v * v;
                }
                sum += p;
                sum_in += (input.channel(active[0] as u16)[i] as f64).powi(2);
                n += 1;
            }
        }
    }
    ((sum / n as f64).sqrt() as f32, (sum_in / n as f64).sqrt() as f32)
}

/// Run the whole check; returns `Err` with a message when a hard expectation fails.
pub fn run() -> Result<(), String> {
    let t0 = Instant::now();
    let scene_db = build_world();
    let world = &scene_db.world;
    println!("[check] SceneDB world built headless in {:.2} s", t0.elapsed().as_secs_f64());

    let counts = render_counts(world);
    println!("[check] SceneDB render side: {counts:?}");

    let built = audio_demo::build_engine(
        world,
        SR,
        "assets/8_Channel_ID.wav",
        8,
        PhysicalOutputLayout::Surround714,
        AUDIENCE,
        audio_demo::tracer_config(SR),
    );
    let s = &built.scene_stats;
    println!("[check] acoustic scene: {}", s.summary());
    println!(
        "[check] completeness: authored {} tris, SceneDB draw ranges {} tris over {} objects ({} mesh rows), audio scene {} tris over {} instances",
        counts.authored_triangles, counts.draw_triangles, counts.objects, counts.mesh_rows, s.triangles, s.instances
    );
    let mut failures: Vec<String> = Vec::new();
    if s.instances != counts.objects {
        failures.push(format!("instances {} != SceneDB objects {}", s.instances, counts.objects));
    }
    if s.triangles != counts.draw_triangles || s.triangles != counts.authored_triangles {
        failures.push("triangle counts disagree".into());
    }
    if s.fallback_materials != 0 {
        failures.push(format!("{} material rows needed the inferred fallback", s.fallback_materials));
    }
    if !s.skipped.is_empty() || s.invalid_triangles != 0 {
        failures.push(format!("skipped {:?}, invalid triangles {}", s.skipped, s.invalid_triangles));
    }
    println!(
        "[check] scene AABB {:?} .. {:?} (size {:?}); {} degenerate triangles",
        s.aabb_min, s.aabb_max, s.aabb_size(), s.degenerate_triangles
    );
    println!(
        "[check] backend: BVH + planes built in {:.1} ms, extraction {:.1} ms, {} mirror planes, room_is_closed = {}",
        built.backend_build_ms, built.extract_ms, built.reflection_planes, built.room_closed
    );
    println!(
        "[check] statistical late-field T60 per band (62.5 Hz..8 kHz) {:?} s; probe grid {:?} probes @ {:?} m",
        built.statistical_t60.0.iter().map(|t| (t * 100.0).round() / 100.0).collect::<Vec<_>>(),
        built.probe_grid.dims,
        built.probe_grid.spacing,
    );

    // Independent copy of the scene for ray statistics and plane analysis.
    let handles = built.class_handles.clone();
    let ex = acoustic_geometry::extract_acoustic_scene(world, |c| handles[&c]);
    let tris = world_triangles(&ex);
    let (planes, plane_candidates) = dominant_planes(&tris, 32);
    println!("[check] mirror-plane candidates {plane_candidates}; the 32 largest (replica of the crate ranking):");
    for (i, p) in planes.iter().enumerate() {
        println!(
            "[check]   #{i:<2} n=({:+.2},{:+.2},{:+.2}) d={:+7.2} area {:8.1} m2, {:6} tris, mostly {}",
            p.normal[0], p.normal[1], p.normal[2], p.offset, p.area, p.triangles, p.dominant_class.name()
        );
    }

    // Speaker clearance and position checks.
    let speakers = audio_demo::speaker_positions(AUDIENCE);
    let probe_backend = CpuSimdComputeBackend::new(ex.scene.clone(), audio_demo::tracer_config(SR));
    for (i, &sp) in speakers.iter().enumerate() {
        let clearance = tris.iter().map(|(t, _)| point_triangle_distance(sp, t[0], t[1], t[2])).fold(f32::MAX, f32::min);
        let up = probe_backend
            .trace_ray(&Ray { origin: sp.to_array(), direction: [0.0, 1.0, 0.0], min_distance: 0.0, max_distance: f32::MAX })
            .first()
            .map(|h| h.point[1]);
        println!("[check] speaker {i} at {:?}: nearest surface {:.2} m, first hit above y = {:?}", sp.to_array(), clearance, up);
        if clearance < 0.3 {
            failures.push(format!("speaker {i} is only {clearance:.2} m from geometry"));
        }
    }

    // Engine: spatial updates + levels from several listener positions.
    let mut engine = built.engine;
    let lid = built.listener_id;
    engine.set_output_safety(lid, OutputSafetyConfig { enabled: false, ..OutputSafetyConfig::default() });
    let queries = |listener: [f32; 3]| -> Vec<SpatialQuery> {
        speakers.iter().enumerate().map(|(i, p)| SpatialQuery { source_position: p.to_array(), listener_position: listener, source_id: i as u32 }).collect()
    };

    let positions: [(&str, [f32; 3]); 7] = [
        ("audience point / start", AUDIENCE),
        ("entrance", [0.0, 2.3, 67.0]),
        ("2 m from centre speaker", [0.0, 2.3, speakers[2].z + 2.0]),
        ("nave middle", [0.0, 2.3, 0.0]),
        ("near altar", [0.0, 2.3, -60.0]),
        ("outside, past the entrance", [0.0, 2.3, 100.0]),
        ("outside, beside the north aisle", [60.0, 2.3, 0.0]),
    ];
    println!("[check] spatial updates (8 emitters x 1 listener), release build recommended:");
    let materials = engine.materials();
    for (label, pos) in positions {
        probe_backend.reset_ray_counter();
        let t = Instant::now();
        let results = probe_backend.query_spatial(&queries(pos), materials);
        let q_ms = t.elapsed().as_secs_f64() * 1e3;
        let rays = probe_backend.rays_traced();
        let early: usize = results.iter().map(|r| r.early_reflections.len()).sum();
        let occluded = results.iter().filter(|r| r.direct_path.occluded).count();
        println!("[check]   {label:<32} {pos:?}: {rays:6} rays, query {q_ms:7.2} ms, {early:3} early paths, {occluded} occluded directs");
    }
    // Wall-clock of the real engine update (compute thread work) at the same positions.
    for (label, pos) in positions {
        engine.update_listener(lid, pos, [0.0, 0.0, -1.0]);
        engine.update_scene_spatial(); // warm (first update after a jump)
        let t = Instant::now();
        for _ in 0..5 {
            engine.update_listener(lid, pos, [0.0, 0.0, -1.0]);
            engine.update_scene_spatial();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 5.0;
        println!("[check]   update_scene_spatial {label:<32} {ms:7.2} ms (30 Hz budget 33.3 ms)");
    }

    // Levels, dB re one input channel, total power over the 8 output channels.
    println!("[check] level table (7.1 output, noise; direct / early / late power over all channels, re ONE input channel):");
    let centre_wav = audio_demo::CHANNEL_MAP[2] as usize;
    for (label, pos) in positions.iter().take(5) {
        for (what, active) in [("centre speaker only", vec![centre_wav]), ("all 8 speakers", (0..NUM_SPEAKERS).collect::<Vec<_>>())] {
            engine.update_listener(lid, *pos, [0.0, 0.0, -1.0]);
            engine.update_scene_spatial();
            let mut lv = [0.0_f32; 3];
            let mut input = 1.0;
            for (i, stage) in [2u8, 3, 4].iter().enumerate() {
                let (o, inp) = render_rms(&mut engine, *stage, 8, &active, 1000);
                lv[i] = o;
                input = inp;
            }
            let direct = lv[0];
            let early = (lv[1] * lv[1] - lv[0] * lv[0]).max(0.0).sqrt();
            let late = (lv[2] * lv[2] - lv[1] * lv[1]).max(0.0).sqrt();
            println!(
                "[check]   {label:<26} {what:<20} direct {:6.1} | early {:6.1} | late {:6.1} dB   late-direct {:+5.1} dB",
                db(direct / input),
                db(early / input),
                db(late / input),
                db(late) - db(direct)
            );
        }
    }

    if failures.is_empty() {
        println!("[check] PASS");
        Ok(())
    } else {
        for f in &failures {
            println!("[check] FAIL: {f}");
        }
        Err(failures.join("; "))
    }
}
