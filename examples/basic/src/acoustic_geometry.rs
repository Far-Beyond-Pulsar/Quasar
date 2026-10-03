//! Acoustic geometry read from the SceneDB world.
//!
//! The renderer's scene is the single source of truth: every object row
//! (`StaticObjectComponent`: mesh slot, material slot, transform) in the
//! [`World`] becomes one [`AcousticMesh`] of Quasar's [`AcousticScene`], with
//! the CPU vertex/index payload read back from the object's `MeshComponent` and
//! the material class read from the object's material row. Nothing in the
//! scene is hand-built for audio, and anything spawned into the world (now or
//! before the next extraction) is included.
//!
//! Acoustic materials: the material row carries an [`AcousticSurface`] tag
//! (authored next to the render material, see `cathedral_large.rs`). A material
//! without a tag is classified from the Helio material data (alpha blend ->
//! glass, metallic -> bronze, emissive -> flame, otherwise rough stone) and the
//! row is counted as a *fallback* so a missing tag is never silent.
//!
//! The absorption / transmission tables below are plausible engineering
//! estimates from typical published octave-band tables (stone masonry,
//! polished marble, oak, window glass, bronze) and from the mass law for the
//! transmission of the heavy shell. They are NOT measurements of any building.
//! Scattering is 0 for every material for now.

use glam::Vec3;
use helio_pass_gbuffer::{MaterialComponent, MeshComponent, StaticObjectComponent};
use pulsar_scenedb::World;
use quasar_core::bands::Band8;
use quasar_core::scene::{AcousticMesh, AcousticScene};
use std::collections::HashMap;


/// Acoustic surface class of a material row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AcousticClass {
    /// Rough limestone masonry: walls, clerestory, vault web, window surrounds.
    Stone,
    /// Carved limestone: ribs, piers, arcade arches, mullions, tracery.
    CarvedStone,
    /// Dark polished basalt: the procession strips of the nave floor.
    Basalt,
    /// Marble / stone flag paving: the nave floor.
    Paving,
    /// Altar and sanctuary steps (smooth limestone / marble).
    AltarStone,
    /// Carved oak: pews.
    Oak,
    /// Bronze: chandeliers, candle holders, window saddle bars.
    Bronze,
    /// Candle wax (small cylinders).
    Wax,
    /// Candle flames (hot gas: acoustically almost transparent).
    Flame,
    /// Leaded stained glass panes.
    StainedGlass,
}

impl AcousticClass {
    pub const ALL: [AcousticClass; 10] = [
        AcousticClass::Stone,
        AcousticClass::CarvedStone,
        AcousticClass::Basalt,
        AcousticClass::Paving,
        AcousticClass::AltarStone,
        AcousticClass::Oak,
        AcousticClass::Bronze,
        AcousticClass::Wax,
        AcousticClass::Flame,
        AcousticClass::StainedGlass,
    ];

    pub fn name(self) -> &'static str {
        match self {
            AcousticClass::Stone => "stone",
            AcousticClass::CarvedStone => "carved stone",
            AcousticClass::Basalt => "basalt",
            AcousticClass::Paving => "marble paving",
            AcousticClass::AltarStone => "altar stone",
            AcousticClass::Oak => "oak",
            AcousticClass::Bronze => "bronze",
            AcousticClass::Wax => "wax",
            AcousticClass::Flame => "flame",
            AcousticClass::StainedGlass => "stained glass",
        }
    }

    /// Per-band energy absorption coefficient (62.5 Hz .. 8 kHz).
    pub fn absorption(self) -> [f32; 8] {
        match self {
            // Unpainted limestone masonry with joints: published 125 Hz-4 kHz values 0.02-0.08.
            AcousticClass::Stone => [0.03, 0.03, 0.04, 0.06, 0.07, 0.08, 0.10, 0.12],
            // Carved / moulded stone: more surface area and relief, slightly higher.
            AcousticClass::CarvedStone => [0.04, 0.04, 0.05, 0.07, 0.08, 0.10, 0.12, 0.14],
            // Polished stone: 0.01-0.02 up to 2 kHz.
            AcousticClass::Basalt => [0.01, 0.01, 0.01, 0.015, 0.02, 0.02, 0.025, 0.03],
            AcousticClass::Paving => [0.01, 0.01, 0.015, 0.02, 0.02, 0.025, 0.03, 0.04],
            AcousticClass::AltarStone => [0.01, 0.01, 0.015, 0.02, 0.02, 0.025, 0.03, 0.04],
            // Varnished timber furniture / panelling.
            AcousticClass::Oak => [0.20, 0.15, 0.11, 0.09, 0.08, 0.08, 0.08, 0.09],
            // Metal.
            AcousticClass::Bronze => [0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05],
            AcousticClass::Wax => [0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            AcousticClass::Flame => [0.0; 8],
            // Window glass: 0.35 (125 Hz) falling to 0.04 (4 kHz).
            AcousticClass::StainedGlass => [0.35, 0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.04],
        }
    }

    /// Per-band AMPLITUDE transmission of one surface crossing.
    pub fn transmission(self) -> [f32; 8] {
        match self {
            // Heavy masonry shell: the proposal of GitHub issue #142 (about -32 dB at
            // 62 Hz down to -60 dB at 4 kHz; the real 0.9 m walls are more massive still).
            AcousticClass::Stone
            | AcousticClass::CarvedStone
            | AcousticClass::Basalt
            | AcousticClass::Paving
            | AcousticClass::AltarStone => [0.025, 0.016, 0.009, 0.005, 0.0028, 0.0016, 0.001, 0.001],
            // Solid oak, tens of mm thick (mass law, 15-30 dB).
            AcousticClass::Oak => [0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02],
            // Thin rods and small bodies (radius 2-7 cm): below a few kHz the wavelength is
            // far larger than the object, so sound diffracts around it and the surface
            // crossing barely attenuates; only the highest bands see a shadow.
            AcousticClass::Bronze => [0.98, 0.96, 0.93, 0.85, 0.70, 0.50, 0.35, 0.30],
            AcousticClass::Wax => [0.98, 0.96, 0.93, 0.85, 0.70, 0.50, 0.35, 0.30],
            AcousticClass::Flame => [1.0, 1.0, 1.0, 0.99, 0.98, 0.97, 0.95, 0.95],
            // ~5 mm leaded glass, mass law with a coincidence dip near 2-4 kHz.
            AcousticClass::StainedGlass => [0.17, 0.12, 0.07, 0.04, 0.025, 0.035, 0.04, 0.03],
        }
    }

    /// Scattering coefficient (0 for every class for now).
    pub fn scattering(self) -> [f32; 8] {
        [0.0; 8]
    }

    /// Absorption / scattering / transmission as `Band8`s, ready for
    /// `Tabular8BandEvaluator::create_params`.
    pub fn bands(self) -> (Band8, Band8, Band8) {
        (Band8::new(self.absorption()), Band8::new(self.scattering()), Band8::new(self.transmission()))
    }
}

/// Tag component authored on a *material* row (next to its `MaterialComponent`)
/// stating which acoustic class the surface belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcousticSurface(pub AcousticClass);

/// How a material row got its acoustic class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassSource {
    /// An explicit [`AcousticSurface`] tag.
    Tagged,
    /// Inferred from the Helio material data (counts as a fallback).
    Inferred,
}

/// Classify an untagged material from its Helio render data.
pub fn infer_class(material: &MaterialComponent) -> AcousticClass {
    if material.flags & helio_mats::FLAG_ALPHA_BLEND != 0 {
        AcousticClass::StainedGlass
    } else if material.roughness_metallic[1] >= 0.5 {
        AcousticClass::Bronze
    } else if material.emissive[3] > 0.0 && material.emissive[..3].iter().any(|&e| e > 0.0) {
        AcousticClass::Flame
    } else {
        AcousticClass::Stone
    }
}

/// Per-class totals of an extraction.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassStats {
    pub instances: usize,
    pub triangles: usize,
    /// Sum of triangle areas, m^2 (both faces of a closed solid count once each).
    pub area: f64,
}

/// Counts and bounds of one extraction, for logs and tests.
#[derive(Clone, Debug, Default)]
pub struct ExtractStats {
    /// Object rows (`StaticObjectComponent`) found in the world.
    pub object_rows: usize,
    /// Object rows converted to acoustic meshes.
    pub instances: usize,
    /// Object rows skipped (missing mesh row / empty mesh / non-finite transform), with a reason.
    pub skipped: Vec<String>,
    /// Distinct mesh rows referenced by the instances.
    pub unique_meshes: usize,
    pub triangles: usize,
    pub vertices: usize,
    /// Triangles with an out-of-range index (dropped by the backend; must be 0).
    pub invalid_triangles: usize,
    /// Triangles whose world-space area is below 1e-9 m^2 (dropped by the backend).
    pub degenerate_triangles: usize,
    /// World-space AABB of every vertex.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Material rows referenced by at least one object, and how many needed inference.
    pub materials_used: usize,
    pub tagged_materials: usize,
    pub fallback_materials: usize,
    pub by_class: HashMap<AcousticClass, ClassStats>,
    /// Instances whose transform is not the identity (the cathedral batches everything into
    /// identity-placed meshes; this proves the transform path is exercised when non-zero).
    pub transformed_instances: usize,
}

impl ExtractStats {
    pub fn aabb_size(&self) -> [f32; 3] {
        std::array::from_fn(|i| self.aabb_max[i] - self.aabb_min[i])
    }

    pub fn summary(&self) -> String {
        let mut classes: Vec<_> = self.by_class.iter().collect();
        classes.sort_by_key(|(c, _)| **c);
        let per_class = classes
            .iter()
            .map(|(c, s)| format!("{} {}i/{}t/{:.0}m2", c.name(), s.instances, s.triangles, s.area))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{} object rows -> {} instances ({} unique meshes, {} transformed), {} triangles, {} vertices, \
             AABB {:?}..{:?}, {} materials ({} tagged, {} FALLBACK), {} invalid / {} degenerate triangles, {} skipped; [{}]",
            self.object_rows,
            self.instances,
            self.unique_meshes,
            self.transformed_instances,
            self.triangles,
            self.vertices,
            self.aabb_min,
            self.aabb_max,
            self.materials_used,
            self.tagged_materials,
            self.fallback_materials,
            self.invalid_triangles,
            self.degenerate_triangles,
            self.skipped.len(),
            per_class,
        )
    }
}

/// Result of [`extract_acoustic_scene`].
pub struct ExtractedScene {
    pub scene: AcousticScene,
    pub stats: ExtractStats,
    /// Acoustic class of each instance, parallel to `scene.meshes`.
    pub classes: Vec<AcousticClass>,
}

/// Read every renderable object out of `world` into an [`AcousticScene`].
///
/// `material_handle(class)` returns the engine material handle of a class (the
/// caller registers one acoustic material instance per class).
pub fn extract_acoustic_scene(world: &World, mut material_handle: impl FnMut(AcousticClass) -> u32) -> ExtractedScene {
    let mut stats = ExtractStats::default();

    // Mesh rows by SceneDB row index.
    let meshes: HashMap<u32, &MeshComponent> =
        world.query::<(&MeshComponent,)>().map(|(entity, (mesh,))| (entity.index(), mesh)).collect();

    // Material rows by row index: class + how it was found.
    let mut materials: HashMap<u32, (AcousticClass, ClassSource)> = HashMap::new();
    for (entity, (material,)) in world.query::<(&MaterialComponent,)>() {
        let entry = match world.get::<AcousticSurface>(entity) {
            Some(tag) => (tag.0, ClassSource::Tagged),
            None => (infer_class(material), ClassSource::Inferred),
        };
        materials.insert(entity.index(), entry);
    }

    let mut scene = AcousticScene::new();
    let mut classes = Vec::new();
    let mut used_materials: HashMap<u32, ()> = HashMap::new();
    let mut used_meshes: HashMap<u32, ()> = HashMap::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    // Deterministic order: by object row index.
    let mut objects: Vec<(u32, &StaticObjectComponent)> =
        world.query::<(&StaticObjectComponent,)>().map(|(entity, (object,))| (entity.index(), object)).collect();
    objects.sort_by_key(|(index, _)| *index);
    stats.object_rows = objects.len();

    for (object_index, object) in objects {
        let Some(mesh) = meshes.get(&object.mesh_slot) else {
            stats.skipped.push(format!("object {object_index}: no MeshComponent row {}", object.mesh_slot));
            continue;
        };
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            stats.skipped.push(format!("object {object_index}: empty mesh row {}", object.mesh_slot));
            continue;
        }
        if object.index_count as usize != mesh.indices.len() {
            stats.skipped.push(format!(
                "object {object_index}: draws {} of the mesh's {} indices (sub-ranges are not supported)",
                object.index_count,
                mesh.indices.len()
            ));
            continue;
        }
        // Column-major, exactly `Mat4::to_cols_array_2d` flattened.
        let flat: [f32; 16] = std::array::from_fn(|i| object.transform[i / 4][i % 4]);
        if flat.iter().any(|v| !v.is_finite()) {
            stats.skipped.push(format!("object {object_index}: non-finite transform"));
            continue;
        }
        let Some(&(class, source)) = materials.get(&object.material_slot) else {
            stats.skipped.push(format!("object {object_index}: no MaterialComponent row {}", object.material_slot));
            continue;
        };
        if used_materials.insert(object.material_slot, ()).is_none() {
            stats.materials_used += 1;
            match source {
                ClassSource::Tagged => stats.tagged_materials += 1,
                ClassSource::Inferred => stats.fallback_materials += 1,
            }
        }
        used_meshes.insert(object.mesh_slot, ());

        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        let identity = glam::Mat4::from_cols_array(&flat) == glam::Mat4::IDENTITY;
        if !identity {
            stats.transformed_instances += 1;
        }
        let class_stats = stats.by_class.entry(class).or_default();
        class_stats.instances += 1;
        let m = glam::Mat4::from_cols_array(&flat);
        let world_pos: Vec<Vec3> = positions.iter().map(|p| m.transform_point3(Vec3::from_array(*p))).collect();
        for p in &world_pos {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        for tri in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            if a >= world_pos.len() || b >= world_pos.len() || c >= world_pos.len() {
                stats.invalid_triangles += 1;
                continue;
            }
            let area = 0.5 * (world_pos[b] - world_pos[a]).cross(world_pos[c] - world_pos[a]).length();
            if area < 1.0e-9 {
                stats.degenerate_triangles += 1;
            }
            class_stats.triangles += 1;
            class_stats.area += area as f64;
        }
        stats.triangles += mesh.indices.len() / 3;
        stats.vertices += positions.len();
        stats.instances += 1;

        scene.add_mesh(AcousticMesh {
            id: object_index as u64,
            positions,
            indices: mesh.indices.clone(),
            material_handle: material_handle(class),
            transform: flat,
        });
        classes.push(class);
    }
    stats.unique_meshes = used_meshes.len();
    if stats.instances == 0 {
        min = [0.0; 3];
        max = [0.0; 3];
    }
    stats.aabb_min = min;
    stats.aabb_max = max;
    ExtractedScene { scene, stats, classes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_core::PackedVertex;

    fn vertex(p: [f32; 3]) -> PackedVertex {
        PackedVertex { position: p, ..Default::default() }
    }

    fn quad_mesh(world: &mut World) -> pulsar_scenedb::Entity {
        let e = world.spawn();
        world.insert(
            e,
            MeshComponent {
                vertices: vec![vertex([0.0, 0.0, 0.0]), vertex([1.0, 0.0, 0.0]), vertex([1.0, 0.0, 1.0]), vertex([0.0, 0.0, 1.0])],
                indices: vec![0, 1, 2, 0, 2, 3],
            },
        );
        e
    }

    fn add_object(world: &mut World, mesh: pulsar_scenedb::Entity, material: pulsar_scenedb::Entity, m: glam::Mat4) {
        let o = world.spawn();
        world.insert(o, StaticObjectComponent::new(mesh.index(), 1, material.index(), 1, m, [0.0, 0.0, 0.0, 1.0], 6, 0, 0, 0, 0, 0));
    }

    #[test]
    fn every_class_has_valid_tables() {
        for class in AcousticClass::ALL {
            for v in class.absorption().into_iter().chain(class.transmission()) {
                assert!((0.0..=1.0).contains(&v), "{class:?}: {v}");
            }
            // The tracer treats a surface as a mirror with reflection sqrt(1 - alpha); an
            // absorption of exactly 1 would kill every path (and none of ours is that high).
            assert!(class.absorption().iter().all(|&a| a < 0.5), "{class:?}");
            assert_eq!(class.scattering(), [0.0; 8]);
        }
        // Heavy stone transmits almost nothing; glass a little at low frequencies.
        assert!(AcousticClass::Stone.transmission()[0] <= 0.025);
        assert!(AcousticClass::StainedGlass.transmission()[0] > AcousticClass::StainedGlass.transmission()[3]);
        assert!(AcousticClass::StainedGlass.transmission()[0] > AcousticClass::Stone.transmission()[0]);
    }

    #[test]
    fn extraction_applies_transforms_tags_and_counts_fallbacks() {
        let mut world = World::new();
        let mesh = quad_mesh(&mut world);
        let tagged = world.spawn();
        world.insert(tagged, MaterialComponent::new([0.5, 0.5, 0.5, 1.0], 0.5, 0.0, [0.0; 3], 0.0));
        world.insert(tagged, AcousticSurface(AcousticClass::Oak));
        // Untagged metallic material: inferred bronze, counted as a fallback.
        let untagged = world.spawn();
        world.insert(untagged, MaterialComponent::new([0.4, 0.3, 0.1, 1.0], 0.2, 0.9, [0.0; 3], 0.0));
        add_object(&mut world, mesh, tagged, glam::Mat4::IDENTITY);
        add_object(&mut world, mesh, untagged, glam::Mat4::from_translation(Vec3::new(10.0, 2.0, -3.0)));
        // Object whose mesh row does not exist is reported, not silently dropped.
        let stale = world.spawn();
        world.insert(stale, StaticObjectComponent::new(9999, 1, tagged.index(), 1, glam::Mat4::IDENTITY, [0.0; 4], 6, 0, 0, 0, 0, 0));

        let mut handles = HashMap::new();
        let ex = extract_acoustic_scene(&world, |c| {
            let n = handles.len() as u32;
            *handles.entry(c).or_insert(n)
        });
        let s = &ex.stats;
        assert_eq!(s.object_rows, 3);
        assert_eq!(s.instances, 2);
        assert_eq!(s.skipped.len(), 1, "{:?}", s.skipped);
        assert_eq!(s.unique_meshes, 1);
        assert_eq!(s.triangles, 4);
        assert_eq!(s.tagged_materials, 1);
        assert_eq!(s.fallback_materials, 1);
        assert_eq!(s.transformed_instances, 1);
        assert_eq!(ex.classes, vec![AcousticClass::Oak, AcousticClass::Bronze]);
        assert_eq!(s.aabb_min, [0.0, 0.0, -3.0]);
        assert_eq!(s.aabb_max, [11.0, 2.0, 1.0]);
        // Column-major translation lives in elements 12..15.
        let m = &ex.scene.meshes[1];
        assert_eq!(&m.transform[12..15], &[10.0, 2.0, -3.0]);
        assert_eq!(ex.scene.total_triangle_count(), 4);
        assert!((s.by_class[&AcousticClass::Oak].area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inference_rules() {
        let glass = MaterialComponent::from_surface([0.2, 0.3, 0.8], 0.6, 0.1, 0.0, [0.0; 3], 0.0);
        assert_eq!(infer_class(&glass), AcousticClass::StainedGlass);
        let metal = MaterialComponent::new([1.0; 4], 0.3, 1.0, [0.0; 3], 0.0);
        assert_eq!(infer_class(&metal), AcousticClass::Bronze);
        let flame = MaterialComponent::new([1.0; 4], 0.5, 0.0, [1.0, 0.4, 0.05], 5.0);
        assert_eq!(infer_class(&flame), AcousticClass::Flame);
        let plain = MaterialComponent::new([0.5; 4], 0.8, 0.0, [0.0; 3], 0.0);
        assert_eq!(infer_class(&plain), AcousticClass::Stone);
    }
}
