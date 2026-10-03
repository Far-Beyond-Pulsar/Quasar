//! Shared fixtures of the ray-budget / closed-room tests: a closed hall with columns.
#![allow(dead_code)]

use quasar_core::backend::MaterialProvider;
use quasar_core::bands::Band8;
use quasar_core::rays::RayInteractionContext;
use quasar_core::scene::{AcousticMesh, AcousticScene};

/// Opaque walls (transmission 0, the demo's setting), absorption 0.2.
pub struct Opaque;
impl MaterialProvider for Opaque {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(0.2)
    }
    fn evaluate_transmission(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::zeros()
    }
}

/// Walls that let a little through (not opaque): the shortcuts must not apply.
pub struct Leaky;
impl MaterialProvider for Leaky {
    fn evaluate_material(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(0.2)
    }
    fn evaluate_transmission(&self, _h: u32, _c: &RayInteractionContext) -> Band8 {
        Band8::splat(0.05)
    }
}

/// Consistently wound closed box (outward when `inward` is false, else the room shell).
pub fn box_mesh(id: u64, lo: [f32; 3], hi: [f32; 3], inward: bool) -> AcousticMesh {
    let p = vec![
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ];
    let mut idx: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    ];
    if inward {
        for t in idx.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
    }
    AcousticMesh::new(id, p, idx, 0)
}

/// 20 x 12 x 30 m hall, four 1 m columns.
pub fn hall() -> AcousticScene {
    let mut s = AcousticScene::new();
    s.add_mesh(box_mesh(1, [0.0, 0.0, 0.0], [20.0, 12.0, 30.0], true));
    for (i, (x, z)) in [(6.0, 9.0), (14.0, 9.0), (6.0, 21.0), (14.0, 21.0)].into_iter().enumerate() {
        s.add_mesh(box_mesh(10 + i as u64, [x - 0.5, 0.0, z - 0.5], [x + 0.5, 12.0, z + 0.5], false));
    }
    s
}

pub const SRC: [f32; 3] = [10.0, 1.7, 5.0];
pub const INSIDE: [f32; 3] = [10.0, 1.7, 25.0];
