//! Interactive-path helpers carried over from Helio's `hlfs_capture.rs`
//! (the offscreen capture runner was dropped: it is not needed here).
use helio::{Camera, Renderer};
use pulsar_scenedb::{SceneDb, World};

/// The draw-range table is read back asynchronously. Prime it before the
/// first presented/captured frame so the initial image contains the scene.
pub fn warm_up_cathedral(
    scene_db: &SceneDb,
    renderer: &mut Renderer,
    acceleration: Option<&helio_pass_hlfs::SceneDbRayTracing>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera: &Camera,
    target: &wgpu::TextureView,
) {
    for _ in 0..4 {
        crate::v3_demo_common::flush_scene_db(scene_db, queue);
        if let Some(acceleration) = acceleration {
            renderer.set_ray_tracing_frame_with_transmission(
                acceleration.tlas(), acceleration.transmission());
        }
        renderer.render(camera, target).expect("cathedral warmup frame");
        device.poll(wgpu::PollType::wait_indefinitely()).expect("cathedral warmup poll");
    }
}

/// Include opaque geometry and explicitly authored thin-sheet RT materials.
pub fn enable_ray_shadows(world: &mut World) {
    let ids: Vec<_> = world
        .query::<(&helio_pass_forward_lit::LightComponent,)>()
        .map(|(id, _)| id)
        .collect();
    for id in ids {
        let mut component = world
            .get_mut::<helio_pass_forward_lit::LightComponent>(id)
            .unwrap();
        let mut light: helio_pass_forward_lit::GpuLight = (*component).into();
        light.set_ray_traced_shadows(std::env::var_os("HLFS_UNSHADOWED").is_none());
        *component = light.into();
    }
    let ids: Vec<_> = world
        .query::<(&helio_pass_gbuffer::StaticObjectComponent,)>()
        .map(|(id, _)| id)
        .collect();
    for id in ids {
        let object = world
            .get::<helio_pass_gbuffer::StaticObjectComponent>(id)
            .unwrap();
        let transparent = world
            .query::<(&helio_pass_gbuffer::MaterialComponent,)>()
            .any(|(entity, (material,))| {
                entity.index() == object.material_slot
                    && material.flags & helio_mats::FLAG_ALPHA_BLEND != 0
                    && world.get::<helio_pass_hlfs::RayTransmission>(entity).is_none()
            });
        // Display alpha alone is not a transmission model.
        if !transparent {
            world
                .get_mut::<helio_pass_gbuffer::StaticObjectComponent>(id)
                .unwrap()
                .flags |= helio_pass_object_batch::INSTANCE_FLAG_CASTS_SHADOW;
        }
    }
}

