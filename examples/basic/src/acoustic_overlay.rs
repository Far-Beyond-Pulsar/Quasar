//! Uncapped GPU line overlay for captured acoustic ray tests and reflection paths.
use quasar_backends::debug_capture::AcousticDebugFrame;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

pub struct AcousticOverlay {
    pipeline: wgpu::RenderPipeline,
    camera: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    capacity: usize,
    count: u32,
}

impl AcousticOverlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Acoustic ray overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("acoustic_overlay.wgsl").into()),
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Acoustic overlay camera"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Acoustic overlay layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Acoustic overlay camera"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Acoustic overlay pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Acoustic overlay lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Acoustic overlay vertices"),
            contents: &[0; 4],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            pipeline,
            camera,
            bindings,
            vertices,
            capacity: 4,
            count: 0,
        }
    }

    /// Upload one spatial snapshot. Capacity grows to fit every captured segment.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &AcousticDebugFrame,
    ) {
        let mut vertices = Vec::new();
        let mut line = |a: [f32; 3], b: [f32; 3], color: [f32; 4]| {
            vertices.push(Vertex { position: a, color });
            vertices.push(Vertex { position: b, color });
        };
        for sample in &frame.rays {
            let ray = &sample.ray;
            let end = sample.hit.as_ref().map(|h| h.point).unwrap_or_else(|| {
                ray.point_at(if ray.max_distance < 1.0e6 {
                    ray.max_distance
                } else {
                    60.0
                })
            });
            line(
                ray.point_at(ray.min_distance),
                end,
                if sample.hit.is_some() {
                    [1.0, 0.18, 0.12, 0.16]
                } else {
                    [0.1, 0.65, 1.0, 0.08]
                },
            );
            if let Some(hit) = &sample.hit {
                let p = glam::Vec3::from_array(hit.point);
                line(
                    hit.point,
                    (p + glam::Vec3::from_array(hit.normal) * 0.12).to_array(),
                    [1.0, 0.25, 0.15, 0.3],
                );
            }
        }
        // Selected paths come last and remain visible over candidate and visibility rays.
        for selected in [false, true] {
            for path in frame.paths.iter().filter(|p| p.selected == selected) {
                let color = if selected {
                    [0.2, 1.0, 0.3, 0.85]
                } else {
                    [0.7, 0.4, 1.0, 0.15]
                };
                let mut from = path.source;
                for &bounce in &path.bounces {
                    line(from, bounce, color);
                    from = bounce;
                }
                line(from, path.listener, color);
                for (&point, &normal) in path.bounces.iter().zip(&path.normals) {
                    let p = glam::Vec3::from_array(point);
                    let mark = if selected {
                        [1.0, 0.85, 0.05, 1.0]
                    } else {
                        color
                    };
                    for axis in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z] {
                        line(
                            (p - axis * 0.07).to_array(),
                            (p + axis * 0.07).to_array(),
                            mark,
                        );
                    }
                    line(
                        point,
                        (p + glam::Vec3::from_array(normal) * 0.35).to_array(),
                        mark,
                    );
                }
            }
        }
        self.count = vertices.len().try_into().expect("acoustic vertex count");
        let bytes = bytemuck::cast_slice(&vertices);
        if bytes.len() > self.capacity {
            self.capacity = bytes.len().next_power_of_two();
            self.vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Acoustic overlay vertices"),
                size: self.capacity as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.vertices, 0, bytes);
        }
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &helio::Camera,
        target: &wgpu::TextureView,
    ) {
        if self.count == 0 {
            return;
        }
        queue.write_buffer(
            &self.camera,
            0,
            bytemuck::cast_slice(&(camera.proj * camera.view).to_cols_array()),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Acoustic trace overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..self.count, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quasar_backends::debug_capture::DebugRay;
    use quasar_core::rays::Ray;

    #[test]
    fn renders_more_than_helio_debug_limit_and_clears() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let mut overlay = AcousticOverlay::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let ray = DebugRay {
            ray: Ray {
                origin: [-1.0, 0.0, -3.0],
                direction: [1.0, 0.0, 0.0],
                min_distance: 0.0,
                max_distance: 2.0,
            },
            hit: None,
        };
        overlay.update(
            &device,
            &queue,
            &AcousticDebugFrame {
                rays: vec![ray; 40_000],
                paths: vec![],
            },
        );
        assert_eq!(overlay.count, 80_000);
        assert!(overlay.capacity >= 80_000 * std::mem::size_of::<Vertex>());
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay check"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let camera = helio::Camera::perspective_look_at(
            glam::Vec3::ZERO,
            -glam::Vec3::Z,
            glam::Vec3::Y,
            60f32.to_radians(),
            1.0,
            0.1,
            100.0,
        );
        overlay.render(
            &device,
            &queue,
            &camera,
            &texture.create_view(&Default::default()),
        );
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        overlay.update(&device, &queue, &AcousticDebugFrame::default());
        assert_eq!(overlay.count, 0);
    }
}
