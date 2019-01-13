use gfx_backend_vulkan as backend;
use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::device::Device;
use gfx_hal::format::ChannelType;
use gfx_hal::format::Swizzle;
use gfx_hal::pso::DescriptorPool;
use gfx_hal::window::Swapchain;
use gfx_hal::window::{Extent2D, Surface};
use gfx_hal::Backbuffer;
use gfx_hal::SwapchainConfig;
use gfx_hal::{
    buffer as b, command, format as f, image as i, memory as m, pass as p, pool, pso, queue as q,
    FrameSync, Graphics, Instance, MemoryType, Primitive,
};
use log::info;
use nalgebra::Matrix4;
use nalgebra::Point3;
use nalgebra::Vector3;
use num_traits::identities::One;
use std::error::Error;
use std::io::Read;
use std::mem;

type M4 = Matrix4<f32>;
type V3 = Vector3<f32>;
type P3 = Point3<f32>;
type Vertex = ([f32; 4], [f32; 4]);

#[cfg_attr(rustfmt, rustfmt_skip)]
const DIMS: Extent2D = Extent2D { width: 512, height: 512 };

const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

const DEPTH_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::DEPTH,
    levels: 0..1,
    layers: 0..1,
};

#[cfg_attr(rustfmt, rustfmt_skip)]
const CUBE_FACES_COLOR : &[Vertex] = &[
    // red face
    ([-1.0, -1.0,  1.0,  1.0], [1.0, 0.0, 0.0, 1.0]),
    ([-1.0,  1.0,  1.0,  1.0], [1.0, 0.0, 0.0, 1.0]),
    ([ 1.0, -1.0,  1.0,  1.0], [1.0, 0.0, 0.0, 1.0]),
    ([ 1.0, -1.0,  1.0,  1.0], [1.0, 0.0, 0.0, 1.0]),
    ([-1.0,  1.0,  1.0,  1.0], [1.0, 0.0, 0.0, 1.0]),
    ([ 1.0,  1.0,  1.0,  1.0], [1.0, 0.0, 0.0, 1.0]),

    // green face
    ([-1.0, -1.0, -1.0,  1.0], [0.0, 1.0, 0.0, 1.0]),
    ([ 1.0, -1.0, -1.0,  1.0], [0.0, 1.0, 0.0, 1.0]),
    ([-1.0,  1.0, -1.0,  1.0], [0.0, 1.0, 0.0, 1.0]),
    ([-1.0,  1.0, -1.0,  1.0], [0.0, 1.0, 0.0, 1.0]),
    ([ 1.0, -1.0, -1.0,  1.0], [0.0, 1.0, 0.0, 1.0]),
    ([ 1.0,  1.0, -1.0,  1.0], [0.0, 1.0, 0.0, 1.0]),

    // blue face
    ([-1.0,  1.0,  1.0,  1.0], [0.0, 0.0, 1.0, 1.0]),
    ([-1.0, -1.0,  1.0,  1.0], [0.0, 0.0, 1.0, 1.0]),
    ([-1.0,  1.0, -1.0,  1.0], [0.0, 0.0, 1.0, 1.0]),
    ([-1.0,  1.0, -1.0,  1.0], [0.0, 0.0, 1.0, 1.0]),
    ([-1.0, -1.0,  1.0,  1.0], [0.0, 0.0, 1.0, 1.0]),
    ([-1.0, -1.0, -1.0,  1.0], [0.0, 0.0, 1.0, 1.0]),

    // yellow face
    ([ 1.0,  1.0,  1.0,  1.0], [1.0, 1.0, 0.0, 1.0]),
    ([ 1.0,  1.0, -1.0,  1.0], [1.0, 1.0, 0.0, 1.0]),
    ([ 1.0, -1.0,  1.0,  1.0], [1.0, 1.0, 0.0, 1.0]),
    ([ 1.0, -1.0,  1.0,  1.0], [1.0, 1.0, 0.0, 1.0]),
    ([ 1.0,  1.0, -1.0,  1.0], [1.0, 1.0, 0.0, 1.0]),
    ([ 1.0, -1.0, -1.0,  1.0], [1.0, 1.0, 0.0, 1.0]),

    // magenta face
    ([ 1.0,  1.0,  1.0,  1.0], [1.0, 0.0, 1.0, 1.0]),
    ([-1.0,  1.0,  1.0,  1.0], [1.0, 0.0, 1.0, 1.0]),
    ([ 1.0,  1.0, -1.0,  1.0], [1.0, 0.0, 1.0, 1.0]),
    ([ 1.0,  1.0, -1.0,  1.0], [1.0, 0.0, 1.0, 1.0]),
    ([-1.0,  1.0,  1.0,  1.0], [1.0, 0.0, 1.0, 1.0]),
    ([-1.0,  1.0, -1.0,  1.0], [1.0, 0.0, 1.0, 1.0]),

    // cyan face
    ([ 1.0, -1.0,  1.0,  1.0], [0.0, 1.0, 1.0, 1.0]),
    ([ 1.0, -1.0, -1.0,  1.0], [0.0, 1.0, 1.0, 1.0]),
    ([-1.0, -1.0,  1.0,  1.0], [0.0, 1.0, 1.0, 1.0]),
    ([-1.0, -1.0,  1.0,  1.0], [0.0, 1.0, 1.0, 1.0]),
    ([ 1.0, -1.0, -1.0,  1.0], [0.0, 1.0, 1.0, 1.0]),
    ([-1.0, -1.0, -1.0,  1.0], [0.0, 1.0, 1.0, 1.0]),
];

fn main() -> Result<(), Box<Error>> {
    env_logger::init();

    let (mut events_loop, window) = build_window("Basic Window", DIMS)?;

    let instance = backend::Instance::create("vulkansamples_instance", 1);
    let mut surface = instance.create_surface(&window);

    let mut adapters = instance.enumerate_adapters();

    for adapter in &adapters {
        info!("{:?}", adapter.info);
    }

    let adapter = adapters.remove(0);
    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let limits = adapter.physical_device.limits();

    info!("{:?}", memory_types);
    info!("{:?}", limits);

    let (device, mut queue_group) = adapter
        .open_with::<_, Graphics>(1, |family| surface.supports_queue_family(family))
        .map_err(debug_error)?;

    let mut command_pool = unsafe {
        device.create_command_pool_typed(&queue_group, pool::CommandPoolCreateFlags::empty())
    }
    .map_err(debug_error)?;

    let (caps, formats, _present_modes, _compisite_alphas) =
        surface.compatibility(&adapter.physical_device);

    info!("{:?}", formats);

    let format = formats.map_or(f::Format::Rgb8Unorm, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == ChannelType::Unorm)
            .cloned()
            .unwrap_or(formats[0])
    });

    let swapchain_config = SwapchainConfig::from_caps(&caps, format, DIMS);
    let extent = swapchain_config.extent.to_extent();

    info!("{:?}", swapchain_config);

    let (mut swapchain, backbuffer) =
        unsafe { device.create_swapchain(&mut surface, swapchain_config, None) }
            .map_err(debug_error)?;

    let (depth_image, depth_srv, depth_memory) =
        depth_image::<backend::Backend>(&device, &memory_types)?;

    let mvp = mvp();

    let (uniform_buffer, uniform_memory) =
        uniform_buffer::<backend::Backend>(&device, &memory_types, mvp)?;

    let descriptor_layout = unsafe {
        device.create_descriptor_set_layout(
            &[pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            }],
            &[],
        )
    }
    .map_err(debug_error)?;

    let pipeline_layout = unsafe { device.create_pipeline_layout(Some(&descriptor_layout), &[]) }
        .map_err(debug_error)?;

    let mut descriptor_pool = unsafe {
        device.create_descriptor_pool(
            1,
            Some(&pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
            }),
        )
    }
    .map_err(debug_error)?;

    let descriptor_set =
        unsafe { descriptor_pool.allocate_set(&descriptor_layout) }.map_err(debug_error)?;

    unsafe {
        device.write_descriptor_sets(Some(pso::DescriptorSetWrite {
            set: &descriptor_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(&pso::Descriptor::Buffer(&uniform_buffer, None..None)),
        }))
    }

    let attachments = [
        p::Attachment {
            format: Some(format),
            samples: 1,
            ops: p::AttachmentOps {
                load: p::AttachmentLoadOp::Clear,
                store: p::AttachmentStoreOp::Store,
            },
            stencil_ops: p::AttachmentOps {
                load: p::AttachmentLoadOp::DontCare,
                store: p::AttachmentStoreOp::DontCare,
            },
            layouts: i::Layout::Undefined..i::Layout::Present,
        },
        p::Attachment {
            format: Some(f::Format::D16Unorm),
            samples: 1,
            ops: p::AttachmentOps {
                load: p::AttachmentLoadOp::Clear,
                store: p::AttachmentStoreOp::DontCare,
            },
            stencil_ops: p::AttachmentOps {
                load: p::AttachmentLoadOp::DontCare,
                store: p::AttachmentStoreOp::DontCare,
            },
            layouts: i::Layout::Undefined..i::Layout::DepthStencilAttachmentOptimal,
        },
    ];

    let subpass = p::SubpassDesc {
        colors: &[(0, i::Layout::ColorAttachmentOptimal)],
        depth_stencil: Some(&(1, i::Layout::DepthStencilAttachmentOptimal)),
        inputs: &[],
        resolves: &[],
        preserves: &[],
    };

    let rp_info = unsafe { device.create_render_pass(&attachments, Some(&subpass), &[]) }
        .map_err(debug_error)?;

    let mut vtx_spv = Vec::new();
    glsl_to_spirv::compile(
        include_str!("../data/basic.vert"),
        glsl_to_spirv::ShaderType::Vertex,
    )?
    .read_to_end(&mut vtx_spv)?;

    let vtx = unsafe { device.create_shader_module(&vtx_spv) }.map_err(debug_error)?;

    let mut frag_spv = Vec::new();
    glsl_to_spirv::compile(
        include_str!("../data/basic.frag"),
        glsl_to_spirv::ShaderType::Fragment,
    )?
    .read_to_end(&mut frag_spv)?;

    let frag = unsafe { device.create_shader_module(&frag_spv) }.map_err(debug_error)?;

    let (frame_images, framebuffers) = match backbuffer {
        Backbuffer::Images(images) => {
            let pairs: Vec<_> = images
                .into_iter()
                .map(|image| unsafe {
                    let rtv = device.create_image_view(
                        &image,
                        i::ViewKind::D2,
                        format,
                        Swizzle::NO,
                        COLOR_RANGE,
                    )?;

                    Ok((image, rtv))
                })
                .collect::<Result<_, i::ViewError>>()
                .map_err(debug_error)?;

            let fbos = pairs
                .iter()
                .map(|(_, rtv)| unsafe {
                    device.create_framebuffer(&rp_info, vec![rtv, &depth_srv], extent)
                })
                .collect::<Result<_, _>>()
                .map_err(debug_error)?;

            (pairs, fbos)
        }
        Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
    };

    let (vertex_buffer, vertex_memory) =
        vertex_buffer::<backend::Backend>(&device, &memory_types, CUBE_FACES_COLOR)?;

    let desc = pso::GraphicsPipelineDesc {
        shaders: pso::GraphicsShaderSet {
            vertex: pso::EntryPoint {
                entry: "main",
                module: &vtx,
                specialization: pso::Specialization::default(),
            },
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(pso::EntryPoint {
                entry: "main",
                module: &frag,
                specialization: pso::Specialization {
                    constants: &[],
                    data: &[],
                },
            }),
        },
        rasterizer: pso::Rasterizer {
            polygon_mode: pso::PolygonMode::Fill,
            cull_face: pso::Face::BACK,
            front_face: pso::FrontFace::Clockwise,
            depth_clamping: false,
            depth_bias: None,
            conservative: false,
        },
        vertex_buffers: vec![pso::VertexBufferDesc {
            binding: 0,
            stride: mem::size_of::<Vertex>() as u32,
            rate: 0,
        }],
        attributes: vec![
            pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: pso::Element {
                    format: f::Format::Rgba32Float,
                    offset: 0,
                },
            },
            pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: pso::Element {
                    format: f::Format::Rgba32Float,
                    offset: 16,
                },
            },
        ],
        input_assembler: pso::InputAssemblerDesc {
            primitive: Primitive::TriangleList,
            primitive_restart: pso::PrimitiveRestart::Disabled,
        },
        blender: pso::BlendDesc {
            logic_op: None,
            targets: vec![pso::ColorBlendDesc::EMPTY],
        },
        depth_stencil: pso::DepthStencilDesc {
            depth: pso::DepthTest::Off,
            depth_bounds: false,
            stencil: pso::StencilTest::Off,
        },
        multisampling: None,
        baked_states: pso::BakedStates {
            viewport: None,
            scissor: None,
            blend_color: None,
            depth_bounds: None,
        },
        layout: &pipeline_layout,
        subpass: p::Subpass {
            index: 0,
            main_pass: &rp_info,
        },
        flags: pso::PipelineCreationFlags::empty(),
        parent: pso::BasePipeline::None,
    };

    let pipeline = unsafe { device.create_graphics_pipeline(&desc, None) }.map_err(debug_error)?;

    let image_aquire_semaphore = device.create_semaphore().map_err(debug_error)?;

    let image_index = unsafe {
        swapchain.acquire_image(std::u64::MAX, FrameSync::Semaphore(&image_aquire_semaphore))
    }
    .map_err(debug_error)?;

    let mut cmd = command_pool.acquire_command_buffer::<command::OneShot>();

    let draw_fence = device.create_fence(false).map_err(debug_error)?;

    let rect = pso::Rect {
        x: 0,
        y: 0,
        w: DIMS.width as i16,
        h: DIMS.height as i16,
    };

    unsafe {
        cmd.begin();

        cmd.bind_graphics_pipeline(&pipeline);

        cmd.bind_graphics_descriptor_sets(&pipeline_layout, 0, Some(&descriptor_set), &[]);

        cmd.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));

        cmd.set_viewports(
            0,
            &[pso::Viewport {
                rect,
                depth: 0.0..1.0,
            }],
        );

        cmd.set_scissors(0, &[rect]);

        {
            let mut encoder = cmd.begin_render_pass_inline(
                &rp_info,
                &framebuffers[image_index as usize],
                rect,
                &[
                    command::ClearValue::Color(command::ClearColor::Float([0.2; 4])),
                    command::ClearValue::DepthStencil(command::ClearDepthStencil(1.0, 0)),
                ],
            );

            encoder.draw(0..12 * 3, 0..1);
        }

        cmd.finish();

        queue_group.queues[0].submit(
            q::Submission {
                command_buffers: Some(&cmd),
                wait_semaphores: Some((
                    &image_aquire_semaphore,
                    pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                )),
                signal_semaphores: &[],
            },
            Some(&draw_fence),
        );

        loop {
            if device
                .wait_for_fence(&draw_fence, !0)
                .map_err(debug_error)?
            {
                break;
            }
        }

        swapchain
            .present_nosemaphores(&mut queue_group.queues[0], image_index)
            .map_err(|_| "failed to present")?;
    }

    run(&mut events_loop);

    device.wait_idle().map_err(debug_error)?;

    unsafe {
        device.destroy_fence(draw_fence);
        device.destroy_graphics_pipeline(pipeline);
        device.destroy_semaphore(image_aquire_semaphore);
        device.destroy_shader_module(vtx);
        device.destroy_shader_module(frag);
        device.destroy_render_pass(rp_info);
        device.destroy_descriptor_pool(descriptor_pool);
        device.destroy_pipeline_layout(pipeline_layout);
        device.destroy_descriptor_set_layout(descriptor_layout);
        device.free_memory(depth_memory);
        device.free_memory(uniform_memory);
        device.free_memory(vertex_memory);
        device.destroy_buffer(uniform_buffer);
        device.destroy_buffer(vertex_buffer);
        device.destroy_image_view(depth_srv);
        device.destroy_image(depth_image);
        device.destroy_swapchain(swapchain);
        device.destroy_command_pool(command_pool.into_raw());

        for (_, image_view) in frame_images {
            device.destroy_image_view(image_view);
        }

        for fb in framebuffers {
            device.destroy_framebuffer(fb);
        }
    }

    Ok(())
}

fn build_window(
    title: &str,
    dims: Extent2D,
) -> Result<(winit::EventsLoop, winit::Window), winit::CreationError> {
    use winit::dpi::PhysicalSize;
    use winit::{EventsLoop, WindowBuilder};

    let events_loop = EventsLoop::new();

    let size: PhysicalSize = (dims.width, dims.height).into();

    let factor = events_loop.get_primary_monitor().get_hidpi_factor();

    let window = WindowBuilder::new()
        .with_title(title)
        .with_dimensions(size.to_logical(factor))
        .build(&events_loop)?;

    Ok((events_loop, window))
}

fn depth_image<B: gfx_hal::Backend>(
    device: &B::Device,
    memory_types: &[MemoryType],
) -> Result<(B::Image, B::ImageView, B::Memory), Box<Error>> {
    let mut depth_image = unsafe {
        device.create_image(
            i::Kind::D2(DIMS.width, DIMS.height, 1, 1),
            1,
            f::Format::D16Unorm,
            i::Tiling::Optimal,
            i::Usage::DEPTH_STENCIL_ATTACHMENT,
            i::ViewCapabilities::empty(),
        )
    }
    .map_err(debug_error)?;

    let depth_req = unsafe { device.get_image_requirements(&depth_image) };

    info!("{:?}", depth_req);

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            depth_req.type_mask & (1 << id) != 0
                && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
        })
        .ok_or("No valid memory type found")?
        .into();

    let depth_memory =
        unsafe { device.allocate_memory(device_type, depth_req.size) }.map_err(debug_error)?;

    unsafe { device.bind_image_memory(&depth_memory, 0, &mut depth_image) }.map_err(debug_error)?;

    let depth_srv = unsafe {
        device.create_image_view(
            &depth_image,
            i::ViewKind::D2,
            f::Format::D16Unorm,
            Swizzle::NO,
            DEPTH_RANGE,
        )
    }
    .map_err(debug_error)?;

    Ok((depth_image, depth_srv, depth_memory))
}

fn uniform_buffer<B: gfx_hal::Backend>(
    device: &B::Device,
    memory_types: &[MemoryType],
    mvp: M4,
) -> Result<(B::Buffer, B::Memory), Box<Error>> {
    let mut buffer =
        unsafe { device.create_buffer(mem::size_of::<M4>() as u64, b::Usage::UNIFORM) }
            .map_err(debug_error)?;

    let req = unsafe { device.get_buffer_requirements(&buffer) };

    info!("{:?}", req);

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            req.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(m::Properties::CPU_VISIBLE | m::Properties::COHERENT)
        })
        .ok_or("No valid memory type found")?
        .into();

    let buffer_memory =
        unsafe { device.allocate_memory(device_type, req.size) }.map_err(debug_error)?;

    unsafe {
        #[allow(clippy::cast_ptr_alignment)]
        let data = device
            .map_memory(&buffer_memory, 0..req.size)
            .map_err(debug_error)? as *mut M4;

        *data = mvp;

        device.unmap_memory(&buffer_memory);

        device
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .map_err(debug_error)?;
    }

    Ok((buffer, buffer_memory))
}

fn vertex_buffer<B: gfx_hal::Backend>(
    device: &B::Device,
    memory_types: &[MemoryType],
    vtx: &[Vertex],
) -> Result<(B::Buffer, B::Memory), Box<Error>> {
    let mut buffer = unsafe {
        device.create_buffer(
            (mem::size_of::<Vertex>() * vtx.len()) as u64,
            b::Usage::VERTEX,
        )
    }
    .map_err(debug_error)?;

    let req = unsafe { device.get_buffer_requirements(&buffer) };

    info!("{:?}", req);

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            req.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(m::Properties::CPU_VISIBLE | m::Properties::COHERENT)
        })
        .ok_or("No valid memory type found")?
        .into();

    let buffer_memory =
        unsafe { device.allocate_memory(device_type, req.size) }.map_err(debug_error)?;

    unsafe {
        #[allow(clippy::cast_ptr_alignment)]
        let data = device
            .map_memory(&buffer_memory, 0..req.size)
            .map_err(debug_error)? as *mut Vertex;

        std::ptr::copy_nonoverlapping(vtx.as_ptr(), data, vtx.len());

        device.unmap_memory(&buffer_memory);

        device
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .map_err(debug_error)?;
    }

    Ok((buffer, buffer_memory))
}

fn mvp() -> M4 {
    let projection = M4::new_perspective(f32::to_radians(45.0), 1.0, 0.1, 100.0);
    let view = M4::look_at_rh(
        &P3::new(-5.0, 3.0, -10.0),
        &P3::origin(),
        &V3::new(0.0, -1.0, 0.0),
    );
    let model = M4::one();

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let clip = M4::from_row_slice(&[
        1.0,  0.0,  0.0,  0.0,
        0.0, -1.0,  0.0,  0.0,
        0.0,  0.0,  0.5,  0.0,
        0.0,  0.0,  0.5,  1.0,
    ]);

    clip * projection * view * model
}

fn run(events_loop: &mut winit::EventsLoop) {
    let mut running = true;
    loop {
        events_loop.poll_events(|event| {
            use winit::{Event, WindowEvent};

            // info!("{:?}", event);

            if let Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } = event
            {
                running = false;
            }
        });

        if !running {
            break;
        }
    }
}

fn debug_error(error: impl std::fmt::Debug) -> String {
    format!("{:?}", error)
}
