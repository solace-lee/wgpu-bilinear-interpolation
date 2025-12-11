use std::borrow::Cow;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

// 嵌入 WGSL Shader 代码 (保持不变)
const SHADER_CODE: &str = r#"
struct Uniforms {
  imageSize: vec2<f32>,
  slope: f32,
  intercept: f32,
  ww: f32,
  wl: f32,
  invert: i32,
  pad1: f32, 
  transformMatrix: mat4x4<f32>,
};

struct ColorMap {
  data: array<f32, 768>,
};

@group(0) @binding(0) var<uniform> u_uniforms: Uniforms;
@group(0) @binding(1) var u_imageData: texture_2d<f32>; 
@group(0) @binding(2) var<storage, read> u_colorMap: ColorMap;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uvw: vec2<f32>,
};

@vertex
fn vs_main(@location(0) a_position: vec2<f32>) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(a_position, 0.0, 1.0);
  let uv = (a_position + 1.0) * 0.5;
  let pos = vec2<f32>(uv.x * u_uniforms.imageSize.x, (1.0 - uv.y) * u_uniforms.imageSize.y);
  let transformed = u_uniforms.transformMatrix * vec4<f32>(pos, 1.0, 1.0);
  out.uvw = transformed.xy;
  return out;
}

fn getValue(texCoord: vec2<f32>) -> f32 {
  let iTexCoord = vec2<i32>(floor(texCoord));
  let fTexCoord = fract(texCoord);
  
  let r00 = textureLoad(u_imageData, iTexCoord, 0).r;
  let r10 = textureLoad(u_imageData, iTexCoord + vec2<i32>(1, 0), 0).r;
  let r01 = textureLoad(u_imageData, iTexCoord + vec2<i32>(0, 1), 0).r;
  let r11 = textureLoad(u_imageData, iTexCoord + vec2<i32>(1, 1), 0).r;

  let c00 = r00 * u_uniforms.slope + u_uniforms.intercept;
  let c10 = r10 * u_uniforms.slope + u_uniforms.intercept;
  let c01 = r01 * u_uniforms.slope + u_uniforms.intercept;
  let c11 = r11 * u_uniforms.slope + u_uniforms.intercept;

  let ct = mix(mix(c00, c10, fTexCoord.x), mix(c01, c11, fTexCoord.x), fTexCoord.y);
  return ct;
}

@fragment
fn fs_main(@location(0) uvw: vec2<f32>) -> @location(0) vec4<f32> {
  let texCoord = uvw;
  let ct = getValue(texCoord);
  
  let width = u_uniforms.ww;
  let center = u_uniforms.wl;
  
  let minCt = center - width / 2.0;
  let maxCt = center + width / 2.0;
  
  var color: f32 = 0.0;
  
  if (ct < minCt) {
    color = 0.0;
  } else if (ct > maxCt) {
    color = 1.0;
  } else {
    color = (ct - minCt) / width;
  }
  
  if (u_uniforms.invert > 0) {
    color = 1.0 - color;
  }
  
  return vec4<f32>(color, color, color, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    image_size: [f32; 2],
    slope: f32,
    intercept: f32,
    ww: f32,
    wl: f32,
    invert: i32,
    pad1: f32,
    transform_matrix: [[f32; 4]; 4],
}

#[wasm_bindgen]
pub struct OffscreenImageRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    
    uniform_buffer: wgpu::Buffer,
    color_map_buffer: wgpu::Buffer,
    input_texture: Option<wgpu::Texture>,
}

#[wasm_bindgen]
impl OffscreenImageRenderer {
    pub async fn new() -> Result<OffscreenImageRenderer, JsValue> {
        // --- API 变更 1: Instance 创建 ---
        // v24 需要 InstanceDescriptor
        let instance = wgpu::Instance::new(&&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), 
            flags: wgpu::InstanceFlags::default(),
            backend_options: Default::default(),
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| JsValue::from_str("No appropriate GPUAdapter found."))?;

        // --- API 变更 2: 解决 Limits 报错 ---
        // 直接使用 adapter 支持的 limits，避免手动设置不被识别的旧参数
        let required_limits = adapter.limits();

        // --- API 变更 3: DeviceDescriptor 增加 memory_hints ---
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WebGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance, // v24 新增
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;

        // 编译 Shader
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_CODE)),
        });

        // 绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("双线性插值绑定组"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("双线性插值渲染管线"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"), // v24 变更：Option<&str>
                compilation_options: wgpu::PipelineCompilationOptions::default(), // v24 新增字段
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 2 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"), // v24 变更：Option<&str>
                compilation_options: wgpu::PipelineCompilationOptions::default(), // v24 新增字段
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None, // v24 新增字段
        });

        let vertices: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_map_size = 768 * 4;
        let color_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ColorMap Buffer"),
            size: color_map_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            vertex_buffer,
            uniform_buffer,
            color_map_buffer,
            input_texture: None,
        })
    }

    pub async fn render(
        &mut self,
        img_width: u32,
        img_height: u32,
        img_data: &[f32],
        slope: f32,
        intercept: f32,
        canvas_width: f32,
        canvas_height: f32,
        transform_matrix: &[f32],
        invert: bool,
        ww: f32,
        wl: f32,
        color_map: Option<Vec<f32>>,
    ) -> Result<Vec<u8>, JsValue> {
        
        let texture_needs_update = match &self.input_texture {
            Some(tex) => tex.width() != img_width || tex.height() != img_height,
            None => true,
        };

        if texture_needs_update {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Input Image Texture"),
                size: wgpu::Extent3d {
                    width: img_width,
                    height: img_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.input_texture = Some(texture);
        }

        let input_texture = self.input_texture.as_ref().unwrap();

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(img_data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(img_width * 4),
                rows_per_image: Some(img_height),
            },
            wgpu::Extent3d {
                width: img_width,
                height: img_height,
                depth_or_array_layers: 1,
            },
        );

        let mut mat = [[0.0; 4]; 4];
        if transform_matrix.len() == 16 {
            for i in 0..4 {
                for j in 0..4 {
                    mat[i][j] = transform_matrix[i * 4 + j];
                }
            }
        }

        let uniforms = Uniforms {
            image_size: [canvas_width, canvas_height],
            slope,
            intercept,
            ww,
            wl,
            invert: if invert { 1 } else { 0 },
            pad1: 0.0,
            transform_matrix: mat,
        };

        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        if let Some(map) = color_map {
            if map.len() == 768 {
                 self.queue.write_buffer(&self.color_map_buffer, 0, bytemuck::cast_slice(&map));
            }
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&input_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.color_map_buffer.as_entire_binding(),
                },
            ],
        });

        let out_width = canvas_width as u32;
        let out_height = canvas_height as u32;

        let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
            size: wgpu::Extent3d {
                width: out_width,
                height: out_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let unaligned_bytes_per_row = out_width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let padded_bytes_per_row_padding = (align - unaligned_bytes_per_row % align) % align;
        let padded_bytes_per_row = unaligned_bytes_per_row + padded_bytes_per_row_padding;
        
        let output_buffer_size = (padded_bytes_per_row * out_height) as u64;
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(out_height),
                },
            },
            wgpu::Extent3d {
                width: out_width,
                height: out_height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        // 读取数据
        let buffer_slice = read_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        // 等待 map 完成
        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            
            let mut result_pixels = Vec::with_capacity((out_width * out_height * 4) as usize);
            
            for y in 0..out_height {
                let start = (y * padded_bytes_per_row) as usize;
                let end = start + (out_width * 4) as usize;
                result_pixels.extend_from_slice(&data[start..end]);
            }
            
            drop(data);
            read_buffer.unmap();
            
            return Ok(result_pixels);
        } else {
            return Err(JsValue::from_str("Failed to map buffer for reading"));
        }
    }
}