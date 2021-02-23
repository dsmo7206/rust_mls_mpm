use itertools::iproduct;
use kiss3d::{
    camera::Camera,
    context::Context,
    planar_camera::{FixedView, PlanarCamera},
    post_processing::PostProcessingEffect,
    renderer::Renderer,
    resource::{AllocationType, BufferType, Effect, GPUVec, ShaderAttribute, ShaderUniform},
    text::Font,
    window::{State, Window},
};
use mls_mpm::{Mat2, Particle2D, Simulation2D, Vec2};
use nalgebra::{Matrix4, Point2, Point3};
use std::time::Instant;

struct AppState {
    point_cloud_renderer: PointCloudRenderer,
    simulation: Simulation2D,
    camera: FixedView,
    frames: usize,
    zero_frame_time: Instant,
    last_known_fps: u32
}

impl State for AppState {
    // Return the custom renderer that will be called at each render loop.
    fn cameras_and_effect_and_renderer(
        &mut self,
    ) -> (
        Option<&mut dyn Camera>,
        Option<&mut dyn PlanarCamera>,
        Option<&mut dyn Renderer>,
        Option<&mut dyn PostProcessingEffect>,
    ) {
        (None, Some(&mut self.camera), Some(&mut self.point_cloud_renderer), None)
    }

    fn step(&mut self, window: &mut Window) {
        let data = self.point_cloud_renderer.colored_points.data_mut().as_mut().unwrap();
        data.clear();

        for particle in self.simulation.particles.iter() {
            data.push(Point3::new(-(particle.position.x - 16.0), particle.position.y - 16.0, 40.0));
            data.push(Point3::new(1.0, 1.0, 1.0));
        }

        self.simulation.step();

        if self.frames % 50 == 0 {
            let duration = self.zero_frame_time.elapsed();
            self.last_known_fps = (50.0 / duration.as_secs_f32()) as u32;
            self.zero_frame_time = Instant::now();
        }
        self.frames += 1;

        window.draw_text(
            &format!("Number of particles: {}, {} fps", self.point_cloud_renderer.num_points(), self.last_known_fps),
            &Point2::new(0.0, 20.0),
            60.0,
            &Font::default(),
            &Point3::new(1.0, 1.0, 1.0),
        );
    }
}

fn main() {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    let window = Window::new("Kiss3d: persistent_point_cloud");
    let app = AppState {
        point_cloud_renderer: PointCloudRenderer::new(4.0),
        simulation: Simulation2D::new(
            32,
            32,
            iproduct!(0..100, 0..100)
                .map(|(x, y)| Particle2D {
                    position: Vec2::new(10.0 + 0.1 * x as f32, 10.0 + 0.1 * y as f32),
                    velocity: Vec2::new(20.0 * (rng.gen::<f32>() - 0.5), 20.0 * (rng.gen::<f32>() - 0.5)),
                    momentum: Mat2::zero(),
                    mass: 1.0,
                })
                .collect::<Vec<_>>(),
            Vec2::new(0.0, -0.1),
            0.02,
        ),
        camera: FixedView::new(),
        frames: 0,
        zero_frame_time: Instant::now(),
        last_known_fps: 0
    };

    window.render_loop(app)
}

/// Structure which manages the display of long-living points.
struct PointCloudRenderer {
    shader: Effect,
    pos: ShaderAttribute<Point3<f32>>,
    color: ShaderAttribute<Point3<f32>>,
    proj: ShaderUniform<Matrix4<f32>>,
    view: ShaderUniform<Matrix4<f32>>,
    colored_points: GPUVec<Point3<f32>>,
    point_size: f32,
}

impl PointCloudRenderer {
    /// Creates a new points renderer.
    fn new(point_size: f32) -> PointCloudRenderer {
        let mut shader = Effect::new_from_str(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC);
        shader.use_program();

        PointCloudRenderer {
            colored_points: GPUVec::new(Vec::new(), BufferType::Array, AllocationType::StreamDraw),
            pos: shader.get_attrib::<Point3<f32>>("position").unwrap(),
            color: shader.get_attrib::<Point3<f32>>("color").unwrap(),
            proj: shader.get_uniform::<Matrix4<f32>>("proj").unwrap(),
            view: shader.get_uniform::<Matrix4<f32>>("view").unwrap(),
            shader,
            point_size,
        }
    }

    fn num_points(&self) -> usize {
        self.colored_points.len() / 2
    }
}

impl Renderer for PointCloudRenderer {
    /// Actually draws the points.
    fn render(&mut self, pass: usize, camera: &mut dyn Camera) {
        if self.colored_points.len() == 0 {
            return;
        }

        self.colored_points.load_to_gpu();

        self.shader.use_program();
        self.pos.enable();
        self.color.enable();

        camera.upload(pass, &mut self.proj, &mut self.view);

        self.color.bind_sub_buffer(&mut self.colored_points, 1, 1);
        self.pos.bind_sub_buffer(&mut self.colored_points, 1, 0);

        let ctx = Context::get();
        ctx.point_size(self.point_size);
        ctx.draw_arrays(Context::POINTS, 0, (self.colored_points.len() / 2) as i32);

        self.pos.disable();
        self.color.disable();
    }
}

const VERTEX_SHADER_SRC: &'static str = "#version 100
    attribute vec3 position;
    attribute vec3 color;
    varying   vec3 Color;
    uniform   mat4 proj;
    uniform   mat4 view;
    void main() {
        gl_Position = proj * view * vec4(position, 1.0);
        Color = color;
    }";

const FRAGMENT_SHADER_SRC: &'static str = "#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

    varying vec3 Color;
    void main() {
        gl_FragColor = vec4(Color, 1.0);
    }";
