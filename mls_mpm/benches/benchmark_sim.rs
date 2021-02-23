use criterion::{criterion_group, criterion_main, Criterion};
use itertools::iproduct;
use mls_mpm::{Mat2, Particle2D, Simulation2D, Vec2};

pub fn criterion_benchmark(c: &mut Criterion) {
    let particles = iproduct!(0..100, 0..100)
        .map(|(x, y)| Particle2D {
            position: Vec2::new(10.0 + 0.1 * x as f32, 10.0 + 0.1 * y as f32),
            velocity: Vec2::new(0.5, 0.03),
            momentum: Mat2::zero(),
            mass: 1.0,
        })
        .collect::<Vec<_>>();

    let num_particles = particles.len();

    let mut sim = Simulation2D::new(32, 32, particles, Vec2::new(0.0, -0.05), 1.0);
    c.bench_function(&format!("Sim step with {} particles", num_particles), |b| b.iter(|| sim.step()));

    println!("Steps run: {}", sim.steps_run);
    println!("Particles[0]: {:?}", sim.particles[0]);
    println!("Particles[{}]: {:?}", num_particles - 1, sim.particles.last());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
