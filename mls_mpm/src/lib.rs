pub use glam::{f32::Mat2, f32::Vec2, IVec2, UVec2};

#[derive(Debug)]
pub struct Particle2D {
    pub position: Vec2,
    pub velocity: Vec2,
    pub momentum: Mat2, // Affine momentum
    pub mass: f32,
}

pub struct Simulation2D {
    pub x_size: usize,
    pub y_size: usize,
    pub num_cells: usize,
    pub particles: Vec<Particle2D>,
    pub gravity_times_dt: Vec2,
    pub dt: f32,
    pub steps_run: usize,
}

impl Simulation2D {
    pub fn new(x_size: usize, y_size: usize, particles: Vec<Particle2D>, gravity: Vec2, dt: f32) -> Simulation2D {
        Simulation2D {
            x_size,
            y_size,
            num_cells: x_size * y_size,
            particles,
            gravity_times_dt: gravity * dt,
            dt,
            steps_run: 0,
        }
    }

    pub fn step(&mut self) {
        self.steps_run += 1;

        // Build empty grid
        let mut grid = vec![Cell::zero(); self.num_cells];

        let half = Vec2::new(0.5, 0.5);
        let tq = Vec2::new(0.75, 0.75);

        // Convert particles to cell grid
        for particle in self.particles.iter() {
            let cell_diff: Vec2 = particle.position - particle.position.floor() - half;

            let weights: [Vec2; 3] = [
                (half - cell_diff).powf(2.0) * 0.5f32,
                tq - cell_diff.powf(2.0),
                (half + cell_diff).powf(2.0) * 0.5f32,
            ];

            let cell_index = particle.position.as_i32();

            for (x_offset, x_weight) in (-1i32..=1).zip(weights.iter()) {
                for (y_offset, y_weight) in (-1i32..=1).zip(weights.iter()) {
                    let cell_position: UVec2 = (cell_index + IVec2::new(x_offset, y_offset)).as_u32();

                    let cell_dist: Vec2 = cell_position.as_f32() - particle.position + half;

                    let q = particle.momentum * cell_dist;

                    let cell = &mut grid[cell_position.y as usize * self.x_size + cell_position.x as usize];

                    let mass_contrib = x_weight.x * y_weight.y * particle.mass;

                    cell.mass += mass_contrib;
                    cell.velocity += (particle.velocity + q) * mass_contrib;
                }
            }
        }

        // Update grid velocity
        for (i, cell) in grid.iter_mut().enumerate().filter(|(_, cell)| cell.mass > 0.0) {
            // Convert momentum to velocity; apply gravity
            cell.velocity /= cell.mass;
            cell.velocity += self.gravity_times_dt;

            // Boundary conditions
            let x = i % self.x_size;
            let y = i / self.x_size;

            if x < 2 || x > self.x_size - 3 {
                cell.velocity.x = 0.0;
            }

            if y < 2 || y > self.y_size - 3 {
                cell.velocity.y = 0.0;
            }
        }

        // Convert cell grid back to particles
        for particle in self.particles.iter_mut() {
            particle.velocity = Vec2::zero();

            let cell_diff: Vec2 = particle.position - particle.position.floor() - half;

            let weights: [Vec2; 3] = [
                (half - cell_diff).powf(2.0) * 0.5f32,
                tq - cell_diff.powf(2.0),
                (half + cell_diff).powf(2.0) * 0.5f32,
            ];

            let cell_index = particle.position.as_i32();

            let mut b = Mat2::zero();

            for (x_offset, x_weight) in (-1i32..=1).zip(weights.iter()) {
                for (y_offset, y_weight) in (-1i32..=1).zip(weights.iter()) {
                    let weight = x_weight.x * y_weight.y;

                    let cell_position: UVec2 = (cell_index + IVec2::new(x_offset, y_offset)).as_u32();

                    let cell_dist: Vec2 = cell_position.as_f32() - particle.position + half;

                    let weighted_velocity: Vec2 = grid[cell_position.y as usize * self.x_size + cell_position.x as usize].velocity * weight;

                    // APIC paper equation 10, constructing inner term for B
                    let term = Mat2::from_cols(weighted_velocity * cell_dist.x, weighted_velocity * cell_dist.y);

                    b = b.add_mat2(&term);

                    particle.velocity += weighted_velocity;
                }
            }

            particle.momentum = b * 4.0;
            particle.position += particle.velocity * self.dt; // Advect particles

            // Safety clamp to ensure particles don't exit simulation domain
            particle.position = Vec2::new(
                particle.position.x.clamp(1.0, self.x_size as f32 - 2.0),
                particle.position.y.clamp(1.0, self.y_size as f32 - 2.0),
            );
        }
    }
}

#[derive(Clone)]
struct Cell {
    pub velocity: Vec2,
    pub mass: f32,
}

impl Cell {
    pub fn zero() -> Cell {
        Cell {
            velocity: Vec2::zero(),
            mass: 0.0,
        }
    }
}
