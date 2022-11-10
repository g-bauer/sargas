use rand_distr::{Distribution, Normal};

use super::Integrator;
use crate::{system::System, vec::Vec3};
use std::fmt;

#[derive(Clone)]
pub struct Langevin {
    /// time step
    dt: f64,
    /// temperature
    temperature: f64,
    /// damping constant (gamma)
    damping_constant: f64,
    /// distribution
    distribution: Normal<f64>,
}

impl Langevin {
    pub fn new(dt: f64, temperature: f64, damping_constant: f64) -> Self {
        Self {
            dt,
            temperature,
            damping_constant,
            distribution: Normal::new(0.0, temperature.sqrt()).unwrap(),
        }
    }
}

impl Integrator for Langevin {
    fn apply(&mut self, system: &mut System) {
        let mut squared_velocity = 0.0;
        let mut rng = rand::thread_rng();
        let damping = self.damping_constant * self.dt;

        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                // velocity
                v[i] += system.configuration.forces[i] * 0.5 * self.dt;
                // positions (0.5 * dt)
                system.configuration.positions[i] += 0.5 * self.dt * v[i];
                system.configuration.positions[i].apply_pbc(system.configuration.box_length);
                // middle step: friction and random force
                let g = Vec3::new(
                    self.distribution.sample(&mut rng),
                    self.distribution.sample(&mut rng),
                    self.distribution.sample(&mut rng),
                );
                v[i] = (-damping).exp() * v[i] + g * (1.0 - (-2.0 * damping).exp()).sqrt();
            }
        }
        system.compute_forces_inplace();
        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                v[i] += system.configuration.forces[i] * 0.5 * self.dt;
                squared_velocity += v[i].dot(&v[i]);
            }
        }
        system.kinetic_energy = Some(0.5 * squared_velocity);
    }
}

impl fmt::Display for Langevin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Langevin\n=====================\n")?;
        write!(f, "  time step:        {}", self.dt)?;
        write!(f, "  temperature:      {}", self.temperature)?;
        write!(f, "  damping constant: {}", self.damping_constant)
    }
}
