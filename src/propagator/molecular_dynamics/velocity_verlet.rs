use super::Integrator;
use crate::system::System;
use std::fmt;

#[derive(Clone)]
pub struct VelocityVerlet {
    /// time step
    dt: f64,
}

impl VelocityVerlet {
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }
}

impl Integrator for VelocityVerlet {
    fn apply(&mut self, system: &mut System) {
        let mut squared_velocity = 0.0;

        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                // v[i] +=
                // system.configuration.positions[i] +=
            }
        }
        // compute forces
        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                // v[i] += 
                // squared_velocity += 
            }
        }
        system.kinetic_energy = Some(0.5 * squared_velocity);
    }
}

impl fmt::Display for VelocityVerlet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Velocity Verlet\n=====================\ntimestep: {}",
            self.dt
        )
    }
}
