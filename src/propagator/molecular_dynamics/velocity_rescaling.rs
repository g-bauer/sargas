use std::fmt::Display;

use super::Thermostat;
use crate::system::System;

pub struct VelocityRescaling {
    target_temperature: f64,
}

impl VelocityRescaling {
    pub fn new(target_temperature: f64) -> Self {
        Self { target_temperature }
    }
}

impl Display for VelocityRescaling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Velocity Rescaling\n")?;
        write!(f, "  target temperature: {}\n", self.target_temperature)
    }
}

impl Thermostat for VelocityRescaling {
    fn apply(&self, system: &mut System) {
        if let Some(v) = system.configuration.velocities.as_mut() {
            let current_temperature =
                2.0 / 3.0 / system.configuration.nparticles as f64 * system.kinetic_energy.unwrap();
            let scaling_factor = (self.target_temperature / current_temperature).sqrt();
            v.iter_mut().for_each(|vi| *vi *= scaling_factor);
            system.kinetic_energy = system.kinetic_energy.map(|ke| ke * scaling_factor.powi(2));
        } else {
            return;
        }
    }
}
