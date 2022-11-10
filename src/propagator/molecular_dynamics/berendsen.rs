use std::fmt::Display;

use super::Thermostat;
use crate::system::System;

pub struct Berendsen {
    target_temperature: f64,
    timestep: f64,
    tau: f64,
}

impl Berendsen {
    pub fn new(target_temperature: f64, timestep: f64, tau: f64) -> Self {
        Self {
            target_temperature,
            timestep,
            tau,
        }
    }
}

impl Display for Berendsen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Berendsen thermostat\n")?;
        write!(f, "  target temperature: {}\n", self.target_temperature)?;
        write!(f, "  time step:          {}\n", self.timestep)?;
        write!(f, "  tau:                {}\n", self.tau)
    }
}

impl Thermostat for Berendsen {
    fn apply(&self, system: &mut System) {
        if let Some(v) = system.configuration.velocities.as_mut() {
            let current_temperature =
                2.0 / 3.0 / system.configuration.nparticles as f64 * system.kinetic_energy.unwrap();
            let scaling_factor = (1.0
                + self.timestep / self.tau * (self.target_temperature / current_temperature - 1.0))
                .sqrt();
            for i in 0..system.configuration.nparticles {
                v[i] *= scaling_factor;
            }
        } else {
            return;
        }
    }
}
