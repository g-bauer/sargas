use std::fmt::Display;

use super::Thermostat;
use crate::system::System;
use crate::vec::Vec3;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

pub struct Andersen {
    timestep: f64,
    collision_frequency: f64,
    target_temperature: f64,
    distribution: Normal<f64>,
}

impl Andersen {
    pub fn new(target_temperature: f64, timestep: f64, collision_frequency: f64) -> Self {
        Self {
            timestep,
            collision_frequency,
            target_temperature,
            distribution: Normal::new(0.0, target_temperature.sqrt()).unwrap(),
        }
    }
}

impl Display for Andersen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Andersen thermostat\n")?;
        write!(f, "  target temperature:  {}\n", self.target_temperature)?;
        write!(f, "  time step:           {}\n", self.timestep)?;
        write!(f, "  collision frequency: {}\n", self.collision_frequency)
    }
}

impl Thermostat for Andersen {
    fn apply(&self, system: &mut System) {
        let mut rng = rand::thread_rng();
        if let Some(v) = system.configuration.velocities.as_mut() {
            let mut squared_velocity = 0.0;
            for i in 0..system.configuration.nparticles {
                if rng.gen::<f64>() < self.collision_frequency * self.timestep {
                    v[i] = Vec3::new(
                        self.distribution.sample(&mut rng),
                        self.distribution.sample(&mut rng),
                        self.distribution.sample(&mut rng),
                    )
                }
                squared_velocity += v[i].dot(&v[i]);
            }
            system.kinetic_energy = Some(0.5 * squared_velocity);
        } else {
            return;
        }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::utils::test_system;

    #[test]
    fn andersen() {
        let mut system = test_system();
        let thermostat = Andersen::new(0.8, 0.1, 0.1);
        thermostat.apply(&mut system);
        assert_relative_eq!(
            system.kinetic_energy.unwrap(),
            system
                .configuration
                .kinetic_energy_from_velocities()
                .unwrap()
        )
    }
}
