use std::fmt::Display;

use super::Thermostat;
use crate::system::System;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

pub struct LoweAndersen {
    timestep: f64,
    collision_frequency: f64,
    target_temperature: f64,
    rt2: f64,
    distribution: Normal<f64>,
}

impl LoweAndersen {
    pub fn new(
        target_temperature: f64,
        timestep: f64,
        collision_frequency: f64,
        interaction_radius: f64,
    ) -> Self {
        Self {
            timestep,
            collision_frequency,
            target_temperature,
            rt2: interaction_radius.powi(2),
            distribution: Normal::new(0.0, (2.0 * target_temperature).sqrt()).unwrap(),
        }
    }
}

impl Display for LoweAndersen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lowe-Andersen thermostat\n")?;
        write!(f, "  target temperature:  {}\n", self.target_temperature)?;
        write!(f, "  time step:           {}\n", self.timestep)?;
        write!(f, "  collision frequency: {}\n", self.collision_frequency)
    }
}

impl Thermostat for LoweAndersen {
    fn apply(&self, system: &mut System) {
        let mut rng = rand::thread_rng();
        let box_length = system.configuration.box_length;
        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles - 1 {
                let position_i = system.configuration.positions[i];
                for j in i + 1..system.configuration.nparticles {
                    let rij =
                        (position_i - system.configuration.positions[j]).nearest_image(box_length);
                    let r2 = rij.dot(&rij);
                    if r2 <= self.rt2 {
                        if rng.gen::<f64>() < self.collision_frequency * self.timestep {
                            let mb = self.distribution.sample(&mut rng);
                            let rij_norm = rij / r2.sqrt();
                            let velocity_change =
                                0.5 * (mb - (v[i] - v[j]).dot(&rij_norm)) * rij_norm;
                            v[i] += velocity_change;
                            v[j] -= velocity_change;
                        }
                    }
                }
            }
            system.kinetic_energy = system.configuration.kinetic_energy_from_velocities();
        }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::utils::test_system;

    #[test]
    fn lowe_andersen() {
        let mut system = test_system();
        let thermostat = LoweAndersen::new(0.8, 0.1, 0.1, 1.0);
        thermostat.apply(&mut system);
        assert_relative_eq!(
            system.kinetic_energy.unwrap(),
            system
                .configuration
                .kinetic_energy_from_velocities()
                .unwrap(),
            epsilon = 1e-12
        )
    }
}
