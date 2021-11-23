use super::Thermostat;
use crate::system::System;
use crate::vec::Vec3;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

pub struct Andersen {
    timestep: f64,
    collision_frequency: f64,
    distribution: Normal<f64>,
}

impl Andersen {
    pub fn new(target_temperature: f64, timestep: f64, collision_frequency: f64) -> Self {
        Self {
            timestep,
            collision_frequency,
            distribution: Normal::new(0.0, target_temperature.sqrt()).unwrap(),
        }
    }
}

impl Thermostat for Andersen {
    fn apply(&self, system: &mut System) {
        let mut rng = rand::thread_rng();
        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                if rng.gen::<f64>() < self.collision_frequency * self.timestep {
                    v[i] = Vec3::new(
                        self.distribution.sample(&mut rng),
                        self.distribution.sample(&mut rng),
                        self.distribution.sample(&mut rng),
                    )
                }
            }
        } else {
            return;
        }
    }
}
