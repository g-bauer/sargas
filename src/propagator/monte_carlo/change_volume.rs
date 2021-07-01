use super::MCMove;
use crate::system::System;
use rand::{distributions::Uniform, Rng};
use rand::{rngs::ThreadRng, thread_rng};
use std::fmt;

/// Methods to change the volume of the system.
#[derive(Clone)]
pub struct ChangeVolume {
    pub pressure: f64,
    pub temperature: f64,
    beta: f64,
    accepted_total: usize,
    attempted_total: usize,
    pub accepted: usize,
    pub attempted: usize,
    pub target_acceptance: f64,
    pub rng: ThreadRng,
    pub maximum_scaling: f64,
    pub select_scaling: Uniform<f64>,
}

impl ChangeVolume {
    pub fn new(
        maximum_scaling: f64,
        target_acceptance: f64,
        pressure: f64,
        temperature: f64,
    ) -> Self {
        Self {
            pressure: pressure,
            temperature: temperature,
            beta: 1.0 / temperature,
            accepted_total: 0,
            attempted_total: 0,
            accepted: 0,
            attempted: 0,
            target_acceptance,
            rng: thread_rng(),
            maximum_scaling,
            select_scaling: Uniform::new_inclusive(-1.0 * maximum_scaling, 1.0 * maximum_scaling),
        }
    }
}

impl MCMove for ChangeVolume {
    fn initialize(&mut self, _system: &System) {}

    fn apply(&mut self, system: &mut System) {
        if system.configuration.nparticles == 0 {
            return;
        }
        self.attempted += 1;

        // Store current values for volume and energy
        let energy_old = system.energy;
        let volume_old = system.configuration.volume();
        let box_length_old = system.configuration.box_length;

        let delta_ln_l = self.rng.sample(&self.select_scaling);
        let delta_l = delta_ln_l.exp();
        let box_length_new = box_length_old * delta_l;
        system.configuration.rescale_box_length(box_length_new);
        let volume_new = box_length_new.powi(3);
        let (energy_new, virial_new) = system.energy_virial();

        let boltzmann_factor = -self.beta
            * (energy_new - energy_old + self.pressure * (volume_new - volume_old))
            + (system.configuration.nparticles + 1) as f64 * (volume_old / volume_new).ln();

        let acceptance: f64 = self.rng.gen();
        if acceptance < f64::exp(boltzmann_factor) {
            self.accepted += 1;
            system.energy = energy_new;
            system.virial = virial_new;
        } else {
            system.configuration.rescale_box_length(box_length_old);
        }
    }

    fn adjust(&mut self, system: &System) {
        self.attempted_total += self.attempted;
        self.accepted_total += self.accepted;
        if self.attempted == 0 {
            return;
        }
        let current_acceptance = self.accepted as f64 / self.attempted as f64;
        let quotient = current_acceptance / self.target_acceptance;
        self.maximum_scaling *= match quotient {
            q if q > 1.5 => 1.5,
            q if q < 0.5 => 0.5,
            q => q,
        };
        self.select_scaling =
            Uniform::new_inclusive(-1.0 * self.maximum_scaling, 1.0 * self.maximum_scaling);
        self.attempted = 0;
        self.accepted = 0;
    }

    fn print_statistic(&self) {
        println!(
            "{:?}",
            self.accepted_total as f64 / self.attempted_total as f64
        );
    }
}

impl fmt::Display for ChangeVolume {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let acceptance = if self.attempted_total == 0 {
            0.0
        } else {
            self.accepted_total as f64 / self.attempted_total as f64 * 100.0
        };
        write!(
            f,
            "Volume Change Move\n==========================\nattempts: {}\naccepted: {}\naccepted: {:.2} %\ncurrent scaling: {:.3}",
            self.attempted_total, self.accepted_total, acceptance, self.maximum_scaling
        )
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "ChangeVolume", unsendable)]
    #[derive(Clone)]
    pub struct PyChangeVolume {
        pub _data: ChangeVolume,
    }

    #[pymethods]
    impl PyChangeVolume {
        #[new]
        fn new(
            maximum_displacement: f64,
            target_acceptance: f64,
            pressure: f64,
            temperature: f64,
        ) -> Self {
            Self {
                _data: ChangeVolume::new(
                    maximum_displacement,
                    target_acceptance,
                    pressure,
                    temperature,
                ),
            }
        }
    }
}
