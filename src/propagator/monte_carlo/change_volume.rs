use super::{metropolis, MCMove, MoveProposal};
use crate::system::System;
use pyo3::prelude::*;
use rand::{distributions::Uniform, Rng};
use rand::{rngs::ThreadRng, thread_rng};
use std::fmt;

/// Methods to change the volume of the system.
#[derive(Clone)]
pub struct ChangeVolume {
    pub pressure: f64,
    accepted_total: usize,
    attempted_total: usize,
    pub accepted: usize,
    pub attempted: usize,
    pub target_acceptance: f64,
    pub rng: ThreadRng,
    pub maximum_displacement: f64,
    pub select_displacement: Uniform<f64>,
}

impl ChangeVolume {
    pub fn new(maximum_displacement: f64, target_acceptance: f64, pressure: f64) -> Self {
        Self {
            pressure: pressure,
            accepted_total: 0,
            attempted_total: 0,
            accepted: 0,
            attempted: 0,
            target_acceptance,
            rng: thread_rng(),
            maximum_displacement,
            select_displacement: Uniform::new_inclusive(
                -maximum_displacement,
                maximum_displacement,
            ),
        }
    }
}

impl MCMove for ChangeVolume {
    fn initialize(&mut self, _system: &System) {}

    fn apply(&mut self, system: &mut System) {
        if system.nparticles == 0 {
            return;
        }
        self.attempted += 1;

        // Store current values for volume and energy
        let energy_old = system.energy(); // todo: replace with currently stored value for energy
        let volume_old = system.volume();

        // Change (logarithmic) volume
        let delta_ln_v = self.rng.sample(&self.select_displacement);
        let delta_v = delta_ln_v.exp();
        let box_length_new = (volume_old + delta_v).cbrt();
        system.rescale_box_length(box_length_new);
        let (energy_new, virial_new) = system.energy_virial();

        let boltzmann_factor = -system.beta
            * (energy_new - energy_old + self.pressure * delta_v)
            * (system.nparticles + 1) as f64
            * (system.volume() / volume_old).ln();
        match metropolis(boltzmann_factor, &mut self.rng) {
            MoveProposal::Accepted => {
                self.accepted += 1;
                system.energy = energy_new;
                system.virial = virial_new;
            }
            MoveProposal::Rejected => {
                system.rescale_box_length(1.0 / box_length_new);
            }
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
        self.maximum_displacement *= match quotient {
            q if q > 1.5 => 1.5,
            q if q < 0.5 => 0.5,
            q => q,
        };
        if self.maximum_displacement > 0.5 * system.box_length {
            self.maximum_displacement = 0.5 * system.box_length;
        }
        self.select_displacement =
            Uniform::new_inclusive(-self.maximum_displacement, self.maximum_displacement);
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
            "Particle Displacement Move\n==========================\nattempts: {}\naccepted: {}\naccepted: {:.2} %\ncurrent displacement: {:.3}",
            self.attempted_total, self.accepted_total, acceptance, self.maximum_displacement
        )
    }
}

#[pyclass(name = "ChangeVolume", unsendable)]
#[derive(Clone)]
pub struct PyChangeVolume {
    pub _data: ChangeVolume,
}

#[pymethods]
impl PyChangeVolume {
    #[new]
    fn new(maximum_displacement: f64, target_acceptance: f64, pressure: f64) -> Self {
        Self {
            _data: ChangeVolume::new(maximum_displacement, target_acceptance, pressure),
        }
    }
}
