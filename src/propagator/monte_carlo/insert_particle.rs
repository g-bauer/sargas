use super::{metropolis, MCMove, MoveProposal};
use crate::system::System;
use crate::vec::Vec3;
use rand::{distributions::Uniform, Rng};
use rand::{rngs::ThreadRng, thread_rng};
use std::fmt;

#[derive(Clone)]
pub struct InsertDeleteParticle {
    pub chemical_potential: f64,
    pub temperature: f64,
    beta: f64,
    accepted_total: usize,
    attempted_total: usize,
    pub accepted: usize,
    pub attempted: usize,
    pub rng: ThreadRng,
    exp_beta_mu: f64,
}

impl InsertDeleteParticle {
    pub fn new(chemical_potential: f64, temperature: f64) -> Self {
        Self {
            accepted_total: 0,
            attempted_total: 0,
            accepted: 0,
            attempted: 0,
            rng: thread_rng(),
            chemical_potential,
            temperature,
            beta: 1.0 / temperature,
            exp_beta_mu: 0.0,
        }
    }

    pub fn insert_particle(&mut self, system: &mut System) {
        if system.configuration.nparticles == system.configuration.max_nparticles {
            return;
        }
        let dist = Uniform::new(0.0, system.configuration.box_length);
        system.configuration.positions.push(Vec3::new(
            self.rng.sample(dist),
            self.rng.sample(dist),
            self.rng.sample(dist),
        ));
        system.configuration.nparticles = system.configuration.positions.len();
        if system.configuration.nparticles == 1 {
            self.accepted += 1;
            return;
        }
        let (energy, virial) =
            system.particle_energy_virial(system.configuration.nparticles - 1, None);
        let boltzmann_factor =
            self.beta * (self.chemical_potential - energy) / system.configuration.density().ln();
        match metropolis(boltzmann_factor, &mut self.rng) {
            MoveProposal::Accepted => {
                self.accepted += 1;
                system.energy += energy;
                system.virial += virial;
            }
            MoveProposal::Rejected => {
                system.configuration.positions.pop();
                system.configuration.nparticles = system.configuration.positions.len();
            }
        }
    }

    fn delete_particle(&mut self, system: &mut System) {
        if system.configuration.nparticles == 0 {
            return;
        }
        let i = self
            .rng
            .sample(Uniform::from(0..system.configuration.nparticles));
        let (energy, virial) = system.particle_energy_virial(i, None);
        let boltzmann_factor = (-self.beta * (self.chemical_potential + energy)).exp()
            * system.configuration.density();
        match metropolis(boltzmann_factor, &mut self.rng) {
            MoveProposal::Accepted => {
                self.accepted += 1;
                system.energy -= energy;
                system.virial -= virial;
                system.configuration.positions.remove(i);
                system.configuration.nparticles = system.configuration.positions.len();
            }
            MoveProposal::Rejected => {
                system.configuration.positions.pop();
                system.configuration.nparticles = system.configuration.positions.len();
            }
        }
    }
}

impl MCMove for InsertDeleteParticle {
    fn initialize(&mut self, _system: &System) {
        self.exp_beta_mu = (self.beta * self.chemical_potential).exp();
    }

    fn apply(&mut self, system: &mut System) {
        self.attempted += 1;
        let insertion = self.rng.gen_bool(0.5);
        if insertion {
            self.insert_particle(system)
        } else {
            self.delete_particle(system)
        }
    }

    fn adjust(&mut self, _system: &System) {
        self.attempted_total += self.attempted;
        self.accepted_total += self.accepted;
        if self.attempted == 0 {
            return;
        }
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

impl fmt::Display for InsertDeleteParticle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let acceptance = if self.attempted_total == 0 {
            0.0
        } else {
            self.accepted_total as f64 / self.attempted_total as f64 * 100.0
        };
        write!(
            f,
            "Particle Displacement Move\n==========================\nattempts: {}\naccepted: {}\naccepted: {:.2} %",
            self.attempted_total, self.accepted_total, acceptance
        )
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "InsertDeleteParticle", unsendable)]
    #[derive(Clone)]
    pub struct PyInsertDeleteParticle {
        pub _data: InsertDeleteParticle,
    }

    #[pymethods]
    impl PyInsertDeleteParticle {
        #[new]
        fn new(chemical_potential: f64, temperature: f64) -> Self {
            Self {
                _data: InsertDeleteParticle::new(chemical_potential, temperature),
            }
        }
    }
}
