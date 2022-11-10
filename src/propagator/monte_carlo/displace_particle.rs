use super::MCMove;
use crate::system::System;
use crate::vec::Vec3;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use rand::{rngs::ThreadRng, thread_rng};
use std::fmt;
#[derive(Clone)]
pub struct DisplaceParticle {
    pub temperature: f64,
    beta: f64,
    accepted_total: usize,
    attempted_total: usize,
    pub accepted: usize,
    pub attempted: usize,
    pub target_acceptance: f64,
    pub rng: ThreadRng,
    pub maximum_displacement: f64,
    pub select_displacement: Uniform<f64>,
}

impl DisplaceParticle {
    pub fn new(maximum_displacement: f64, target_acceptance: f64, temperature: f64) -> Self {
        Self {
            temperature,
            beta: 1.0 / temperature,
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

impl MCMove for DisplaceParticle {
    fn initialize(&mut self, _system: &System) {}

    fn apply(&mut self, system: &mut System) {
        if system.configuration.nparticles == 0 {
            return;
        }
        self.attempted += 1;
        let i = self
            .rng
            .sample(Uniform::from(0..system.configuration.nparticles));

        let position_old = system.configuration.positions[i];
        let (energy_old, virial_old) = system.particle_energy_virial(i, &position_old, None);
        let d = self
            .select_displacement
            .sample_iter(&mut self.rng)
            .take(3)
            .collect::<Vec<f64>>();
        let displacement = Vec3::new(d[0], d[1], d[2]);
        let mut position_new = position_old + displacement;
        position_new.apply_pbc(system.configuration.box_length);
        system.configuration.positions[i] = position_new;
        let (energy_new, virial_new) = system.particle_energy_virial(i, &position_new, None);

        let acceptance: f64 = self.rng.gen();
        if acceptance < f64::exp(-self.beta * (energy_new - energy_old)) {
            self.accepted += 1;
            system.potential_energy += energy_new - energy_old;
            system.virial += virial_new - virial_old
        } else {
            system.configuration.positions[i] = position_old;
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
        if self.maximum_displacement > 0.5 * system.configuration.box_length {
            self.maximum_displacement = 0.5 * system.configuration.box_length;
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

impl fmt::Display for DisplaceParticle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let acceptance = if self.attempted_total == 0 {
            0.0
        } else {
            self.accepted_total as f64 / self.attempted_total as f64 * 100.0
        };
        write!(f, "Particle Displacement Move\n")?;
        write!(f, "  attempts:             {}\n", self.attempted_total)?;
        write!(f, "  accepted:             {}\n", self.accepted_total)?;
        write!(f, "  acceptance:           {:.2}\n", acceptance)?;
        write!(f, "  target acceptance:    {:.2}\n", self.target_acceptance)?;
        write!(
            f,
            "  current displacement: {:.3}\n",
            self.maximum_displacement
        )
    }
}

// #[cfg(feature = "python")]
// pub mod python {
//     use super::*;
//     use pyo3::prelude::*;

//     #[pyclass(name = "DisplaceParticle", unsendable)]
//     #[derive(Clone)]
//     pub struct PyDisplaceParticle {
//         pub _data: DisplaceParticle,
//     }

//     #[pymethods]
//     impl PyDisplaceParticle {
//         #[new]
//         fn new(maximum_displacement: f64, target_acceptance: f64, temperature: f64) -> Self {
//             Self {
//                 _data: DisplaceParticle::new(maximum_displacement, target_acceptance, temperature),
//             }
//         }
//     }
// }
