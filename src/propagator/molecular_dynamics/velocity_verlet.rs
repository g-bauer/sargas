use super::Propagator;
use crate::system::System;
use pyo3::prelude::*;
use std::fmt;
#[derive(Clone)]
pub struct VelocityVerlet {
    dt: f64,
    dt2_2: f64,
}

impl VelocityVerlet {
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            dt2_2: 0.5 * dt * dt,
        }
    }
}

impl Propagator for VelocityVerlet {
    fn propagate(&mut self, system: &mut System) {
        let mut squared_velocity = 0.0;

        for i in 0..system.nparticles {
            system.positions[i] += self.dt * system.velocities[i] + self.dt2_2 * system.forces[i];
            system.positions[i].apply_pbc(system.box_length);
        }

        let new_forces = system.compute_forces();

        for i in 0..system.nparticles {
            system.velocities[i] += (new_forces[i] + system.forces[i]) * 0.5 * self.dt;
            squared_velocity += system.velocities[i].dot_product();
            system.forces[i] = new_forces[i];
        }

        system.kinetic_energy = 0.5 * squared_velocity;
    }

    fn adjust(&mut self, _: &System) {}

    fn report(&self) {}
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

#[pyclass(name = "VelocityVerlet", unsendable)]
#[derive(Clone)]
pub struct PyVelocityVerlet {
    pub _data: VelocityVerlet,
}

#[pymethods]
impl PyVelocityVerlet {
    #[new]
    fn new(dt: f64) -> Self {
        Self {
            _data: VelocityVerlet::new(dt),
        }
    }
}
