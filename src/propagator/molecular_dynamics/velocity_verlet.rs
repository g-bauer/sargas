use super::{Integrator, Propagator};
use crate::system::System;
use pyo3::prelude::*;
use std::fmt;

#[derive(Clone)]
pub struct VelocityVerlet {
    /// time step
    dt: f64,
    /// one half times squared time step
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

impl Integrator for VelocityVerlet {
    fn apply(&mut self, system: &mut System) {
        let mut squared_velocity = 0.0;
        {
            let v = system.configuration.velocities.as_ref().unwrap();
            for i in 0..system.configuration.nparticles {
                system.configuration.positions[i] +=
                    self.dt * v[i] + self.dt2_2 * system.configuration.forces[i];
                system.configuration.positions[i].apply_pbc(system.configuration.box_length);
            }
        }

        // let new_forces = system.compute_forces();
        system.compute_forces_inplace();

        {
            let v = system.configuration.velocities.as_mut().unwrap();
            for i in 0..system.configuration.nparticles {
                v[i] += system.configuration.forces[i] * 0.5 * self.dt;
                squared_velocity += v[i].dot(&v[i]);
                // system.configuration.forces[i] = new_forces[i];
            }
        }
        system.kinetic_energy = Some(0.5 * squared_velocity);
    }
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

// #[pyclass(name = "VelocityVerlet", unsendable)]
// #[derive(Clone)]
// pub struct PyVelocityVerlet {
//     pub _data: VelocityVerlet,
// }

// #[pymethods]
// impl PyVelocityVerlet {
//     #[new]
//     fn new(dt: f64) -> Self {
//         Self {
//             _data: VelocityVerlet::new(dt),
//         }
//     }
// }
