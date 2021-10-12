use super::Integrator;
use crate::system::System;
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
        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                v[i] += system.configuration.forces[i] * 0.5 * self.dt;
                system.configuration.positions[i] += self.dt * v[i];
                system.configuration.positions[i].apply_pbc(system.configuration.box_length);
            }
        }
        system.compute_forces_inplace();
        if let Some(v) = system.configuration.velocities.as_mut() {
            for i in 0..system.configuration.nparticles {
                v[i] += system.configuration.forces[i] * 0.5 * self.dt;
                squared_velocity += v[i].dot(&v[i]);
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
