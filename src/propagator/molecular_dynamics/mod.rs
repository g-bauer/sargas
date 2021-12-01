use super::{Propagator};
use crate::error::SargasError;
use crate::system::System;
use std::cell::RefCell;
use std::rc::Rc;
pub mod andersen;
pub mod velocity_rescaling;
pub mod velocity_verlet;
use andersen::Andersen;
use velocity_rescaling::VelocityRescaling;
use velocity_verlet::VelocityVerlet;

pub trait Integrator {
    fn apply(&mut self, system: &mut System);
}

pub trait Thermostat {
    fn apply(&self, system: &mut System);
    // fn frequency(&self) -> usize;
}

struct MolecularDynamics {
    pub integrator: Rc<RefCell<dyn Integrator>>,
    pub thermostat: Option<Rc<RefCell<dyn Thermostat>>>,
}

impl Propagator for MolecularDynamics {
    fn propagate(&mut self, system: &mut System) -> Result<(), SargasError> {
        self.integrator.as_ref().borrow_mut().apply(system);
        Ok(())
    }

    fn adjust(&mut self, system: &mut System) {
        if let Some(thermostat) = self.thermostat.as_mut() {
            thermostat.as_ref().borrow_mut().apply(system)
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "Integrator", unsendable)]
    #[derive(Clone)]
    pub struct PyIntegrator {
        pub _data: Rc<RefCell<dyn Integrator>>,
    }

    #[pymethods]
    impl PyIntegrator {
        #[staticmethod]
        fn velocity_verlet(timestep: f64) -> Self {
            Self {
                _data: Rc::new(RefCell::new(VelocityVerlet::new(timestep))),
            }
        }
    }

    #[pyclass(name = "Thermostat", unsendable)]
    #[derive(Clone)]
    pub struct PyThermostat {
        pub _data: Rc<RefCell<dyn Thermostat>>,
    }

    #[pymethods]
    impl PyThermostat {
        #[staticmethod]
        fn velocity_rescaling(target_temperature: f64) -> Self {
            Self {
                _data: Rc::new(RefCell::new(VelocityRescaling::new(target_temperature))),
            }
        }

        #[staticmethod]
        fn andersen(target_temperature: f64, timestep: f64, collision_frequency: f64) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Andersen::new(
                    target_temperature,
                    timestep,
                    collision_frequency,
                ))),
            }
        }
    }

    #[pyclass(name = "MolecularDynamics", unsendable)]
    #[derive(Clone)]
    pub struct PyMolecularDynamics {
        pub _data: Rc<RefCell<dyn Propagator>>,
    }

    #[pymethods]
    impl PyMolecularDynamics {
        #[new]
        fn new(integrator: PyIntegrator, thermostat: Option<PyThermostat>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(MolecularDynamics {
                    integrator: integrator._data.clone(),
                    thermostat: thermostat.map(|t| t._data.clone()),
                })),
            }
        }
    }
}
