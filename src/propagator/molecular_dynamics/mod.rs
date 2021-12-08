use super::Propagator;
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
        self.integrator.borrow_mut().apply(system);
        Ok(())
    }

    fn adjust(&mut self, system: &mut System) {
        if let Some(thermostat) = self.thermostat.as_mut() {
            thermostat.borrow_mut().apply(system)
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
        /// Velocity Verlet integrator.
        ///
        /// Parameters
        /// ----------
        /// timestep : float
        ///     the timestep in reduced time.
        ///
        /// Returns
        /// -------
        /// Integrator
        #[staticmethod]
        #[pyo3(text_signature = "(timestep)")]
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
        /// Velocity rescaling thermostat.
        ///
        /// This algorithm computes a scaling factor that contains
        /// the ratio between the target and the current temperature.
        /// This scaling factor is then used to modify the velocities
        /// such that the target temperature is reached.
        ///
        /// Parameters
        /// ----------
        /// target_temperature : float
        ///     target reduced temperature of the system
        ///
        /// Returns
        /// -------
        /// Thermostat
        #[staticmethod]
        #[pyo3(text_signature = "(target_temperature)")]
        fn velocity_rescaling(target_temperature: f64) -> Self {
            Self {
                _data: Rc::new(RefCell::new(VelocityRescaling::new(target_temperature))),
            }
        }

        /// Andersen thermostat.
        ///
        /// Imprints a Maxwell-Boltzmann distributed velocity
        /// according to target temperature on a random particle.
        ///
        /// Parameters
        /// ----------
        /// target_temperature : float
        ///     target reduced temperature of the system
        /// timestep : float
        ///     simulation time step in reduced time
        /// collision_frequency : float
        ///     frequency with which particles are selected in reduced time
        ///
        /// Returns
        /// -------
        /// Thermostat
        #[staticmethod]
        #[pyo3(text_signature = "(target_temperature, timestep, collision_frequency)")]
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

    /// Molecular Dynamics propagator
    ///
    /// If no thermostat is used, an NVE ensemble is generated.
    ///
    /// Parameters
    /// ----------
    /// integrator : Integrator
    ///     the integration algorithm that propagates the system
    /// thermostat : Thermostat, optional
    ///     an alorithm used to control the system temperature.
    ///     Defaults to None.
    ///
    /// Returns
    /// -------
    /// MolecularDynamics
    #[pyclass(name = "MolecularDynamics", unsendable)]
    #[pyo3(text_signature = "(integrator, thermostat=None)")]
    #[derive(Clone)]
    pub struct PyMolecularDynamics {
        pub _data: Rc<RefCell<dyn Propagator>>,
    }

    /// Molecular Dynamics propagator.
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
