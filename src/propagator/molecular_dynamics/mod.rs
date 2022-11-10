use super::Propagator;
use crate::error::SargasError;
use crate::system::System;
use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;
pub mod andersen;
pub mod berendsen;
pub mod langevin;
pub mod lowe_andersen;
pub mod velocity_rescaling;
pub mod velocity_verlet;
use andersen::Andersen;
use berendsen::Berendsen;
use langevin::Langevin;
use lowe_andersen::LoweAndersen;
use velocity_rescaling::VelocityRescaling;
use velocity_verlet::VelocityVerlet;

pub trait Integrator: Display {
    fn apply(&mut self, system: &mut System);
}

pub trait Thermostat: Display {
    fn apply(&self, system: &mut System);
    // fn frequency(&self) -> usize;
}

struct MolecularDynamics {
    pub integrator: Rc<RefCell<dyn Integrator>>,
    pub thermostat: Option<Rc<RefCell<dyn Thermostat>>>,
}

impl Display for MolecularDynamics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Molecular Dynamics\n==================\n")?;
        write!(
            f,
            "{}\n",
            self.integrator.try_borrow().expect("Already borrowed.")
        )?;
        if let Some(thermostat) = &self.thermostat {
            write!(
                f,
                "\n{}\n",
                thermostat.try_borrow().expect("Already borrowed.")
            )?;
        }
        Ok(())
    }
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

        /// Langevin/Brownian dynamics integrator.
        ///
        /// Integrates the Hamiltonian equations of motion with additional
        /// forces that describe friction with surrounding solvent and random
        /// forces. Produces canonical distribution.
        ///
        /// Parameters
        /// ----------
        /// timestep : float
        ///     the timestep in reduced time.
        /// temperature : float
        ///     temperature.
        /// damping_constang : float
        ///     damping constant gamma for frictional
        ///     and random forces.
        ///
        /// Returns
        /// -------
        /// Integrator
        #[staticmethod]
        #[pyo3(text_signature = "(timestep, temperature, damping_constant)")]
        fn langevin(timestep: f64, temperature: f64, damping_constant: f64) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Langevin::new(
                    timestep,
                    temperature,
                    damping_constant,
                ))),
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

        /// Lowe-Andersen thermostat.
        ///
        /// Selects particle pairs within interaction radius and
        /// imprints Maxwell Boltzmann distribution to the relative velocities
        /// along the separation vector and thus conserves angular and
        /// translational momentum.
        ///
        /// Parameters
        /// ----------
        /// target_temperature : float
        ///     target reduced temperature of the system
        /// timestep : float
        ///     simulation time step in reduced time
        /// collision_frequency : float
        ///     frequency with which particles are selected in reduced time
        /// interaction_radius : float
        ///     radius in which bath collisions between pairs of particles
        ///     are considered
        ///
        /// Returns
        /// -------
        /// Thermostat
        #[staticmethod]
        #[pyo3(
            text_signature = "(target_temperature, timestep, collision_frequency, interaction_radius)"
        )]
        fn lowe_andersen(
            target_temperature: f64,
            timestep: f64,
            collision_frequency: f64,
            interaction_radius: f64,
        ) -> Self {
            Self {
                _data: Rc::new(RefCell::new(LoweAndersen::new(
                    target_temperature,
                    timestep,
                    collision_frequency,
                    interaction_radius,
                ))),
            }
        }

        /// Berendsen thermostat (weak coupling).
        ///
        /// Parameters
        /// ----------
        /// target_temperature : float
        ///     target reduced temperature of the system
        /// timestep : float
        ///     simulation time step in reduced time
        /// tau : float
        ///     time constant for coupling
        ///
        /// Returns
        /// -------
        /// Thermostat
        #[staticmethod]
        #[pyo3(text_signature = "(target_temperature, timestep, tau)")]
        fn berendsen(target_temperature: f64, timestep: f64, tau: f64) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Berendsen::new(
                    target_temperature,
                    timestep,
                    tau,
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
