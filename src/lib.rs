pub mod configuration;
pub mod observer;
pub mod potential;
pub mod propagator;
pub mod simulation;
pub mod system;
pub mod vec;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod prelude {
    pub use crate::configuration::Configuration;
    pub use crate::observer::Observer;
    pub use crate::potential::{LennardJones, Potential};
    pub use crate::propagator::monte_carlo::MonteCarlo;
    pub use crate::simulation::Simulation;
}

#[cfg(feature = "python")]
#[pymodule]
fn sargas(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<simulation::python::PySimulation>()?;
    m.add_class::<configuration::python::PyConfiguration>()?;
    m.add_class::<system::python::PySystem>()?;

    // Monte Carlo
    m.add_class::<propagator::monte_carlo::python::PyMonteCarlo>()?;
    m.add_class::<propagator::monte_carlo::python::PyMCMove>()?;

    // Molecular Dynamics
    m.add_class::<propagator::molecular_dynamics::python::PyMolecularDynamics>()?;
    m.add_class::<propagator::molecular_dynamics::python::PyIntegrator>()?;

    m.add_class::<potential::python::PyPotential>()?;
    m.add_class::<observer::python::PyObserver>()?;
    Ok(())
}
