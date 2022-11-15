pub mod configuration;
pub mod lennard_jones;
pub mod propagator;
pub mod sampler;
pub mod simulation;
pub mod system;
pub mod vec;
pub mod error;
pub mod utils;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod prelude {
    pub use crate::configuration::Configuration;
    pub use crate::lennard_jones::{LennardJones};
    pub use crate::propagator::monte_carlo::MonteCarlo;
    pub use crate::sampler::Sampler;
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
    m.add_class::<propagator::molecular_dynamics::python::PyThermostat>()?;

    // Trajectory Reader
    #[cfg(chemfiles)]
    m.add_class::<propagator::trajectory_reader::python::PyTrajectoryReader>()?;

    m.add_class::<lennard_jones::python::PyLennardJones>()?;
    m.add_class::<sampler::python::PySampler>()?;
    Ok(())
}
