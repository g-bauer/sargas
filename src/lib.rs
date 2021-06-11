pub mod observer;
pub mod potential;
pub mod propagator;
pub mod simulation;
pub mod system;
pub mod vec;
use pyo3::prelude::*;

pub mod prelude {
    pub use crate::observer::Observer;
    pub use crate::potential::{LennardJones, Potential};
    pub use crate::propagator::monte_carlo::{MonteCarlo, DisplaceParticle, PyDisplaceParticle};
    pub use crate::simulation::Simulation;
    pub use crate::system::System;
}

#[pymodule]
fn loki(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<simulation::PySimulation>()?;
    m.add_class::<system::PySystem>()?;
    m.add_class::<propagator::monte_carlo::PyDisplaceParticle>()?;
    m.add_class::<propagator::monte_carlo::PyMonteCarlo>()?;
    m.add_class::<propagator::monte_carlo::PyMCMove>()?;
    // m.add_class::<propagator::velocity_verlet::PyVelocityVerlet>()?;
    m.add_class::<potential::PyPotential>()?;
    m.add_class::<observer::PyObserver>()?;
    Ok(())
}
