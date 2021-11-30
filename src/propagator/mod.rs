use crate::system::System;
use chemfiles;
use thiserror::Error;
pub mod molecular_dynamics;
pub mod monte_carlo;
pub mod trajectory_reader;

#[derive(Error, Debug)]
pub enum PropagatorError {
    #[error(transparent)]
    ChemfilesError(#[from] chemfiles::Error),
    #[error("End of trajectory reached.")]
    TrajectoryEnd,
}

pub trait Propagator {
    fn propagate(&mut self, system: &mut System) -> Result<(), PropagatorError>;
    fn adjust(&mut self, system: &mut System);
}

#[cfg(feature = "python")]
mod python {
    use super::PropagatorError;
    use pyo3::exceptions::PyIOError;
    use pyo3::PyErr;

    impl From<PropagatorError> for PyErr {
        fn from(err: PropagatorError) -> Self {
            PyIOError::new_err(err.to_string())
        }
    }
}
