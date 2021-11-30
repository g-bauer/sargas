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
}

pub trait Propagator {
    fn propagate(&mut self, system: &mut System) -> Result<(), PropagatorError>;
    fn adjust(&mut self, system: &mut System);
}
