use std::fmt::Display;

use crate::{error::SargasError, system::System};
pub mod molecular_dynamics;
pub mod monte_carlo;
#[cfg(feature = "chemfiles")]
pub mod trajectory_reader;

pub trait Propagator: Display {
    fn propagate(&mut self, system: &mut System) -> Result<(), SargasError>;
    fn adjust(&mut self, system: &mut System);
}

