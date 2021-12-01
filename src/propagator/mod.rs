use crate::{error::SargasError, system::System};
pub mod molecular_dynamics;
pub mod monte_carlo;
pub mod trajectory_reader;

pub trait Propagator {
    fn propagate(&mut self, system: &mut System) -> Result<(), SargasError>;
    fn adjust(&mut self, system: &mut System);
}

