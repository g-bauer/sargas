use crate::system::System;
pub mod monte_carlo;

pub trait Propagator {
    fn propagate(&mut self, system: &mut System);
    fn adjust(&mut self, system: &System);
}
