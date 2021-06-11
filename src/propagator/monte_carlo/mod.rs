use super::Propagator;
use crate::system::System;
use std::cell::RefCell;
use std::rc::Rc;
mod displace_particle;
pub use displace_particle::{DisplaceParticle, PyDisplaceParticle};
mod insert_particle;
pub use insert_particle::{InsertDeleteParticle, PyInsertDeleteParticle};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::Distribution;

pub enum AcceptanceCriterion {
    Accepted,
    Rejected,
}

impl AcceptanceCriterion {
    pub fn apply<R: Rng + ?Sized>(ln_boltzmann: f64, rng: &mut R) -> Self {
        if ln_boltzmann < 0.0 {
            AcceptanceCriterion::Accepted
        } else {
            if rng.gen::<f64>() < -ln_boltzmann.exp() {
                AcceptanceCriterion::Accepted
            } else {
                AcceptanceCriterion::Rejected
            }
        }
    }
}

pub trait MCMove {
    fn initialize(&mut self, system: &System);
    fn apply(&mut self, system: &mut System);
    fn adjust(&mut self, system: &System);
    fn print_statistic(&self);
}

pub struct MonteCarlo {
    pub moves: Vec<Rc<RefCell<dyn MCMove>>>,
    pub temperature: f64,
    rng: ThreadRng,
    weights: WeightedIndex<usize>,
}

impl MonteCarlo {
    fn new(moves: Vec<Rc<RefCell<dyn MCMove>>>, weights: Vec<usize>, temperature: f64) -> Self {
        Self {
            moves,
            temperature,
            rng: ThreadRng::default(),
            weights: WeightedIndex::new(weights).unwrap(),
        }
    }
}

impl Propagator for MonteCarlo {
    fn propagate(&mut self, system: &mut System) {
        self.moves[self.weights.sample(&mut self.rng)]
            .as_ref()
            .borrow_mut()
            .apply(system)
    }

    fn adjust(&mut self, system: &System) {
        self.moves
            .iter()
            .for_each(|m| m.as_ref().borrow_mut().adjust(&system))
    }
}

#[pyclass(name = "MonteCarlo", unsendable)]
#[derive(Clone)]
pub struct PyMonteCarlo {
    pub _data: Rc<RefCell<MonteCarlo>>,
}

#[pymethods]
impl PyMonteCarlo {
    #[new]
    fn new(moves: Vec<PyRef<PyMCMove>>, weights: Vec<usize>, temperature: f64) -> Self {
        let mvs = moves.iter().map(|mi| mi._data.clone()).collect();
        Self {
            _data: Rc::new(RefCell::new(MonteCarlo::new(mvs, weights, temperature))),
        }
    }
}

#[pyclass(name = "MCMove", unsendable)]
pub struct PyMCMove {
    _data: Rc<RefCell<dyn MCMove>>,
}

#[pymethods]
impl PyMCMove {
    #[staticmethod]
    fn displace_particle(
        maximum_displacement: f64,
        target_acceptance: f64,
        nparticles: usize,
    ) -> Self {
        let mv = DisplaceParticle::new(maximum_displacement, target_acceptance, nparticles);
        Self {
            _data: Rc::new(RefCell::new(mv)),
        }
    }

    #[staticmethod]
    fn insert_delete_particle(chemical_potential: f64) -> Self {
        let mv = InsertDeleteParticle::new(chemical_potential);
        Self {
            _data: Rc::new(RefCell::new(mv)),
        }
    }
}
