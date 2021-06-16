use super::Propagator;
use std::cell::RefCell;
use std::rc::Rc;
pub mod change_volume;
pub mod displace_particle;
use crate::system::System;
pub mod insert_particle;
use rand::distributions::WeightedIndex;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::Distribution;

const MAXIMUM_BOLTZMANN_FACTOR: f64 = 75.0;

pub enum MoveProposal {
    Accepted,
    Rejected,
}

pub fn metropolis<R: Rng + ?Sized>(boltzmann_factor: f64, rng: &mut R) -> MoveProposal {
    match boltzmann_factor {
        f if f > MAXIMUM_BOLTZMANN_FACTOR => MoveProposal::Rejected,
        f if f.is_infinite() => MoveProposal::Rejected,
        f if f < 0.0 => MoveProposal::Accepted,
        f => {
            if rng.gen::<f64>() < (-f).exp() {
                MoveProposal::Accepted
            } else {
                MoveProposal::Rejected
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

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;
    use super::displace_particle::DisplaceParticle;
    use super::change_volume::ChangeVolume;
    use super::insert_particle::InsertDeleteParticle;

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
            temperature: f64,
        ) -> Self {
            let mv = DisplaceParticle::new(maximum_displacement, target_acceptance, temperature);
            Self {
                _data: Rc::new(RefCell::new(mv)),
            }
        }

        #[staticmethod]
        fn change_volume(
            maximum_displacement: f64,
            target_acceptance: f64,
            pressure: f64,
            temperature: f64,
        ) -> Self {
            let mv = ChangeVolume::new(
                maximum_displacement,
                target_acceptance,
                pressure,
                temperature,
            );
            Self {
                _data: Rc::new(RefCell::new(mv)),
            }
        }

        #[staticmethod]
        fn insert_delete_particle(chemical_potential: f64, temperature: f64) -> Self {
            let mv = InsertDeleteParticle::new(chemical_potential, temperature);
            Self {
                _data: Rc::new(RefCell::new(mv)),
            }
        }
    }
}
