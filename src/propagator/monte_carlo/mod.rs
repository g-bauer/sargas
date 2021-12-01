use super::{Propagator};
use std::cell::RefCell;
use std::rc::Rc;
pub mod change_volume;
pub mod displace_particle;
use crate::error::SargasError;
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
    fn propagate(&mut self, system: &mut System) -> Result<(), SargasError> {
        self.moves[self.weights.sample(&mut self.rng)]
            .as_ref()
            .borrow_mut()
            .apply(system);
        Ok(())
    }

    fn adjust(&mut self, system: &mut System) {
        self.moves
            .iter()
            .for_each(|m| m.as_ref().borrow_mut().adjust(&system))
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::change_volume::ChangeVolume;
    use super::displace_particle::DisplaceParticle;
    use super::insert_particle::InsertDeleteParticle;
    use super::*;
    use pyo3::prelude::*;

    /// Metropolis Monte-Carlo propagator.
    ///
    /// Each step, picks a move according to its weight.
    /// A move is accepted according to the Metropolis acceptance criterion.
    ///
    /// Parameters
    /// ----------
    /// moves : List[MCMove]
    ///     the moves used to change the system
    /// weights : List[int]
    ///     the weights for each move.
    ///     Weights are automatically normalized.
    ///     E.g. consider "move 1" and "move 2" with weights=[1, 3].
    ///     The probability of picking "move 1" is
    ///     0.25 while the probability of "move 2" is 0.75.
    /// temperature : float
    ///     reduced temperature
    ///
    /// Returns
    /// -------
    /// MonteCarlo : Metropolis Monte-Carlo propagator.
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
        /// Randomly choose a particle and translate it.
        ///
        /// Parameters
        /// ----------
        /// maximum_displacement : float
        ///     the maximum distance a particle is moved
        /// target_acceptance : float
        ///     probability with which move should be accepted.
        ///     Must be between 0 and 1.
        /// temperature : float
        ///     reduced temperature
        ///
        /// Returns
        /// -------
        /// McMove : the Monte-Carlo move to displace a particle.
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
