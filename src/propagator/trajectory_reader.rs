use super::{Propagator};
use crate::error::SargasError;
use crate::system::System;
use crate::{configuration::Configuration, vec::Vec3};
use chemfiles::{Frame, Trajectory};
use std::path::Path;

pub struct TrajectoryReader {
    trajectory: Trajectory,
    nsteps: usize,
    current_step: usize,
}

impl TrajectoryReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, SargasError> {
        let mut trajectory = Trajectory::open(path, 'r')?;
        let nsteps = trajectory.nsteps();
        Ok(Self {
            trajectory,
            nsteps,
            current_step: 0,
        })
    }
}

impl Propagator for TrajectoryReader {
    fn propagate(&mut self, system: &mut System) -> Result<(), SargasError> {
        let mut frame = Frame::new();
        self.current_step += 1;
        if self.current_step == self.nsteps {
            return Err(SargasError::TrajectoryEnd);
        }
        self.trajectory.read_step(self.current_step, &mut frame)?;
        let box_length = frame.cell().lengths()[0];
        let positions = frame.positions().iter().map(Vec3::from).collect();
        let velocities = if frame.has_velocities() {
            Some(frame.velocities().iter().map(Vec3::from).collect())
        } else {
            None
        };
        let configuration = Configuration::new(positions, velocities, box_length);
        system.configuration = configuration;
        system.recompute_energy_forces();
        Ok(())
    }

    fn adjust(&mut self, _system: &mut System) {}
}

#[cfg(feature = "python")]
pub mod python {
    use super::TrajectoryReader;
    use pyo3::prelude::*;
    use std::cell::RefCell;
    use std::rc::Rc;

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
    /// TrajectoryReader
    #[pyclass(name = "TrajectoryReader", unsendable)]
    #[derive(Clone)]
    pub struct PyTrajectoryReader {
        pub _data: Rc<RefCell<TrajectoryReader>>,
    }

    #[pymethods]
    impl PyTrajectoryReader {
        #[new]
        fn new(path: String) -> PyResult<Self> {
            Ok(Self {
                _data: Rc::new(RefCell::new(TrajectoryReader::new(path)?)),
            })
        }
    }
}
