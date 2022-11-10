use super::Propagator;
use crate::error::SargasError;
use crate::system::System;
use crate::{configuration::Configuration, vec::Vec3};
use chemfiles::{Frame, Trajectory};
use std::fmt::Display;
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

impl Display for TrajectoryReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Trajectory Reader\n===============\n")?;
        write!(f, "  trajectory:      {}\n", self.trajectory.path())?;
        write!(f, "  number of steps: {}\n", self.nsteps)
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
        let velocities = if let Some(velocities) = frame.velocities() {
            Some(velocities.iter().map(Vec3::from).collect())
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

    /// Reads frames of a trajectory instead of propagating the system.
    ///
    /// Parameters
    /// ----------
    /// path : string
    ///     path to trajectory
    ///
    /// Returns
    /// -------
    /// TrajectoryReader
    #[pyclass(name = "TrajectoryReader", unsendable)]
    #[derive(Clone)]
    #[pyo3(text_signature = "(path)")]
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
