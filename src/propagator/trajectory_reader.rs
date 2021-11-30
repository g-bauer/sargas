use super::{Propagator, PropagatorError};
use crate::vec::Vec3;
use chemfiles::{Error, Frame, Trajectory};
use std::path::Path;

pub struct TrajectoryReader {
    trajectory: Trajectory,
    nsteps: usize,
    current_step: usize,
}

impl TrajectoryReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
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
    fn propagate(&mut self, system: &mut crate::system::System) -> Result<(), PropagatorError> {
        let mut frame = Frame::new();
        self.trajectory
            .read_step(self.current_step, &mut frame)
            .unwrap();
        self.current_step += 1;
        let positions = frame.positions().iter().map(Vec3::from).collect();
        system.configuration.positions = positions;
        let velocities = frame.velocities().iter().map(Vec3::from).collect();
        system.configuration.velocities = Some(velocities);
        Ok(())
    }

    fn adjust(&mut self, _system: &mut crate::system::System) {}
}
