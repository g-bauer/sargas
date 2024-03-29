use crate::configuration::{maxwell_boltzmann, Configuration};
use crate::error::SargasError;
use crate::lennard_jones::LennardJones;
use crate::vec::Vec3;
use ndarray::{Array1, Array2};

use rand::{distributions::Uniform, thread_rng};
use rand_distr::Distribution;
use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

pub struct System {
    pub configuration: Configuration,
    pub potential: LennardJones,
    pub potential_energy: f64,
    pub virial: f64,
    pub kinetic_energy: Option<f64>,
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "System Information\n==================\n")?;
        write!(
            f,
            "  number of particles: {}\n",
            self.configuration.nparticles
        )?;
        write!(
            f,
            "  box length:          {}\n",
            self.configuration.box_length
        )?;
        write!(
            f,
            "  volume:              {}\n",
            self.configuration.volume()
        )?;
        write!(
            f,
            "  density:             {}\n",
            self.configuration.density()
        )
    }
}

impl System {
    pub fn new(
        configuration: Configuration,
        potential: LennardJones,
    ) -> Result<Self, SargasError> {
        if potential.rc2().sqrt() > 0.5 * configuration.box_length && configuration.nparticles != 0
        {
            return Err(SargasError::InvalidCutoff {
                found: potential.rc2().sqrt(),
                maximum: 0.5 * configuration.box_length,
            });
        }
        let mut system = Self {
            configuration,
            potential,
            potential_energy: 0.0,
            kinetic_energy: None,
            virial: 0.0,
        };
        let (u, v) = system.energy_virial();
        system.potential_energy = u;
        system.virial = v;
        system.compute_forces_inplace();
        system.kinetic_energy = system.configuration.kinetic_energy_from_velocities();
        Ok(system)
    }

    /// Calculate the energy of particle `i`.
    ///
    /// If a `start_idx` is provided, only particles with
    /// index larger or equal to the `start_idx` are considered.
    pub fn particle_energy(&self, i: usize, position_i: &Vec3, start_idx: Option<usize>) -> f64 {
        if self.configuration.nparticles == 0 {
            return 0.0;
        }
        let mut energy = 0.0;
        for j in start_idx.unwrap_or(0)..self.configuration.nparticles {
            let uij = if i != j {
                let rij = (self.configuration.positions[j] - position_i)
                    .nearest_image(self.configuration.box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.potential.rc2() {
                    if self.potential.overlaps(r2) {
                        return f64::INFINITY;
                    }
                    self.potential.energy(r2)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            energy += uij
        }
        energy
    }

    /// Calculate the energy and virial of particle `i`.
    ///
    /// If a `start_idx` is provided, only particles with
    /// index larger or equal to the `start_idx` are considered.
    pub fn particle_energy_virial(
        &self,
        i: usize,
        position_i: &Vec3,
        start_idx: Option<usize>,
    ) -> (f64, f64) {
        if self.configuration.nparticles == 0 {
            return (0.0, 0.0);
        }
        let mut energy = 0.0;
        let mut virial = 0.0;
        for j in start_idx.unwrap_or(0)..self.configuration.nparticles {
            let (uij, vij) = if i != j {
                let rij = (self.configuration.positions[j] - position_i)
                    .nearest_image(self.configuration.box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.potential.rc2() {
                    if self.potential.overlaps(r2) {
                        return (f64::INFINITY, f64::INFINITY);
                    }
                    self.potential.energy_virial(r2)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };
            energy += uij;
            virial += vij;
        }
        (energy, virial)
    }

    pub fn ghost_particle_energy_array(&self, ninsertions: usize) -> Array1<f64> {
        if self.configuration.nparticles == 0 {
            return Array1::from_elem(1, 0.0);
        }
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, self.configuration.box_length);

        Array1::from_shape_fn(ninsertions, |_| {
            let ri = Vec3::new(
                dist.sample(&mut rng),
                dist.sample(&mut rng),
                dist.sample(&mut rng),
            );
            self.configuration.positions.iter().fold(0.0, |energy, rj| {
                energy + {
                    let rij = (rj - ri).nearest_image(self.configuration.box_length);
                    let r2 = rij.dot(&rij);
                    if r2 <= self.potential.rc2() {
                        self.potential.energy(r2)
                    } else {
                        0.0
                    }
                }
            })
        })
    }

    ///
    pub fn ghost_particle_energy_sum(&self, beta: f64, ninsertions: usize) -> f64 {
        if self.configuration.nparticles == 0 {
            return 0.0;
        }
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, self.configuration.box_length);
        let mut boltzmann_factor = 0.0;
        for _ in 0..ninsertions {
            // position of ghost particle
            let ri = Vec3::new(
                dist.sample(&mut rng),
                dist.sample(&mut rng),
                dist.sample(&mut rng),
            );
            let mut energy = 0.0;
            for rj in self.configuration.positions.iter() {
                let rij = (rj - ri).nearest_image(self.configuration.box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.potential.rc2() {
                    energy += self.potential.energy(r2);
                }
            }
            boltzmann_factor += (-beta * energy).exp();
        }
        boltzmann_factor
    }

    /// Calculate the system energy.
    pub fn energy(&self) -> f64 {
        if self.configuration.nparticles == 0 {
            return 0.0;
        }
        let mut energy = self
            .potential
            .energy_tail(self.configuration.density(), self.configuration.nparticles);

        for (i, position_i) in
            (0..self.configuration.nparticles - 1).zip(self.configuration.positions.iter())
        {
            energy += match self.particle_energy(i, position_i, Some(i + 1)) {
                u if u.is_finite() => u,
                _ => return energy,
            }
        }
        energy
    }

    /// Calculate the system energy and virial.
    pub fn energy_virial(&self) -> (f64, f64) {
        if self.configuration.nparticles == 0 {
            return (0.0, 0.0);
        }
        let mut virial = 0.0;
        let mut energy = self
            .potential
            .energy_tail(self.configuration.density(), self.configuration.nparticles);

        for (i, position_i) in
            (0..self.configuration.nparticles - 1).zip(self.configuration.positions.iter())
        {
            let (u, v) = match self.particle_energy_virial(i, position_i, Some(i + 1)) {
                (u, v) if u.is_finite() => (u, v),
                _ => return (energy, virial),
            };
            energy += u;
            virial += v;
        }
        (energy, virial)
    }

    /// Sets random velocities according to current kinetic energy and
    /// computes kinetic and potential energy, virial and forces.
    pub fn reset(&mut self) {
        self.potential_energy = 0.0;
        self.virial = 0.0;
        self.kinetic_energy = Some(0.0);
        self.configuration
            .forces
            .iter_mut()
            .for_each(|fi| *fi = Vec3::zero());

        // use current temperature from kinetic energy to set velocities from scratch
        if let Some(t) = self.configuration.temperature_from_velocities() {
            self.configuration.velocities =
                Some(maxwell_boltzmann(t, self.configuration.nparticles));
            self.kinetic_energy = Some(3.0 / 2.0 * self.configuration.nparticles as f64 * t);
        } else {
            self.configuration.velocities = None;
            self.kinetic_energy = None;
        }
        if self.configuration.nparticles == 0 {
            return;
        }
        let (e, v) = self.energy_virial();
        self.potential_energy = e;
        self.virial = v;
        self.configuration.forces = self.compute_forces();
    }

    /// Compute kinetic and potential energy, virial and forces from current velocities and positions.
    pub fn recompute_energy_forces(&mut self) {
        self.potential_energy = 0.0;
        self.virial = 0.0;
        self.kinetic_energy = Some(0.0);
        if self.configuration.nparticles == 0 {
            return;
        }
        let (e, v) = self.energy_virial();
        self.potential_energy = e;
        self.virial = v;
        self.compute_forces_inplace();
        if let Some(t) = self.configuration.temperature_from_velocities() {
            self.kinetic_energy = Some(3.0 / 2.0 * self.configuration.nparticles as f64 * t);
        } else {
            self.kinetic_energy = None;
        }
    }

    /// Computes and overwrites forces between all particles in the system.
    pub fn compute_forces_inplace(&mut self) {
        if self.configuration.nparticles == 0 {
            return;
        }
        let box_length = self.configuration.box_length;
        self.configuration
            .forces
            .iter_mut()
            .for_each(|f| *f = Vec3::zero());
        for i in 0..self.configuration.nparticles - 1 {
            let position_i = self.configuration.positions[i];
            for j in (i + 1)..self.configuration.nparticles {
                let rij = (position_i - self.configuration.positions[j]).nearest_image(box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.potential.rc2() {
                    let fij = self.potential.virial(r2) / r2;
                    self.configuration.forces[i] += fij * rij;
                    self.configuration.forces[j] -= fij * rij;
                }
            }
        }
    }

    /// Computes and forces between all particles in the system.
    pub fn compute_forces(&self) -> Vec<Vec3> {
        let box_length = self.configuration.box_length;
        let mut forces: Vec<Vec3> = (0..self.configuration.nparticles)
            .map(|_| Vec3::zero())
            .collect();
        for i in 0..self.configuration.nparticles - 1 {
            let position_i = self.configuration.positions[i];
            for j in (i + 1)..self.configuration.nparticles {
                let rij = (position_i - self.configuration.positions[j]).nearest_image(box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.potential.rc2() {
                    let fij = self.potential.virial(r2) / r2;
                    forces[i] += fij * rij;
                    forces[j] -= fij * rij;
                }
            }
        }
        forces
    }

    /// System volume.
    #[inline]
    pub fn volume(&self) -> f64 {
        self.configuration.volume()
    }

    /// System density.
    #[inline]
    pub fn density(&self) -> f64 {
        self.configuration.density()
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::configuration::python::PyConfiguration;
    use crate::lennard_jones::python::PyLennardJones;
    use numpy::{IntoPyArray, PyArray2};
    use pyo3::prelude::*;

    /// A system.
    ///
    /// Parameters
    /// ----------
    /// configuration : Configuration
    ///     the configuration
    /// potential : Potential
    ///     the pair potential
    ///
    /// Returns
    /// -------
    /// System : the system.
    #[pyclass(name = "System", unsendable)]
    #[derive(Clone)]
    #[pyo3(text_signature = "(configuration, potential)")]
    pub struct PySystem {
        pub _data: Rc<RefCell<System>>,
    }

    #[pymethods]
    impl PySystem {
        #[new]
        fn new(
            configuration: PyConfiguration,
            potential: PyLennardJones,
        ) -> Result<Self, SargasError> {
            Ok(Self {
                _data: Rc::new(RefCell::new(System::new(configuration._data, potential.0)?)),
            })
        }

        #[getter]
        fn get_energy(&self) -> f64 {
            self._data.as_ref().borrow().potential_energy
        }

        #[getter]
        fn get_virial(&self) -> f64 {
            self._data.as_ref().borrow().virial
        }

        /// Compute the current system energy.
        ///
        /// Returns
        /// -------
        /// float : energy of the system in reduced units
        #[pyo3(text_signature = "($self)")]
        fn compute_energy(&self) -> f64 {
            self._data.as_ref().borrow().energy()
        }

        /// Compute the current system energy and virial.
        ///
        /// Returns
        /// -------
        /// (float, float) : energy and virial of the system
        #[pyo3(text_signature = "($self)")]
        fn compute_energy_virial(&self) -> (f64, f64) {
            self._data.as_ref().borrow().energy_virial()
        }

        /// Compute the current kinetic energy of the system
        ///
        /// The kinetic energy is computed from the velocities.
        ///
        /// Returns
        /// -------
        /// float: kinetic energy of the system
        #[pyo3(text_signature = "($self)")]
        pub fn kinetic_energy_from_velocities(&self) -> Option<f64> {
            self._data
                .as_ref()
                .borrow()
                .configuration
                .kinetic_energy_from_velocities()
        }

        /// Compute the current (kinetic) temperature of the system
        ///
        /// The temperature is computed from the kinetic energy of the system.
        ///
        /// Returns
        /// -------
        /// float: temperature of the system
        #[pyo3(text_signature = "($self)")]
        pub fn temperature(&self) -> Option<f64> {
            self._data
                .as_ref()
                .borrow()
                .configuration
                .kinetic_energy_from_velocities()
                .map(|ke| {
                    ke * 2.0 / 3.0 / self._data.as_ref().borrow().configuration.nparticles as f64
                })
        }

        #[getter]
        fn get_kinetic_energy(&self) -> Option<f64> {
            self._data.as_ref().borrow().kinetic_energy
        }

        #[getter]
        fn get_box_length(&self) -> f64 {
            self._data.as_ref().borrow().configuration.box_length
        }

        #[getter]
        fn get_nparticles(&self) -> usize {
            self._data.as_ref().borrow().configuration.nparticles
        }

        #[getter]
        fn get_positions<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self._data
                .as_ref()
                .borrow()
                .configuration
                .positions()
                .into_pyarray(py)
        }

        #[getter]
        fn get_velocities<'py>(&self, py: Python<'py>) -> Option<&'py PyArray2<f64>> {
            Some(self._data
                .as_ref()
                .borrow()
                .configuration
                .velocities()?
                .into_pyarray(py))
        }

        #[getter]
        fn get_forces<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self._data
                .as_ref()
                .borrow()
                .configuration
                .forces()
                .into_pyarray(py)
        }

        /// Compute the current forces acting on the particles of the system
        ///
        /// Returns
        /// -------
        /// numpy.ndarray[float]: forces
        #[pyo3(text_signature = "($self)")]
        fn compute_forces<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            let f = self._data.as_ref().borrow().compute_forces();
            Array2::from_shape_fn(
                (self._data.as_ref().borrow().configuration.nparticles, 3),
                |(i, j)| f[i][j],
            )
            .into_pyarray(py)
        }

        fn configuration(&self) -> PyConfiguration {
            PyConfiguration {
                _data: self._data.as_ref().borrow().configuration.clone(),
            }
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(self._data.borrow().to_string())
        }
    }
}
