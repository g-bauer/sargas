use crate::configuration::{maxwell_boltzmann, Configuration};
use crate::potential::Potential;
use crate::vec::Vec3;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use ndarray::{Array1, Array2};

use rand::{distributions::Uniform, thread_rng};
use rand_distr::Distribution;
use std::cell::RefCell;
use std::rc::Rc;

pub struct System {
    pub configuration: Configuration,
    pub potential: Rc<dyn Potential>,
    pub potential_energy: f64,
    pub virial: f64,
    pub kinetic_energy: Option<f64>,
}

impl System {
    pub fn new(configuration: Configuration, potential: Rc<dyn Potential>) -> Self {
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
        system
    }

    pub fn particle_energy(&self, i: usize, start_idx: Option<usize>) -> f64 {
        if self.configuration.nparticles == 0 {
            return 0.0;
        }
        let position_i = self.configuration.positions[i];
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

    pub fn particle_energy_virial(&self, i: usize, start_idx: Option<usize>) -> (f64, f64) {
        if self.configuration.nparticles == 0 {
            return (0.0, 0.0);
        }
        let position_i = self.configuration.positions[i];
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

    pub fn ghost_particle_energy_sum(&self, beta: f64, ninsertions: usize) -> f64 {
        if self.configuration.nparticles == 0 {
            return 0.0;
        }
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, self.configuration.box_length);

        (0..ninsertions).fold(0.0, |acc, _| {
            let ri = Vec3::new(
                dist.sample(&mut rng),
                dist.sample(&mut rng),
                dist.sample(&mut rng),
            );
            let energy = self.configuration.positions.iter().fold(0.0, |energy, rj| {
                energy + {
                    let rij = (rj - ri).nearest_image(self.configuration.box_length);
                    let r2 = rij.dot(&rij);
                    if r2 <= self.potential.rc2() {
                        self.potential.energy(r2)
                    } else {
                        0.0
                    }
                }
            });
            acc + (-beta * energy).exp()
        })
    }

    pub fn energy(&self) -> f64 {
        if self.configuration.nparticles == 0 {
            return 0.0;
        }
        let u_tail = self
            .potential
            .energy_tail(self.configuration.density(), self.configuration.nparticles);
        (0..self.configuration.nparticles - 1)
            .fold_while(u_tail, |energy, i| {
                match self.particle_energy(i, Some(i + 1)) {
                    u if u.is_finite() => Continue(energy + u),
                    _ => Done(f64::INFINITY),
                }
            })
            .into_inner()
    }

    pub fn energy_virial(&self) -> (f64, f64) {
        if self.configuration.nparticles == 0 {
            return (0.0, 0.0);
        }
        let u_tail = self
            .potential
            .energy_tail(self.configuration.density(), self.configuration.nparticles);
        (0..self.configuration.nparticles - 1)
            .fold_while((u_tail, 0.0), |(energy, virial), i| {
                match self.particle_energy_virial(i, Some(i + 1)) {
                    (u, v) if u.is_finite() => Continue((energy + u, virial + v)),
                    _ => Done((f64::INFINITY, f64::INFINITY)),
                }
            })
            .into_inner()
    }

    pub fn reset(&mut self) {
        self.potential_energy = 0.0;
        self.virial = 0.0;
        self.configuration
            .forces
            .iter_mut()
            .for_each(|fi| *fi = Vec3::zero());

        // use current temperature from kinetic energy to set velocities from scratch
        if let Some(t) = self.configuration.temperature_from_velocities() {
            self.configuration.velocities =
                Some(maxwell_boltzmann(t, self.configuration.nparticles));
            self.kinetic_energy = Some(3.0 / 2.0 * t);
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

    pub fn recompute(&mut self) {
        self.potential_energy = 0.0;
        self.virial = 0.0;
        if self.configuration.nparticles == 0 {
            return;
        }
        let (e, v) = self.energy_virial();
        self.potential_energy = e;
        self.virial = v;
        self.compute_forces_inplace();
        if let Some(t) = self.configuration.temperature_from_velocities() {
            self.kinetic_energy = Some(3.0 / 2.0 * t);
        } else {
            self.kinetic_energy = None;
        }
    }

    pub fn compute_forces_inplace(&mut self) {
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

    #[inline]
    pub fn volume(&self) -> f64 {
        self.configuration.volume()
    }

    #[inline]
    pub fn density(&self) -> f64 {
        self.configuration.density()
    }

    #[inline]
    pub fn residual_pressure(&self) -> f64 {
        self.virial / (3.0 * self.volume()) + self.potential.pressure_tail(self.density())
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::configuration::python::PyConfiguration;
    use crate::potential::python::PyPotential;
    use numpy::{IntoPyArray, PyArray2};
    use pyo3::prelude::*;

    #[pyclass(name = "System", unsendable)]
    #[derive(Clone)]
    pub struct PySystem {
        pub _data: Rc<RefCell<System>>,
    }

    #[pymethods]
    impl PySystem {
        #[new]
        fn new(configuration: PyConfiguration, potential: PyPotential) -> Self {
            Self {
                _data: Rc::new(RefCell::new(System::new(
                    configuration._data,
                    potential._data,
                ))),
            }
        }
        // #[staticmethod]
        // fn from_lattice(
        //     nparticles: usize,
        //     density: f64,
        //     temperature: f64,
        //     rc: f64,
        //     potential: PyPotential,
        //     max_nparticles: Option<usize>,
        // ) -> Self {
        //     Self {
        //         _data: Rc::new(RefCell::new(Configuration::with_lattice(
        //             nparticles,
        //             density,
        //             max_nparticles.unwrap_or(nparticles),
        //         ))),
        //     }
        // }

        // #[staticmethod]
        // fn insert_particles(
        //     nparticles: usize,
        //     volume: f64,
        //     temperature: f64,
        //     chemical_potential: f64,
        //     rc: f64,
        //     potential: PyPotential,
        //     max_nparticles: Option<usize>,
        //     insertion_tries: Option<usize>,
        // ) -> PyResult<Self> {
        //     Ok(Self {
        //         _data: Rc::new(RefCell::new(
        //             Configuration::insert_particles(
        //                 nparticles,
        //                 volume,
        //                 temperature,
        //                 chemical_potential,
        //                 rc,
        //                 potential._data.clone(),
        //                 max_nparticles.unwrap_or(nparticles),
        //                 insertion_tries,
        //             )
        //             .map_err(|e| PyRuntimeError::new_err(e))?,
        //         )),
        //     })
        // }

        #[getter]
        fn get_energy(&self) -> f64 {
            self._data.as_ref().borrow().potential_energy
        }

        #[getter]
        fn get_virial(&self) -> f64 {
            self._data.as_ref().borrow().virial
        }

        fn compute_energy(&self) -> f64 {
            self._data.as_ref().borrow().energy()
        }

        fn compute_energy_virial(&self) -> (f64, f64) {
            self._data.as_ref().borrow().energy_virial()
        }

        /// Compute kinetic energy from velocities.
        pub fn kinetic_energy_from_velocities(&self) -> Option<f64> {
            self._data
                .as_ref()
                .borrow()
                .configuration
                .kinetic_energy_from_velocities()
        }

        /// Compute temperature from kinetic energy.
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
        fn get_velocities<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self._data
                .as_ref()
                .borrow()
                .configuration
                .velocities()
                .unwrap()
                .into_pyarray(py)
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

        fn compute_forces<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            let f = self._data.as_ref().borrow().compute_forces();
            Array2::from_shape_fn(
                (self._data.as_ref().borrow().configuration.nparticles, 3),
                |(i, j)| f[i][j],
            )
            .into_pyarray(py)
        }

        #[getter]
        fn get_configuration(&self) -> PyConfiguration {
            PyConfiguration {
                _data: self._data.as_ref().borrow().configuration.clone()
            }
        }
    }

    // #[pyproto]
    // impl PyObjectProtocol for PySystem {
    //     fn __repr__(&self) -> PyResult<String> {
    //         Ok(fmt::format(format_args!(
    //             "{}\n",
    //             self._data.to_string()
    //         )))
    //     }
    // }
}
