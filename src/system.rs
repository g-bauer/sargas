use crate::potential::{Potential, PyPotential};
use crate::propagator::monte_carlo::InsertDeleteParticle;
use crate::vec::Vec3;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

#[derive(Clone)]
pub struct System {
    pub nparticles: usize,
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub potential: Rc<dyn Potential>,
    pub box_length: f64,
    pub temperature: f64,
    pub beta: f64,
    pub energy: f64,
    pub virial: f64,
    pub kinetic_energy: f64,
    pub rc: f64,
    pub rc2: f64,
    pub max_nparticles: usize,
    compute_forces: bool,
}

impl System {
    pub fn with_lattice(
        nparticles: usize,
        density: f64,
        temperature: f64,
        rc: f64,
        potential: Rc<dyn Potential>,
        max_nparticles: usize,
    ) -> Self {
        let mut positions: Vec<Vec3> = (0..nparticles).map(|_| Vec3::zero()).collect();
        let box_length = f64::cbrt(nparticles as f64 / density);
        let mut n = (nparticles as f64).powf(1.0 / 3.0) as usize + 1;
        if n == 0 {
            n = 1
        };
        let del = box_length / n as f64;
        let mut itel = 0;
        let mut dx = -del;
        for _ in 0..n {
            dx += del;
            let mut dy = -del;
            for _ in 0..n {
                dy += del;
                let mut dz = -del;
                for _ in 0..n {
                    dz += del;
                    if itel < nparticles {
                        positions[itel] = [dx, dy, dz].into();
                        itel += 1;
                    }
                }
            }
        }
        let velocities: Vec<Vec3> = (0..nparticles).map(|_| Vec3::zero()).collect();
        let forces: Vec<Vec3> = (0..nparticles).map(|_| Vec3::zero()).collect();

        let mut system = Self {
            nparticles,
            positions,
            velocities,
            forces,
            potential,
            box_length,
            temperature,
            beta: 1.0 / temperature,
            energy: 0.0,
            virial: 0.0,
            kinetic_energy: 0.0,
            rc,
            rc2: rc * rc,
            max_nparticles: max_nparticles,
            compute_forces: true,
        };
        system.reset();
        system
    }

    pub fn insert_particles(
        nparticles: usize,
        volume: f64,
        temperature: f64,
        chemical_potential: f64,
        rc: f64,
        potential: Rc<dyn Potential>,
        max_nparticles: usize,
        insertion_tries: Option<usize>,
    ) -> Result<Self, String> {
        let insertions = insertion_tries.unwrap_or(10000);
        let positions = Vec::with_capacity(max_nparticles);
        let velocities = Vec::with_capacity(max_nparticles);
        let forces = Vec::with_capacity(max_nparticles);
        let box_length = volume.cbrt();
        let mut system = Self {
            nparticles: 0,
            positions,
            velocities,
            forces,
            potential,
            box_length,
            temperature,
            beta: 1.0 / temperature,
            energy: 0.0,
            virial: 0.0,
            kinetic_energy: 0.0,
            rc,
            rc2: rc * rc,
            max_nparticles: max_nparticles,
            compute_forces: true,
        };
        let mut mv = InsertDeleteParticle::new(chemical_potential);
        for _ in 0..insertions {
            mv.insert_particle(&mut system);
            if system.nparticles == nparticles {
                system.reset();
                return Ok(system);
            }
            // println!("try: {}, n: {}", i, system.nparticles);
        }
        Err(format!("Could not insert all particles (currently: {}). Try to increase number of insertion tries.", system.nparticles))
    }
}

impl System {
    #[inline]
    fn nearest_image(&self, r: &mut Vec3) {
        let il = 1.0 / self.box_length;
        r.x -= self.box_length * f64::round(r.x * il);
        r.y -= self.box_length * f64::round(r.y * il);
        r.z -= self.box_length * f64::round(r.z * il);
    }

    pub fn particle_energy(&self, i: usize, start_idx: Option<usize>) -> f64 {
        if self.nparticles == 0 {
            return 0.0;
        }
        let position_i = self.positions[i];
        let mut energy = 0.0;
        for j in start_idx.unwrap_or(0)..self.nparticles {
            let uij = if i != j {
                let rij = (self.positions[j] - position_i).nearest_image(self.box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.rc2 {
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
        if self.nparticles == 0 {
            return (0.0, 0.0);
        }
        let position_i = self.positions[i];
        let mut energy = 0.0;
        let mut virial = 0.0;
        for j in start_idx.unwrap_or(0)..self.nparticles {
            let (uij, vij) = if i != j {
                let rij = (self.positions[j] - position_i).nearest_image(self.box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.rc2 {
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

    pub fn ghost_particle_energy(&self) -> f64 {
        if self.nparticles == 0 {
            return 0.0;
        }
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, self.box_length);
        let ri = Vec3::new(
            dist.sample(&mut rng),
            dist.sample(&mut rng),
            dist.sample(&mut rng),
        );
        self.positions.iter().fold(0.0, |energy, rj| {
            energy + {
                let rij = (rj - ri).nearest_image(self.box_length);
                let r2 = rij.dot(&rij);
                if r2 <= self.rc2 {
                    self.potential.energy(r2)
                } else {
                    0.0
                }
            }
        })
    }

    pub fn energy(&self) -> f64 {
        if self.nparticles == 0 {
            return 0.0;
        }
        (0..self.nparticles - 1).fold(0.0, |energy, i| {
            energy + self.particle_energy(i, Some(i + 1))
        })
    }

    pub fn energy_virial(&self) -> (f64, f64) {
        if self.nparticles == 0 {
            return (0.0, 0.0);
        }
        (0..self.nparticles - 1).fold((0.0, 0.0), |(energy, virial), i| {
            let (u, v) = self.particle_energy_virial(i, Some(i + 1));
            (energy + u, virial + v)
        })
    }

    pub fn reset(&mut self) {
        self.energy = 0.0;
        self.virial = 0.0;
        self.forces.iter_mut().for_each(|fi| *fi = Vec3::zero());
        self.velocities.iter_mut().for_each(|vi| *vi = Vec3::zero());
        if self.nparticles == 0 {
            return;
        }
        let (e, v) = self.energy_virial();
        self.energy = e;
        self.virial = v;
        self.forces = self.compute_forces();
        self.velocities = maxwell_boltzmann(self.temperature, self.nparticles);
    }

    pub fn compute_forces(&self) -> Vec<Vec3> {
        let mut forces: Vec<Vec3> = (0..self.nparticles).map(|_| Vec3::zero()).collect();
        for i in 0..self.nparticles - 1 {
            let position_i = self.positions[i];
            for j in i + 1..self.nparticles {
                let mut rij = self.positions[j] - position_i;
                self.nearest_image(&mut rij);
                let r2 = rij.dot(&rij);
                if r2 <= self.rc2 {
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
        self.box_length.powi(3)
    }

    #[inline]
    pub fn density(&self) -> f64 {
        self.nparticles as f64 / self.volume()
    }
}

impl System {
    pub fn positions(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.nparticles, 3), |(i, j)| self.positions[i][j])
    }

    pub fn velocities(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.nparticles, 3), |(i, j)| self.velocities[i][j])
    }

    pub fn forces(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.nparticles, 3), |(i, j)| self.forces[i][j])
    }
}

fn maxwell_boltzmann(temperature: f64, nparticles: usize) -> Vec<Vec3> {
    let normal = Normal::new(0.0, temperature.sqrt()).unwrap();
    let mut rng = rand::thread_rng();
    (0..nparticles)
        .map(|_| {
            Vec3::new(
                normal.sample(&mut rng),
                normal.sample(&mut rng),
                normal.sample(&mut rng),
            )
        })
        .collect()
}

impl fmt::Display for System {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let density = self.nparticles as f64 / self.box_length.powi(3);
        write!(
            f,
            "System\n======\nparticles: {}\ntemperature: {:.3}\ndensity: {:.3}",
            self.nparticles, self.temperature, density
        )
    }
}

#[pyclass(name = "System", unsendable)]
#[derive(Clone)]
pub struct PySystem {
    pub _data: Rc<RefCell<System>>,
}

#[pymethods]
impl PySystem {
    #[staticmethod]
    fn from_lattice(
        nparticles: usize,
        density: f64,
        temperature: f64,
        rc: f64,
        potential: PyPotential,
        max_nparticles: Option<usize>,
    ) -> Self {
        Self {
            _data: Rc::new(RefCell::new(System::with_lattice(
                nparticles,
                density,
                temperature,
                rc,
                potential._data.clone(),
                max_nparticles.unwrap_or(nparticles),
            ))),
        }
    }

    #[staticmethod]
    fn insert_particles(
        nparticles: usize,
        volume: f64,
        temperature: f64,
        chemical_potential: f64,
        rc: f64,
        potential: PyPotential,
        max_nparticles: Option<usize>,
        insertion_tries: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            _data: Rc::new(RefCell::new(
                System::insert_particles(
                    nparticles,
                    volume,
                    temperature,
                    chemical_potential,
                    rc,
                    potential._data.clone(),
                    max_nparticles.unwrap_or(nparticles),
                    insertion_tries,
                )
                .map_err(|e| PyRuntimeError::new_err(e))?,
            )),
        })
    }

    #[getter]
    fn get_energy(&self) -> f64 {
        self._data.borrow().energy
    }

    fn compute_energy(&self) -> f64 {
        self._data.borrow().energy()
    }

    fn compute_energy_virial(&self) -> (f64, f64) {
        self._data.borrow().energy_virial()
    }

    #[getter]
    fn get_box_length(&self) -> f64 {
        self._data.borrow().box_length
    }

    #[getter]
    fn get_temperature(&self) -> f64 {
        self._data.borrow().temperature
    }

    #[getter]
    fn get_nparticles(&self) -> usize {
        self._data.borrow().nparticles
    }

    #[getter]
    fn get_positions<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._data.borrow().positions().into_pyarray(py)
    }

    #[getter]
    fn get_velocities<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._data.borrow().velocities().into_pyarray(py)
    }

    #[getter]
    fn get_forces<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._data.borrow().forces().into_pyarray(py)
    }

    fn compute_forces<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let f = self._data.borrow().compute_forces();
        Array2::from_shape_fn((self._data.borrow().nparticles, 3), |(i, j)| f[i][j])
            .into_pyarray(py)
    }
}

#[pyproto]
impl PyObjectProtocol for PySystem {
    fn __repr__(&self) -> PyResult<String> {
        Ok(fmt::format(format_args!(
            "{}\n",
            self._data.borrow().to_string()
        )))
    }
}
