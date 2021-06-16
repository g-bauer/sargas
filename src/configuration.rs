use crate::vec::Vec3;
use chemfiles::{Frame, Trajectory, UnitCell};
use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use std::fmt;
use std::path::Path;

#[derive(Clone)]
pub struct Configuration {
    pub nparticles: usize,
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub box_length: f64,
    pub max_nparticles: usize,
}

impl Configuration {
    pub fn with_lattice(nparticles: usize, density: f64, max_nparticles: usize) -> Self {
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

        Self {
            nparticles,
            positions,
            velocities,
            forces,
            box_length,
            max_nparticles: max_nparticles,
        }
    }

    // pub fn insert_particles(
    //     nparticles: usize,
    //     volume: f64,
    //     temperature: f64,
    //     chemical_potential: f64,
    //     rc: f64,
    //     potential: Rc<dyn Potential>,
    //     max_nparticles: usize,
    //     insertion_tries: Option<usize>,
    // ) -> Result<Self, String> {
    //     let insertions = insertion_tries.unwrap_or(10000);
    //     let positions = Vec::with_capacity(max_nparticles);
    //     let velocities = Vec::with_capacity(max_nparticles);
    //     let forces = Vec::with_capacity(max_nparticles);
    //     let box_length = volume.cbrt();
    //     let mut system = Self {
    //         nparticles: 0,
    //         positions,
    //         velocities,
    //         forces,
    //         box_length,
    //         energy: 0.0,
    //         virial: 0.0,
    //         kinetic_energy: 0.0,
    //         max_nparticles: max_nparticles,
    //         compute_forces: true,
    //     };
    //     let mut mv = InsertDeleteParticle::new(chemical_potential);
    //     for _ in 0..insertions {
    //         mv.insert_particle(&mut system);
    //         if system.nparticles == nparticles {
    //             system.reset();
    //             return Ok(system);
    //         }
    //         // println!("try: {}, n: {}", i, system.nparticles);
    //     }
    //     Err(format!("Could not insert all particles (currently: {}). Try to increase number of insertion tries.", system.nparticles))
    // }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        box_length: f64,
        max_nparticles: usize,
    ) -> Result<Self, String> {
        let mut trajectory = Trajectory::open(&path, 'r').unwrap();
        let mut frame = Frame::new();
        trajectory.read_step(0, &mut frame).unwrap();
        let nparticles = frame.size();
        let positions = frame.positions().into_iter().map(Vec3::from).collect();
        let velocities = if frame.has_velocities() {
            frame.velocities().into_iter().map(Vec3::from).collect()
        } else {
            Vec::with_capacity(max_nparticles)
        };
        frame.set_cell(&UnitCell::new([box_length, box_length, box_length]));
        let box_length = frame.cell().lengths()[0];
        let forces = Vec::with_capacity(max_nparticles);
        Ok(Self {
            nparticles,
            positions,
            velocities,
            forces,
            box_length,
            max_nparticles: max_nparticles,
        })
    }
}

impl Configuration {
    #[inline]
    pub fn nearest_image(&self, r: &mut Vec3) {
        let il = 1.0 / self.box_length;
        r.x -= self.box_length * f64::round(r.x * il);
        r.y -= self.box_length * f64::round(r.y * il);
        r.z -= self.box_length * f64::round(r.z * il);
    }

    #[inline]
    pub fn rescale_box_length(&mut self, box_length_new: f64) {
        let s = box_length_new / self.box_length;
        self.positions.iter_mut().for_each(|r| *r *= s);
        self.box_length = box_length_new;
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

impl Configuration {
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

impl fmt::Display for Configuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let density = self.nparticles as f64 / self.box_length.powi(3);
        write!(
            f,
            "System\n======\nparticles: {:.3}\ndensity: {:.3}",
            self.nparticles, density
        )
    }
}

#[cfg(feature = "python")]
pub mod python {
    use numpy::{IntoPyArray, PyArray2};
    use pyo3::prelude::*;
    use pyo3::PyObjectProtocol;
    use super::*;

    #[pyclass(name = "Configuration", unsendable)]
    #[derive(Clone)]
    pub struct PyConfiguration {
        pub _data: Configuration,
    }

    #[pymethods]
    impl PyConfiguration {
        #[staticmethod]
        fn from_lattice(nparticles: usize, density: f64, max_nparticles: Option<usize>) -> Self {
            Self {
                _data: Configuration::with_lattice(
                    nparticles,
                    density,
                    max_nparticles.unwrap_or(nparticles),
                ),
            }
        }

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
        fn get_box_length(&self) -> f64 {
            self._data.box_length
        }

        #[getter]
        fn get_nparticles(&self) -> usize {
            self._data.nparticles
        }

        #[getter]
        fn get_positions<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self._data.positions().into_pyarray(py)
        }

        #[getter]
        fn get_velocities<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self._data.velocities().into_pyarray(py)
        }

        #[getter]
        fn get_forces<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self._data.forces().into_pyarray(py)
        }
    }

    #[pyproto]
    impl PyObjectProtocol for PyConfiguration {
        fn __repr__(&self) -> PyResult<String> {
            Ok(fmt::format(format_args!("{}\n", self._data.to_string())))
        }
    }
}
