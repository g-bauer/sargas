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
    pub velocities: Option<Vec<Vec3>>,
    pub forces: Vec<Vec3>,
    pub box_length: f64,
    pub max_nparticles: usize,
}

impl Configuration {
    pub fn new(positions: Vec<Vec3>, velocities: Option<Vec<Vec3>>, box_length: f64) -> Self {
        let nparticles = positions.len();
        Self {
            nparticles,
            positions,
            velocities,
            forces: Vec::with_capacity(nparticles),
            box_length,
            max_nparticles: nparticles,
        }
    }

    pub fn lattice(
        nparticles: usize,
        density: f64,
        max_nparticles: usize,
        initial_temperature: Option<f64>,
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
        let velocities = if let Some(t) = initial_temperature {
            Some(maxwell_boltzmann(t, nparticles))
        } else {
            None
        };
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

    pub fn without_particles() -> Self {
        Self {
            nparticles: 0,
            positions: vec![],
            velocities: None,
            forces: vec![],
            box_length: 0.0,
            max_nparticles: 0,
        }
    }

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
            Some(frame.velocities().into_iter().map(Vec3::from).collect())
        } else {
            None
        };
        frame.set_cell(&UnitCell::new([box_length, box_length, box_length]));
        let box_length = frame.cell().lengths()[0];
        let forces: Vec<Vec3> = (0..nparticles).map(|_| Vec3::zero()).collect();
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

    pub fn velocities(&self) -> Option<Array2<f64>> {
        if let Some(v) = self.velocities.as_ref() {
            Some(Array2::from_shape_fn((self.nparticles, 3), |(i, j)| {
                v[i][j]
            }))
        } else {
            None
        }
    }

    pub fn temperature_from_velocities(&self) -> Option<f64> {
        let squared_veloicty = self
            .velocities
            .as_ref()
            .map(|v| v.iter().fold(0.0, |acc, vi| acc + vi.dot(vi)));
        squared_veloicty.map(|v2| v2 / (3.0 * self.nparticles as f64))
    }

    pub fn kinetic_energy_from_velocities(&self) -> Option<f64> {
        let squared_veloicty = self
            .velocities
            .as_ref()
            .map(|v| v.iter().fold(0.0, |acc, vi| acc + vi.dot(vi)));
        squared_veloicty.map(|v2| 0.5 * v2)
    }

    pub fn forces(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.nparticles, 3), |(i, j)| self.forces[i][j])
    }
}

pub fn maxwell_boltzmann(temperature: f64, nparticles: usize) -> Vec<Vec3> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rescale_box_length() {
        let mut configuration = Configuration {
            nparticles: 1,
            positions: vec![Vec3::new(1.0, 1.0, 1.0)],
            velocities: None,
            forces: vec![Vec3::new(1.0, 0.0, 0.0)],
            box_length: 3.0,
            max_nparticles: 1,
        };
        configuration.rescale_box_length(6.0);
        assert_relative_eq!(configuration.positions[0].x, 2.0);
        assert_relative_eq!(configuration.positions[0].y, 2.0);
        assert_relative_eq!(configuration.positions[0].z, 2.0);
        assert_relative_eq!(configuration.volume(), 6.0f64.powi(3));
        configuration.rescale_box_length(3.0);
        assert_relative_eq!(configuration.positions[0].x, 1.0);
        assert_relative_eq!(configuration.positions[0].y, 1.0);
        assert_relative_eq!(configuration.positions[0].z, 1.0);
        assert_relative_eq!(configuration.volume(), 3.0f64.powi(3));
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::PyObjectProtocol;

    #[pyclass(name = "Configuration", unsendable)]
    #[derive(Clone)]
    pub struct PyConfiguration {
        pub _data: Configuration,
    }

    #[pymethods]
    impl PyConfiguration {
        /// Place particles on a lattice.
        ///
        /// Parameters
        /// ----------
        /// nparticles : int
        ///     number of particles
        /// density : float
        ///     reduced density
        /// max_nparticles : int, optional
        ///     maximum number of particles.
        ///     Defaults to nparticles if not provided.
        ///
        /// Returns
        /// -------
        /// Configuration
        #[staticmethod]
        fn lattice(
            nparticles: usize,
            density: f64,
            max_nparticles: Option<usize>,
            initial_temperature: Option<f64>,
        ) -> Self {
            Self {
                _data: Configuration::lattice(
                    nparticles,
                    density,
                    max_nparticles.unwrap_or(nparticles),
                    initial_temperature,
                ),
            }
        }
    }

    #[pyproto]
    impl PyObjectProtocol for PyConfiguration {
        fn __repr__(&self) -> PyResult<String> {
            Ok(fmt::format(format_args!("{}\n", self._data.to_string())))
        }
    }
}
