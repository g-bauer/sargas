use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct LennardJones {
    /// size parameter (diameter)
    sigma: f64,
    /// energetic parameter
    epsilon: f64,
    /// sigma to the power 6
    s6: f64,
    /// epsilon times 4
    e4: f64,
    /// if non-zero, shifts the potential to be zero at cutoff distance
    energy_shift: f64,
    /// distance at which energy is considered too high
    squared_overlap_distance: f64,
    /// controls if tail corrections are computed
    tail_correction: bool,
    /// squared cutoff distance
    rc2: f64,
}

impl LennardJones {
    /// Create a new potential without energy shift and possibly tail corrections.
    pub fn new(sigma: f64, epsilon: f64, rc: f64, tail_correction: bool) -> Self {
        let squared_overlap_distance = 0.2 * sigma.powi(2); // approx 100*epsilon
        Self {
            rc2: rc.powi(2),
            sigma,
            epsilon,
            s6: sigma.powi(6),
            e4: epsilon * 4.0,
            energy_shift: 0.0,
            squared_overlap_distance,
            tail_correction,
        }
    }

    /// Create a new potential with energy shift and possibly tail corrections.
    pub fn new_shifted(sigma: f64, epsilon: f64, rc: f64, tail_correction: bool) -> Self {
        let squared_overlap_distance = (2.0f64 / 11.0).powf(1.0 / 3.0) * sigma.powi(2);
        Self {
            rc2: rc.powi(2),
            sigma,
            epsilon,
            s6: sigma.powi(6),
            e4: epsilon * 4.0,
            energy_shift: 4.0 * epsilon * ((sigma / rc).powi(12) - (sigma / rc).powi(6)),
            squared_overlap_distance: squared_overlap_distance,
            tail_correction,
        }
    }

    #[inline]
    pub fn rc2(&self) -> f64 {
        self.rc2
    }

    #[inline]
    pub fn energy(&self, r2: f64) -> f64 {
        let a = self.s6 / (r2 * r2 * r2);
        self.e4 * (a * a - a) - self.energy_shift
    }

    #[inline]
    pub fn virial(&self, r2: f64) -> f64 {
        let a = self.s6 / (r2 * r2 * r2);
        self.e4 * 6.0 * (2.0 * a * a - a)
    }

    #[inline]
    pub fn energy_virial(&self, r2: f64) -> (f64, f64) {
        let a = self.s6 / (r2 * r2 * r2);
        let rep = self.e4 * a * a;
        let att = -self.e4 * a;
        (rep + att - self.energy_shift, 6.0 * (2.0 * rep + att))
    }

    #[inline]
    pub fn energy_tail(&self, density: f64, nparticles: usize) -> f64 {
        if self.tail_correction {
            let s3 = self.sigma.powi(3) / self.rc2.sqrt().powi(3);
            8.0 / 3.0
                * nparticles as f64
                * PI
                * density
                * self.epsilon
                * self.sigma.powi(3)
                * (1.0 / 3.0 * s3.powi(3) - s3)
                + 2.0 * PI * nparticles as f64 * density / 3.0
                    * self.energy_shift
                    * self.rc2.sqrt().powi(3)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn pressure_tail(&self, density: f64) -> f64 {
        if self.tail_correction {
            let s3 = self.sigma.powi(3) / self.rc2.sqrt().powi(3);
            16.0 / 3.0
                * PI
                * density.powi(2)
                * self.epsilon
                * self.sigma.powi(3)
                * (2.0 / 3.0 * s3.powi(3) - s3)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn overlaps(&self, r2: f64) -> bool {
        r2 < self.squared_overlap_distance
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "LennardJones", unsendable)]
    #[derive(Clone)]
    #[pyo3(text_signature = "(sigma, epsilon, rc, tail_correction, shift_at=None)")]
    pub struct PyLennardJones(pub LennardJones);

    /// Lennard-Jones potential
    ///
    /// Parameters
    /// ----------
    /// sigma : float
    ///     Lennard-Jones size parameter
    /// epsilon : float
    ///     Lennard-Jones energetic parameter
    /// rc : float
    ///     cut-off radius
    /// tail_correction : bool
    ///     if true, tail corrections to energy and virial are computed
    /// shift_at : float, optional
    ///     shift potential to zero at given value. Defaults to None.
    ///
    /// Returns
    /// -------
    /// Potential
    #[pymethods]
    impl PyLennardJones {
        #[new]
        fn new(
            sigma: f64,
            epsilon: f64,
            rc: f64,
            tail_correction: bool,
            shift_at: Option<f64>,
        ) -> Self {
            match shift_at {
                None => Self(LennardJones::new(sigma, epsilon, rc, tail_correction)),
                Some(s) => Self(LennardJones::new_shifted(
                    sigma,
                    epsilon,
                    s,
                    tail_correction,
                )),
            }
        }

        #[getter]
        fn get_rc2(&self) -> f64 {
            self.0.rc2()
        }

        /// Pair energy
        ///
        /// Parameters
        /// ----------
        /// r2 : float
        ///     squared distance
        ///
        /// Returns
        /// -------
        /// float : energy
        #[pyo3(text_signature = "($self, r2)")]
        fn energy(&self, r2: f64) -> f64 {
            self.0.energy(r2)
        }
    }
}
