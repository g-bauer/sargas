use std::f64::consts::PI;
use std::rc::Rc;

pub trait Potential {
    fn rc2(&self) -> f64;
    fn energy(&self, r2: f64) -> f64;
    fn virial(&self, r2: f64) -> f64;
    fn energy_virial(&self, r2: f64) -> (f64, f64);
    fn energy_tail(&self, density: f64, nparticles: usize) -> f64;
    fn pressure_tail(&self, density: f64) -> f64;
    fn overlaps(&self, r2: f64) -> bool;
}

#[derive(Debug)]
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
            squared_overlap_distance: squared_overlap_distance,
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
}

impl Potential for LennardJones {
    #[inline]
    fn rc2(&self) -> f64 {
        self.rc2
    }

    fn energy(&self, r2: f64) -> f64 {
        let a = self.s6 / (r2 * r2 * r2);
        self.e4 * (a * a - a) - self.energy_shift
    }

    fn virial(&self, r2: f64) -> f64 {
        let a = self.s6 / (r2 * r2 * r2);
        self.e4 * 6.0 * (2.0 * a * a - a)
    }

    fn energy_virial(&self, r2: f64) -> (f64, f64) {
        let a = self.s6 / (r2 * r2 * r2);
        let rep = self.e4 * a * a;
        let att = -self.e4 * a;
        (rep + att - self.energy_shift, 6.0 * (2.0 * rep + att))
    }

    fn energy_tail(&self, density: f64, nparticles: usize) -> f64 {
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

    fn pressure_tail(&self, density: f64) -> f64 {
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
    fn overlaps(&self, r2: f64) -> bool {
        r2 < self.squared_overlap_distance
    }
}

#[derive(Debug)]
pub struct HardSphere {
    rc2: f64,
    sigma: f64,
    sigma2: f64,
}

impl HardSphere {
    pub fn new(sigma: f64, rc: f64) -> Self {
        Self {
            rc2: rc.powi(2),
            sigma,
            sigma2: sigma.powi(2),
        }
    }
}

impl Potential for HardSphere {
    #[inline]
    fn rc2(&self) -> f64 {
        self.rc2
    }

    fn energy(&self, r2: f64) -> f64 {
        if r2 > self.sigma2 {
            0.0
        } else {
            f64::INFINITY
        }
    }

    fn virial(&self, r2: f64) -> f64 {
        if r2 > self.sigma2 {
            0.0
        } else {
            -f64::INFINITY
        }
    }

    fn energy_virial(&self, r2: f64) -> (f64, f64) {
        if r2 > self.sigma2 {
            (0.0, 0.0)
        } else {
            (f64::INFINITY, -f64::INFINITY)
        }
    }

    fn energy_tail(&self, _density: f64, _nparticles: usize) -> f64 {
        0.0
    }

    fn pressure_tail(&self, _density: f64) -> f64 {
        0.0
    }

    #[inline]
    fn overlaps(&self, r2: f64) -> bool {
        r2 <= self.sigma2
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "Potential", unsendable)]
    #[derive(Clone)]
    pub struct PyPotential(pub Rc<dyn Potential>);

    #[pymethods]
    impl PyPotential {
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
        #[staticmethod]
        fn lennard_jones(
            sigma: f64,
            epsilon: f64,
            rc: f64,
            tail_correction: bool,
            shift_at: Option<f64>,
        ) -> Self {
            match shift_at {
                None => Self(Rc::new(LennardJones::new(sigma, epsilon, rc, tail_correction))),
                Some(s) => Self(
                    Rc::new(LennardJones::new_shifted(
                        sigma,
                        epsilon,
                        s,
                        tail_correction,
                    )),
                ),
            }
        }

        #[staticmethod]
        fn hard_sphere(sigma: f64, rc: f64) -> Self {
            Self(Rc::new(HardSphere::new(sigma, rc)))
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
        fn energy(&self, r2: f64) -> f64 {
            self.0.energy(r2)
        }
    }

    // #[derive(Debug)]
    // pub struct Mie {
    //     sigma: f64,
    //     epsilon: f64,
    //     m: u32,
    //     n: u32,
    //     s_rep: f64,
    //     s_att: f64,
    //     pref: f64,
    // }

    // impl Mie {
    //     pub fn new(sigma: f64, epsilon: f64, m: u32, n: u32) -> Self {
    //         let nf = n as f64;
    //         let mf = m as f64;
    //         let pref = (nf / (nf - mf)) * (nf / mf).powf(mf / (nf - mf)) * epsilon;
    //         Self {
    //             sigma,
    //             epsilon,
    //             m,
    //             n,
    //             s_rep: sigma.powi(m as i32),
    //             s_att: sigma.powi(n as i32),
    //             pref,
    //         }
    //     }
    // }

    // impl Potential for Mie {
    //     fn energy(&self, r2: f64) -> f64 {
    //         self.pref
    //             * ((self.s_rep / r2.powf(0.5 * self.m as f64))
    //                 - (self.s_att / r2.powf(0.5 * self.n as f64)))
    //     }

    //     fn virial(&self, _r2: f64) -> f64 {
    //         todo!()
    //     }

    //     fn energy_virial(&self, _r2: f64) -> (f64, f64) {
    //         todo!()
    //     }
    // }
}
