use pyo3::prelude::*;
use std::f64::consts::PI;
use std::rc::Rc;

pub trait Potential {
    fn energy(&self, r2: f64) -> f64;
    fn virial(&self, r2: f64) -> f64;
    fn energy_virial(&self, r2: f64) -> (f64, f64);
    fn energy_tail(&self, rc: f64, density: f64, nparticles: usize) -> f64;
}

#[derive(Debug)]
pub struct LennardJones {
    sigma: f64,
    epsilon: f64,
    s6: f64,
    e4: f64,
    energy_shift: f64,
}

impl LennardJones {
    pub fn new(sigma: f64, epsilon: f64) -> Self {
        Self {
            sigma,
            epsilon,
            s6: sigma.powi(6),
            e4: epsilon * 4.0,
            energy_shift: 0.0,
        }
    }

    pub fn new_shifted(sigma: f64, epsilon: f64, rc: f64) -> Self {
        Self {
            sigma,
            epsilon,
            s6: sigma.powi(6),
            e4: epsilon * 4.0,
            energy_shift: 4.0 * epsilon * ((sigma / rc).powi(12) - (sigma / rc).powi(6)),
        }
    }
}

impl Potential for LennardJones {
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
        (
            rep - att - self.energy_shift,
            6.0 * (2.0 * rep + att),
        )
    }

    fn energy_tail(&self, rc: f64, density: f64, nparticles: usize) -> f64 {
        let s3 = self.sigma.powi(3) / rc.powi(3);
        8.0 / 3.0
            * nparticles as f64
            * PI
            * density
            * self.epsilon
            * self.sigma.powi(3)
            * (1.0 / 3.0 * (s3.powi(3) - s3))
            + 2.0 * PI * nparticles as f64 * density / 3.0 * self.energy_shift * rc.powi(3)
    }
}

#[pyclass(name = "Potential", unsendable)]
#[derive(Clone)]
pub struct PyPotential {
    pub _data: Rc<dyn Potential>,
}

#[pymethods]
impl PyPotential {
    #[staticmethod]
    fn lennard_jones(sigma: f64, epsilon: f64) -> Self {
        Self {
            _data: Rc::new(LennardJones::new(sigma, epsilon)),
        }
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
