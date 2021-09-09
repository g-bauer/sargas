use crate::system::System;
use ndarray::ArrayView1;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub trait Observer {
    fn name(&self) -> String;
    fn sample(&mut self, system: &System);
    fn frequency(&self) -> usize;
    fn property(&self) -> HashMap<String, Vec<f64>>;
}

pub struct PotentialEnergyObserver {
    data: Vec<f64>,
    frequency: usize,
}

impl PotentialEnergyObserver {
    pub fn new(frequency: usize, capacity: Option<usize>) -> Self {
        Self {
            data: Vec::with_capacity(capacity.unwrap_or(100)),
            frequency,
        }
    }
}

impl Observer for PotentialEnergyObserver {
    fn name(&self) -> String {
        String::from("energy")
    }
    fn sample(&mut self, system: &System) {
        self.data.push(system.energy)
    }
    fn frequency(&self) -> usize {
        self.frequency
    }
    fn property(&self) -> HashMap<String, Vec<f64>> {
        let mut hm = HashMap::new();
        hm.insert(String::from("energy"), self.data.clone());
        hm
    }
}

pub struct PressureObserver {
    data: Vec<f64>,
    frequency: usize,
}

impl PressureObserver {
    pub fn new(frequency: usize, capacity: Option<usize>) -> Self {
        Self {
            data: Vec::with_capacity(capacity.unwrap_or(100)),
            frequency,
        }
    }
}

impl Observer for PressureObserver {
    fn name(&self) -> String {
        String::from("pressure")
    }

    fn sample(&mut self, system: &System) {
        let pressure = system.virial / (3.0 * system.configuration.volume())
            + system
                .potential
                .pressure_tail(system.configuration.density());
        self.data.push(pressure)
    }

    fn frequency(&self) -> usize {
        self.frequency
    }

    fn property(&self) -> HashMap<String, Vec<f64>> {
        let mut hm = HashMap::new();
        hm.insert(self.name(), self.data.clone());
        hm
    }
}

pub struct PropertiesObserver {
    volume: Vec<f64>,
    pressure: Vec<f64>,
    energy: Vec<f64>,
    virial: Vec<f64>,
    density: Vec<f64>,
    nparticles: Vec<f64>,
    frequency: usize,
}

impl PropertiesObserver {
    pub fn new(frequency: usize, capacity: Option<usize>) -> Self {
        Self {
            volume: Vec::with_capacity(capacity.unwrap_or(100)),
            pressure: Vec::with_capacity(capacity.unwrap_or(100)),
            energy: Vec::with_capacity(capacity.unwrap_or(100)),
            virial: Vec::with_capacity(capacity.unwrap_or(100)),
            density: Vec::with_capacity(capacity.unwrap_or(100)),
            nparticles: Vec::with_capacity(capacity.unwrap_or(100)),
            frequency,
        }
    }
}

impl Observer for PropertiesObserver {
    fn name(&self) -> String {
        String::from("properties")
    }

    fn sample(&mut self, system: &System) {
        let volume = system.configuration.volume();
        let (u, v) = system.energy_virial();
        let pressure = v / (3.0 * volume)
            + system
                .potential
                .pressure_tail(system.configuration.density());
        self.pressure.push(pressure);
        self.volume.push(volume);
        self.energy.push(u);
        self.virial.push(v);
        self.density.push(system.configuration.density());
        self.nparticles.push(system.configuration.nparticles as f64);
    }

    fn frequency(&self) -> usize {
        self.frequency
    }

    fn property(&self) -> HashMap<String, Vec<f64>> {
        let mut hm = HashMap::new();
        hm.insert(String::from("potential_energy"), self.energy.clone());
        hm.insert(String::from("pressure"), self.pressure.clone());
        hm.insert(String::from("volume"), self.volume.clone());
        hm.insert(String::from("virial"), self.virial.clone());
        hm.insert(String::from("density"), self.density.clone());
        hm.insert(String::from("nparticles"), self.nparticles.clone());
        hm
    }
}

fn logsumexp(x: ArrayView1<f64>) -> f64 {
    let x_max = x.iter().cloned().reduce(f64::max).unwrap();
    x_max + x.mapv(|x_i| (x_i - x_max).exp()).sum().ln()
}

pub struct WidomObserver {
    data: Vec<Vec<f64>>,
    inserted: u32,
    frequency: usize,
    beta: f64,
    ninsertions: usize,
}

impl WidomObserver {
    pub fn new(
        frequency: usize,
        temperature: f64,
        ninsertions: usize,
        replicas: Option<usize>,
        capacity: Option<usize>,
    ) -> Self {
        let data = (0..replicas.unwrap_or(1))
            .map(|_| Vec::with_capacity(capacity.unwrap_or(100)))
            .collect();
        Self {
            data,
            inserted: 0,
            frequency,
            beta: 1.0 / temperature,
            ninsertions,
        }
    }
}

impl Observer for WidomObserver {
    fn name(&self) -> String {
        String::from("widom")
    }

    fn sample(&mut self, system: &System) {
        let (b, n, i) = (self.beta, self.ninsertions, self.inserted);
        self.data.iter_mut().for_each(|di| {
            let sum_exp = system.ghost_particle_energy_sum(b, n);
            if let Some(l) = di.last() {
                di.push((l * i as f64 + sum_exp) / (i as f64 + n as f64))
            } else {
                di.push(sum_exp / n as f64)
            }
        });
        self.inserted += self.ninsertions as u32;
    }

    fn frequency(&self) -> usize {
        self.frequency
    }

    fn property(&self) -> HashMap<String, Vec<f64>> {
        let mut hm = HashMap::new();
        self.data.iter().enumerate().for_each(|(i, di)| {
            hm.insert(format!("mu{}", i), di.iter().map(|di| -di.ln()).collect());
        });
        hm
    }
}

// fn widom_insertion(system: &System) -> HashMap<String, f64> {
//     let beta = 0.0;
//     let ninsertions = 0;
//     let mut m = HashMap::new();
//     let du = -beta * system.ghost_particle_energy(ninsertions);
//     let mu = -(logsumexp(du.view())) + (ninsertions as f64).ln();
//     m.insert("beta_mu".into(), mu);
//     m
// }

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "Observer", unsendable)]
    pub struct PyObserver {
        pub _data: Rc<RefCell<dyn Observer>>,
    }

    #[pymethods]
    impl PyObserver {
        #[staticmethod]
        fn energy(frequency: usize, capacity: Option<usize>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(PotentialEnergyObserver::new(
                    frequency, capacity,
                ))),
            }
        }

        #[staticmethod]
        fn pressure(frequency: usize, capacity: Option<usize>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(PressureObserver::new(frequency, capacity))),
            }
        }

        #[staticmethod]
        fn properties(frequency: usize, capacity: Option<usize>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(PropertiesObserver::new(frequency, capacity))),
            }
        }

        #[staticmethod]
        fn widom(
            frequency: usize,
            temperature: f64,
            ninsertions: usize,
            repeat: Option<usize>,
            capacity: Option<usize>,
        ) -> Self {
            Self {
                _data: Rc::new(RefCell::new(WidomObserver::new(
                    frequency,
                    temperature,
                    ninsertions,
                    repeat,
                    capacity,
                ))),
            }
        }

        #[getter]
        fn get_data(&self) -> PyResult<HashMap<String, Vec<f64>>> {
            Ok(self._data.borrow().property())
        }
    }
}
