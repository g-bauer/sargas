use crate::system::System;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub trait Sampler {
    fn name(&self) -> String;
    fn sample(&mut self, system: &System);
    fn frequency(&self) -> usize;
    fn property(&self) -> HashMap<String, Vec<f64>>;
}

pub struct PotentialEnergySampler {
    data: Vec<f64>,
    frequency: usize,
}

impl PotentialEnergySampler {
    pub fn new(frequency: usize, capacity: Option<usize>) -> Self {
        Self {
            data: Vec::with_capacity(capacity.unwrap_or(100)),
            frequency,
        }
    }
}

impl Sampler for PotentialEnergySampler {
    fn name(&self) -> String {
        String::from("energy")
    }
    fn sample(&mut self, system: &System) {
        self.data.push(system.potential_energy)
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

pub struct PressureSampler {
    data: Vec<f64>,
    frequency: usize,
}

impl PressureSampler {
    pub fn new(frequency: usize, capacity: Option<usize>) -> Self {
        Self {
            data: Vec::with_capacity(capacity.unwrap_or(100)),
            frequency,
        }
    }
}

impl Sampler for PressureSampler {
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

pub struct PropertiesSampler {
    pressure: Vec<f64>,
    potential_energy: Vec<f64>,
    kinetic_energy: Vec<f64>,
    virial: Vec<f64>,
    frequency: usize,
}

impl PropertiesSampler {
    pub fn new(frequency: usize, capacity: Option<usize>) -> Self {
        Self {
            pressure: Vec::with_capacity(capacity.unwrap_or(100)),
            potential_energy: Vec::with_capacity(capacity.unwrap_or(100)),
            kinetic_energy: Vec::with_capacity(capacity.unwrap_or(100)),
            virial: Vec::with_capacity(capacity.unwrap_or(100)),
            frequency,
        }
    }
}

impl Sampler for PropertiesSampler {
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
        self.potential_energy.push(u);
        if let Some(ke) = system.configuration.kinetic_energy_from_velocities() {
            self.kinetic_energy.push(ke)
        }
        self.virial.push(v);
    }

    fn frequency(&self) -> usize {
        self.frequency
    }

    fn property(&self) -> HashMap<String, Vec<f64>> {
        let mut hm = HashMap::new();
        hm.insert(
            String::from("potential_energy"),
            self.potential_energy.clone(),
        );
        hm.insert(String::from("pressure"), self.pressure.clone());
        hm.insert(String::from("virial"), self.virial.clone());
        if self.kinetic_energy.len() > 0 {
            hm.insert(String::from("kinetic_energy"), self.kinetic_energy.clone());
            hm.insert(
                String::from("total_energy"),
                self.potential_energy
                    .iter()
                    .zip(self.kinetic_energy.iter())
                    .map(|(p, k)| p + k)
                    .collect(),
            );
        }
        hm
    }
}

pub struct WidomSampler {
    data: Vec<Vec<f64>>,
    inserted: u32,
    frequency: usize,
    beta: f64,
    ninsertions: usize,
}

impl WidomSampler {
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

impl Sampler for WidomSampler {
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

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "Sampler", unsendable)]
    pub struct PySampler {
        pub _data: Rc<RefCell<dyn Sampler>>,
    }

    #[pymethods]
    impl PySampler {
        #[staticmethod]
        fn energy(frequency: usize, capacity: Option<usize>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(PotentialEnergySampler::new(
                    frequency, capacity,
                ))),
            }
        }

        #[staticmethod]
        fn pressure(frequency: usize, capacity: Option<usize>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(PressureSampler::new(frequency, capacity))),
            }
        }

        #[staticmethod]
        fn properties(frequency: usize, capacity: Option<usize>) -> Self {
            Self {
                _data: Rc::new(RefCell::new(PropertiesSampler::new(frequency, capacity))),
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
                _data: Rc::new(RefCell::new(WidomSampler::new(
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
