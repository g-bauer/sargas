use crate::system::System;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Observer {
    pub name: String,
    pub f: Box<dyn Fn(&System) -> HashMap<String, f64>>,
    pub frequency: usize,
    pub property: HashMap<String, Vec<f64>>,
}

impl Observer {
    pub fn new(
        name: String,
        f: Box<dyn Fn(&System) -> HashMap<String, f64>>,
        frequency: usize,
        size: Option<usize>,
    ) -> Self {
        Self {
            name,
            f,
            frequency,
            property: HashMap::with_capacity(size.unwrap_or(1000)),
        }
    }

    pub fn sample(&mut self, system: &System) {
        let p = self.f.as_ref()(system);
        if self.property.is_empty() {
            p.iter().for_each(|(k, &v)| {
                self.property.insert(k.clone(), vec![v]);
            })
        } else {
            p.iter().for_each(|(k, &v)| {
                self.property.get_mut(k).unwrap().push(v);
            })
        }
    }

    pub fn get(&self) -> HashMap<String, Vec<f64>> {
        self.property.clone()
    }
}

fn energy_sample(system: &System) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("energy".into(), system.energy);
    m
}

fn pressure_sample(system: &System) -> HashMap<String, f64> {
    // let p_ig = system.configuration.density() / system.beta;
    let pressure = system.virial / (3.0 * system.configuration.volume())
        + system
            .potential
            .pressure_tail(system.configuration.density());
    let mut m = HashMap::new();
    m.insert("pressure".into(), pressure);
    m
}

fn properties_sample(system: &System) -> HashMap<String, f64> {
    // let p_ig = system.configuration.density() / system.beta;
    let volume = system.configuration.volume();
    let pressure = system.virial / (3.0 * volume)
        + system
            .potential
            .pressure_tail(system.configuration.density());
    let mut m = HashMap::new();
    m.insert("pressure".into(), pressure);
    m.insert("volume".into(), volume);
    m.insert("energy".into(), system.energy);
    m.insert("virial".into(), system.virial);
    m.insert("density".into(), system.configuration.density());
    m.insert("nparticles".into(), system.configuration.nparticles as f64);
    m
}

fn widom_insertion(system: &System) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert("mu".into(), system.ghost_particle_energy());
    m
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;
    use super::*;

    #[pyclass(name = "Observer", unsendable)]
    pub struct PyObserver {
        pub _data: Rc<RefCell<Observer>>,
    }

    #[pymethods]
    impl PyObserver {
        #[staticmethod]
        fn energy(name: String, frequency: usize) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Observer::new(
                    name,
                    Box::new(energy_sample),
                    frequency,
                    None,
                ))),
            }
        }

        #[staticmethod]
        fn pressure(frequency: usize) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Observer::new(
                    "pressure".to_owned(),
                    Box::new(pressure_sample),
                    frequency,
                    None,
                ))),
            }
        }

        #[staticmethod]
        fn properties(frequency: usize) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Observer::new(
                    "properties".to_owned(),
                    Box::new(properties_sample),
                    frequency,
                    None,
                ))),
            }
        }

        #[staticmethod]
        fn widom_insertion(frequency: usize) -> Self {
            Self {
                _data: Rc::new(RefCell::new(Observer::new(
                    "widom".to_owned(),
                    Box::new(widom_insertion),
                    frequency,
                    None,
                ))),
            }
        }

        #[getter]
        fn get_data(&self) -> PyResult<HashMap<String, Vec<f64>>> {
            Ok(self._data.borrow().get())
        }
    }
}
