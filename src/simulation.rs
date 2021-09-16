use crate::observer::Observer;
use crate::propagator::Propagator;
use crate::system::System;
use std::collections::HashMap;
use std::rc::Rc;
use std::{borrow::Borrow, cell::RefCell};

pub struct Simulation {
    pub step: usize,
    pub propagator: Rc<RefCell<dyn Propagator>>,
    pub system: Rc<RefCell<System>>,
    pub adjustment_frequency: Option<usize>,
    pub observers: HashMap<String, Rc<RefCell<dyn Observer>>>,
}

impl Simulation {
    pub fn new(
        system: Rc<RefCell<System>>,
        propagator: Rc<RefCell<dyn Propagator>>,
        adjustment_frequency: Option<usize>,
    ) -> Result<Self, String> {
        if let Some(f) = adjustment_frequency {
            if f == 0 {
                return Err("Nope!".to_owned());
            }
        }
        Ok(Self {
            step: 0,
            propagator,
            system,
            adjustment_frequency: adjustment_frequency,
            observers: HashMap::new(),
        })
    }

    pub fn add_observer(&mut self, observer: Rc<RefCell<dyn Observer>>) {
        self.observers
            .insert(observer.as_ref().borrow().name(), observer.clone());
    }

    pub fn remove_observer(&mut self, observer: Rc<RefCell<dyn Observer>>) {
        self.observers.remove(&observer.as_ref().borrow().name());
    }

    pub fn print_observers(&self) -> Vec<String> {
        self.observers.borrow().keys().cloned().collect()
    }

    pub fn deactivate_propagator_updates(&mut self) {
        self.adjustment_frequency = None
    }

    pub fn run(&mut self, steps: usize) {
        let mut s = self.system.borrow_mut();
        s.recompute();
        for _ in 1..=steps {
            self.step += 1;
            self.propagator.borrow_mut().propagate(&mut s);

            match self.adjustment_frequency {
                Some(f) if self.step % f == 0 => self.propagator.borrow_mut().adjust(&mut s),
                _ => (),
            }

            {
                let i = self.step;
                self.observers.iter_mut().for_each(|(_, v)| {
                    let mut o = v.borrow_mut();
                    if i % o.frequency() == 0 {
                        o.sample(&s)
                    }
                });
            }
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::observer::python::PyObserver;
    use crate::propagator::molecular_dynamics::python::PyMolecularDynamics;
    use crate::propagator::monte_carlo::python::*;
    use crate::system::python::PySystem;
    use pyo3::prelude::*;

    #[pyclass(name = "Simulation", unsendable)]
    pub struct PySimulation {
        _data: Simulation,
    }

    #[pymethods]
    impl PySimulation {
        #[staticmethod]
        fn monte_carlo(
            system: PySystem,
            propagator: PyMonteCarlo,
            adjustment_frequency: Option<usize>,
        ) -> Self {
            Self {
                _data: Simulation::new(system._data, propagator._data, adjustment_frequency)
                    .unwrap(),
            }
        }

        #[staticmethod]
        fn molecular_dynamics(
            system: PySystem,
            propagator: PyMolecularDynamics,
            thermostat_frequency: Option<usize>,
        ) -> Self {
            Self {
                _data: Simulation::new(system._data, propagator._data, thermostat_frequency)
                    .unwrap(),
            }
        }

        fn add_observer(&mut self, observer: &PyObserver) {
            self._data.add_observer(observer._data.clone())
        }

        fn remove_observer(&mut self, observer: &PyObserver) {
            self._data.remove_observer(observer._data.clone())
        }

        fn deactivate_propagator_updates(&mut self) {
            self._data.deactivate_propagator_updates()
        }

        fn run(&mut self, steps: usize) {
            self._data.run(steps)
        }
    }

    // #[pyproto]
    // impl PyObjectProtocol for PySimulation {
    //     fn __repr__(&self) -> PyResult<String> {
    //         Ok(fmt::format(format_args!(
    //             "Simulation\n==========\nadjust displacement: {}\n\n{}\n\n",
    //             self._data.adjustment_frequency.is_some(),
    //             self._data.system.as_ref().borrow().to_string(),
    //             // self._data.propagator.to_string(),
    //         )))
    //     }
    // }

    // #[pyclass(name = "MolecularDynamics", unsendable)]
    // pub struct MolecularDynamics {
    //     _data: Simulation<VelocityVerlet>,
    // }

    // #[pymethods]
    // impl MolecularDynamics {
    //     #[new]
    //     fn new(
    //         System: PySystem,
    //         propagator: PyVelocityVerlet,
    //     ) -> Self {
    //         Self {
    //             _data: Simulation::new(System._data, propagator._data, None).unwrap(),
    //         }
    //     }

    //     fn add_observer(&mut self, observer: &PyObserver) {
    //         self._data.add_observer(observer._data.clone())
    //     }

    //     fn remove_observer(&mut self, observer: &PyObserver) {
    //         self._data.remove_observer(observer._data.clone())
    //     }

    //     fn run(&mut self, steps: usize) {
    //         self._data.run(steps)
    //     }
    // }

    // #[pyproto]
    // impl PyObjectProtocol for MolecularDynamics {
    //     fn __repr__(&self) -> PyResult<String> {
    //         Ok(fmt::format(format_args!("Molecular Dynamics Simulation (NVE)\n==========\n")))
    //     }
    // }
}
