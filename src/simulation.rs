use crate::observer::Observer;
use crate::propagator::Propagator;
use crate::system::System;
use std::collections::HashMap;
use std::rc::Rc;
use std::{borrow::Borrow, cell::RefCell};

pub struct Simulation {
    pub propagator: Rc<RefCell<dyn Propagator>>,
    pub system: Rc<RefCell<System>>,
    pub update_propagator: Option<usize>,
    pub observers: HashMap<String, Rc<RefCell<Observer>>>,
}

impl Simulation {
    pub fn new(
        system: Rc<RefCell<System>>,
        propagator: Rc<RefCell<dyn Propagator>>,
        update_propagator: Option<usize>,
    ) -> Result<Self, String> {
        if let Some(f) = update_propagator {
            if f == 0 {
                return Err("Nope!".to_owned());
            }
        }
        Ok(Self {
            propagator,
            system,
            update_propagator: update_propagator,
            observers: HashMap::new(),
        })
    }

    pub fn add_observer(&mut self, observer: Rc<RefCell<Observer>>) {
        self.observers
            .insert(observer.as_ref().borrow().name.clone(), observer.clone());
    }

    pub fn remove_observer(&mut self, observer: Rc<RefCell<Observer>>) {
        self.observers.remove(&observer.as_ref().borrow().name);
    }

    pub fn print_observers(&self) -> Vec<String> {
        self.observers.borrow().keys().cloned().collect()
    }

    pub fn deactivate_propagator_updates(&mut self) {
        self.update_propagator = None
    }

    pub fn run(&mut self, steps: usize) {
        let mut s = self.system.borrow_mut();
        s.recompute();
        for i in 1..=steps {
            self.propagator.borrow_mut().propagate(&mut s);

            match self.update_propagator {
                Some(f) if i % f == 0 => self.propagator.borrow_mut().adjust(&s),
                _ => (),
            }

            {
                self.observers.iter_mut().for_each(|(_, v)| {
                    let mut o = v.borrow_mut();
                    if i % o.frequency == 0 {
                        o.sample(&s)
                    }
                })
            }
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::observer::python::PyObserver;
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
            update_propagator: Option<usize>,
        ) -> Self {
            Self {
                _data: Simulation::new(system._data, propagator._data, update_propagator).unwrap(),
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
    //             self._data.update_propagator.is_some(),
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
