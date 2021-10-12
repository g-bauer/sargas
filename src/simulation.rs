use crate::propagator::Propagator;
use crate::sampler::Sampler;
use crate::system::System;
use std::collections::HashMap;
use std::rc::Rc;
use std::{borrow::Borrow, cell::RefCell};

/// A molecular simulation object.
pub struct Simulation {
    /// current (time) step
    pub step: usize,
    /// system propagator
    pub propagator: Rc<RefCell<dyn Propagator>>,
    /// system
    pub system: Rc<RefCell<System>>,
    ///
    pub adjustment_frequency: Option<usize>,
    ///
    pub samplers: HashMap<String, Rc<RefCell<dyn Sampler>>>,
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
            samplers: HashMap::new(),
        })
    }

    pub fn add_sampler(&mut self, sampler: Rc<RefCell<dyn Sampler>>) {
        self.samplers
            .insert(sampler.as_ref().borrow().name(), sampler.clone());
    }

    pub fn remove_sampler(&mut self, sampler: Rc<RefCell<dyn Sampler>>) {
        self.samplers.remove(&sampler.as_ref().borrow().name());
    }

    pub fn print_samplers(&self) -> Vec<String> {
        self.samplers.borrow().keys().cloned().collect()
    }

    pub fn deactivate_propagator_updates(&mut self) {
        self.adjustment_frequency = None
    }

    pub fn run(&mut self, steps: usize) {
        let mut s = self.system.borrow_mut();
        s.recompute_energy_forces();
        for _ in 1..=steps {
            self.step += 1;
            self.propagator.borrow_mut().propagate(&mut s);

            match self.adjustment_frequency {
                Some(f) if self.step % f == 0 => self.propagator.borrow_mut().adjust(&mut s),
                _ => (),
            }

            {
                let i = self.step;
                self.samplers.iter_mut().for_each(|(_, v)| {
                    let mut o = v.borrow_mut();
                    if i % o.frequency() == 0 {
                        o.sample(&s)
                    }
                });
            }
        }
    }
}

// #[cfg(feature = "python")]
// impl Simulation {
//     pub fn run_cancelable(&mut self, py: Python, steps: usize) -> PyResult<()> {
//         let mut s = self.system.borrow_mut();
//         s.recompute();
//         for _ in 1..=steps {
//             self.step += 1;
//             self.propagator.borrow_mut().propagate(&mut s);

//             match self.adjustment_frequency {
//                 Some(f) if self.step % f == 0 => self.propagator.borrow_mut().adjust(&mut s),
//                 _ => (),
//             }

//             {
//                 let i = self.step;
//                 self.samplers.iter_mut().for_each(|(_, v)| {
//                     let mut o = v.borrow_mut();
//                     if i % o.frequency() == 0 {
//                         o.sample(&s)
//                     }
//                 });
//             }
//         }
//     }
// }

#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::propagator::molecular_dynamics::python::PyMolecularDynamics;
    use crate::propagator::monte_carlo::python::*;
    use crate::sampler::python::PySampler;
    use crate::system::python::PySystem;
    use pyo3::prelude::*;

    impl Simulation {
        pub fn run_cancelable(&mut self, py: Python, steps: usize) -> PyResult<()> {
            let mut s = self.system.borrow_mut();
            s.recompute_energy_forces();
            for _ in 1..=steps {
                self.step += 1;
                self.propagator.borrow_mut().propagate(&mut s);

                match self.adjustment_frequency {
                    Some(f) if self.step % f == 0 => self.propagator.borrow_mut().adjust(&mut s),
                    _ => (),
                }

                if self.step % 250 == 0 {
                    py.check_signals()?;
                }

                {
                    let i = self.step;
                    self.samplers.iter_mut().for_each(|(_, v)| {
                        let mut o = v.borrow_mut();
                        if i % o.frequency() == 0 {
                            o.sample(&s)
                        }
                    });
                }
            }
            Ok(())
        }
    }

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

        fn add_sampler(&mut self, sampler: &PySampler) {
            self._data.add_sampler(sampler._data.clone())
        }

        fn remove_sampler(&mut self, sampler: &PySampler) {
            self._data.remove_sampler(sampler._data.clone())
        }

        fn deactivate_propagator_updates(&mut self) {
            self._data.deactivate_propagator_updates()
        }

        fn run(&mut self, py: Python, steps: usize) -> PyResult<()> {
            self._data.run_cancelable(py, steps)
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

    //     fn add_sampler(&mut self, sampler: &PySampler) {
    //         self._data.add_sampler(sampler._data.clone())
    //     }

    //     fn remove_sampler(&mut self, sampler: &PySampler) {
    //         self._data.remove_sampler(sampler._data.clone())
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
