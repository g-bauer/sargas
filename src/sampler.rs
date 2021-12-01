use crate::system::System;
use crate::vec::Vec3;
use chemfiles::{Frame, Trajectory, UnitCell};
use rand::thread_rng;
use rand_distr::{Distribution, Uniform};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
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
    boltzmann_factor: Vec<f64>,
    chemical_potential: Vec<f64>,
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
        capacity: Option<usize>,
    ) -> Self {
        Self {
            boltzmann_factor: Vec::with_capacity(capacity.unwrap_or(100)),
            chemical_potential: Vec::with_capacity(capacity.unwrap_or(100)),
            inserted: 0,
            frequency,
            beta: 1.0 / temperature,
            ninsertions,
        }
    }
}

/// Inserts a ghost particle into the simulation box and computes the
/// sum of Boltzmann factors over all insertions.
///
/// Returns: sum of exp(-beta deltaU)
/// where deltaU is the energy of the ghost particle with all particles in the system.
pub fn particle_insertion(system: &System, ninsertions: usize, beta: f64) -> f64 {
    let nparticles = system.configuration.nparticles;
    if nparticles == 0 {
        return 0.0;
    }
    let mut rng = thread_rng();
    let dist = Uniform::new(0.0, system.configuration.box_length);
    let mut boltzmann_factor = 0.0;
    for _ in 0..ninsertions {
        // position of ghost particle
        let ri = Vec3::new(
            dist.sample(&mut rng),
            dist.sample(&mut rng),
            dist.sample(&mut rng),
        );
        let energy = system.particle_energy(nparticles + 1, &ri, None);
        boltzmann_factor += (-beta * energy).exp();
    }
    boltzmann_factor
}

impl Sampler for WidomSampler {
    fn name(&self) -> String {
        String::from("widom")
    }

    fn sample(&mut self, system: &System) {
        let sum_boltzmann_factor = particle_insertion(system, self.ninsertions, self.beta);
        let density = system.density();
        let nparticles = system.configuration.nparticles;
        let u_tail = system.potential.energy_tail(density, nparticles) / nparticles as f64;
        // current_boltzmann_factor_mean is the last value that was recorded for <exp(-beta deltaU)>
        let new_boltzmann_factor_mean =
            if let Some(current_boltzmann_factor_mean) = self.boltzmann_factor.last() {
                (current_boltzmann_factor_mean * self.inserted as f64 + sum_boltzmann_factor)
                    / (self.inserted as f64 + self.ninsertions as f64)
            } else {
                sum_boltzmann_factor / self.ninsertions as f64
            };
        self.chemical_potential
            .push(density.ln() + 2.0 * u_tail - new_boltzmann_factor_mean.ln());
        self.boltzmann_factor.push(new_boltzmann_factor_mean);
        self.inserted += self.ninsertions as u32;
    }

    fn frequency(&self) -> usize {
        self.frequency
    }

    fn property(&self) -> HashMap<String, Vec<f64>> {
        let mut hm = HashMap::new();
        hm.insert(
            String::from("boltzmann_factor"),
            self.boltzmann_factor.clone(),
        );
        hm.insert(
            String::from("chemical_potential"),
            self.chemical_potential.clone(),
        );
        hm
    }
}

pub struct TrajectoryWriter {
    filename: PathBuf,
    frequency: usize,
    step: usize,
}

impl TrajectoryWriter {
    pub fn new(filename: PathBuf, frequency: usize) -> Result<Self, String> {
        let _trajectory = Trajectory::open_with_format(filename.clone(), 'w', "").unwrap();
        Ok(Self {
            filename,
            frequency,
            step: 1,
        })
    }
}

impl Sampler for TrajectoryWriter {
    fn name(&self) -> String {
        String::from("trajectory")
    }

    fn sample(&mut self, system: &System) {
        let mut trj = Trajectory::open_with_format(self.filename.clone(), 'a', "").unwrap();
        let mut frame = Frame::new();
        frame.resize(system.configuration.positions.len());
        frame.set_step(self.step);
        let l = system.configuration.box_length;
        frame.set_cell(&UnitCell::new([l, l, l]));
        frame.add_velocities();

        for (p, frame_position) in system
            .configuration
            .positions
            .iter()
            .zip(frame.positions_mut())
        {
            *frame_position = [p.x, p.y, p.z];
        }

        match system.configuration.velocities.as_ref() {
            Some(v) => v
                .iter()
                .zip(frame.velocities_mut())
                .for_each(|(vi, vif)| *vif = [vi.x, vi.y, vi.z]),
            None => frame
                .velocities_mut()
                .iter_mut()
                .for_each(|v| *v = [0.0; 3]),
        }

        trj.write(&frame).unwrap();
        self.step += 1;
    }

    fn frequency(&self) -> usize {
        self.frequency
    }

    fn property(&self) -> HashMap<String, Vec<f64>> {
        HashMap::new()
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

        /// Store the positions and possibly velocities in a file.
        ///
        /// Parameters
        /// ----------
        /// filename : str
        ///     the file name where the trajectory will be stored.
        /// frequency : int
        ///     the frequency with which the trajectory is stored.
        ///
        /// Returns
        /// -------
        /// Sampler
        #[staticmethod]
        #[pyo3(text_signature = "(filename, frequency)")]
        fn trajectory(filename: &str, frequency: usize) -> Self {
            Self {
                _data: Rc::new(RefCell::new(
                    TrajectoryWriter::new(PathBuf::from(filename), frequency).unwrap(),
                )),
            }
        }

        /// Calculate the chemical potential by inserting ghost particles.
        ///
        /// Parameters
        /// ----------
        /// frequency : int
        ///     the frequency with which ghost particles are inserted.
        /// temperature : float
        ///     the reduced temperature
        /// ninsertions : int
        ///     how many ghost particles are inserted.
        /// capacity : int, optional
        ///     size of the data storage. Defaults to 100.
        #[staticmethod]
        #[pyo3(text_signature = "(frequency, temperature, ninsertions, capacity)")]
        fn widom(
            frequency: usize,
            temperature: f64,
            ninsertions: usize,
            capacity: Option<usize>,
        ) -> Self {
            Self {
                _data: Rc::new(RefCell::new(WidomSampler::new(
                    frequency,
                    temperature,
                    ninsertions,
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
