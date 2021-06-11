use monte_carlo::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;

fn energy_sample(system: &System) -> f64 {
    system.energy
}

fn pressure_sample(system: &System) -> f64 {
    let volume = system.box_length.powi(3);
    system.nparticles as f64 / volume / system.beta + system.virial / (3.0 * volume)
}

fn main() -> Result<(), String> {
    // Setup
    let nparticles = 800;
    let potential = Rc::new(LennardJones::new(1.0, 1.0));
    let system = Rc::new(RefCell::new(System::with_lattice(nparticles, 0.8, 0.9, 3.0, potential)));
    let propagator = DisplaceParticle::new(0.1, 0.4, nparticles);
    let mut simulation = Simulation::new(system, propagator, Some(200))?;

    // Add observer
    // let energy_computation = Box::new(|system: &System| system.energy);
    let energy_sample = Box::new(energy_sample);
    let energy = Rc::new(RefCell::new(Observer::new("energy".to_owned(), energy_sample, 800, Some(250))));
    let pressure_sample = Box::new(pressure_sample);
    let pressure = Rc::new(RefCell::new(Observer::new("pressure".to_owned(), pressure_sample, 800, Some(250))));

    // Run simulation
    let equilibration = 1_000 * nparticles;
    let production = 1_000 * nparticles;
    simulation.run(equilibration); // Equilibration
    simulation.deactivate_propagator_updates();
    simulation.add_observer(energy);
    simulation.add_observer(pressure);
    simulation.system.borrow_mut().initialize_system();
    println!("Starting production.");
    simulation.run(production); // Production

    println!("{:?}", simulation.observers.get("energy").unwrap().as_ref().borrow().property);
    println!("{:?}", simulation.observers.get("pressure").unwrap().as_ref().borrow().property);
    Ok(())
}
