use crate::configuration::Configuration;
use crate::lennard_jones::LennardJones;
use crate::system::System;

/// Test system using
/// - 256 Lennard-Jones particles on a lattice
/// - rc = 2.5
/// - no tail correction and no shift
/// - initial temperature of 0.7
/// - density of 0.9
pub fn test_system() -> System {
    let potential = LennardJones::new(1.0, 1.0, 2.5, false);
    let configuration = Configuration::lattice(256, 0.9, 256, Some(0.7));
    System::new(configuration, potential).unwrap()
}
