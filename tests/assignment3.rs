use approx::assert_relative_eq;
use sargas::configuration::Configuration;
use sargas::error::SargasError;
use sargas::lennard_jones::LennardJones;
use sargas::propagator::molecular_dynamics::velocity_verlet::VelocityVerlet;
use sargas::propagator::molecular_dynamics::Integrator;
use sargas::system::System;
use sargas::vec::Vec3;

#[test]
fn velocity_verlet() -> Result<(), SargasError> {
    let potential = LennardJones::new(1.0, 1.0, 3.5, true);
    let positions = vec![Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0)];
    let velocities = vec![Vec3::zero(), Vec3::zero()];
    let configuration = Configuration::new(positions, Some(velocities), 12.0);
    let mut system = System::new(configuration, potential)?;
    let mut integrator = VelocityVerlet::new(0.005);
    integrator.apply(&mut system);
    assert_relative_eq!(system.configuration.positions[0].x, 5.0 - 3.0e-4);
    assert_relative_eq!(system.configuration.positions[0].y, 0.0);
    assert_relative_eq!(system.configuration.positions[0].z, 0.0);
    assert_relative_eq!(system.configuration.positions[1].x, 6.0 + 3.0e-4);
    assert_relative_eq!(system.configuration.positions[1].y, 0.0);
    assert_relative_eq!(system.configuration.positions[1].z, 0.0);

    assert_relative_eq!(
        system.configuration.forces[0].x,
        -23.72772628866231,
        max_relative = 1e-12
    );
    assert_relative_eq!(system.configuration.forces[0].y, 0.0);
    assert_relative_eq!(system.configuration.forces[0].z, 0.0);
    assert_relative_eq!(
        system.configuration.forces[1].x,
        23.72772628866231,
        max_relative = 1e-12
    );
    assert_relative_eq!(system.configuration.forces[1].y, 0.0);
    assert_relative_eq!(system.configuration.forces[1].z, 0.0);

    if let Some(velocities) = system.configuration.velocities {
        assert_relative_eq!(velocities[0].x, -0.11931931572165577, max_relative = 1e-12);
        assert_relative_eq!(velocities[0].y, 0.0);
        assert_relative_eq!(velocities[0].z, 0.0);
        assert_relative_eq!(velocities[1].x, 0.11931931572165577, max_relative = 1e-12);
        assert_relative_eq!(velocities[1].y, 0.0);
        assert_relative_eq!(velocities[1].z, 0.0);
    }
    Ok(())
}
