use sargas::configuration::Configuration;
use sargas::error::SargasError;
use sargas::lennard_jones::LennardJones;
use sargas::system::System;
use std::path::Path;
use std::rc::Rc;

pub fn get_system(name: &str, rc: f64, box_length: f64) -> Result<System, SargasError> {
    let path = Path::new(file!()).parent().unwrap().join(name);
    let potential = Rc::new(LennardJones::new(1.0, 1.0, rc, true));
    let configuration = Configuration::from_file(path, box_length, 800).unwrap();
    System::new(configuration, potential)
}

mod assignment3 {
    use super::get_system;
    use approx::assert_relative_eq;
    use sargas::configuration::Configuration;
    use sargas::error::SargasError;
    use sargas::lennard_jones::LennardJones;
    use sargas::propagator::molecular_dynamics::velocity_verlet::VelocityVerlet;
    use sargas::propagator::molecular_dynamics::Integrator;
    use sargas::propagator::Propagator;
    use sargas::system::System;
    use sargas::vec::Vec3;
    use std::rc::Rc;

    fn round_to(x: f64, places: i32) -> f64 {
        let f = 10.0f64.powi(places);
        (x * f).round() / f
    }

    #[test]
    fn velocity_verlet() -> Result<(), SargasError> {
        let potential = Rc::new(LennardJones::new(1.0, 1.0, 3.5, true));
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
}
