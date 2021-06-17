use sargas::configuration::Configuration;
use sargas::potential::LennardJones;
use sargas::system::System;
use std::path::Path;
use std::rc::Rc;

pub fn get_system(name: &str, rc: f64, box_length: f64) -> System {
    let path = Path::new(file!()).parent().unwrap().join(name);
    let potential = Rc::new(LennardJones::new(1.0, 1.0, rc, true));
    let configuration = Configuration::from_file(path, box_length, 800).unwrap();
    System::new(configuration, potential)
}

mod assignment1 {
    use super::get_system;
    use approx::assert_relative_eq;
    use sargas::potential::{LennardJones, Potential};
    use sargas::vec::Vec3;

    fn round_to(x: f64, places: i32) -> f64 {
        let f = 10.0f64.powi(places);
        (x * f).round() / f
    }

    #[test]
    fn nearest_image_distance() {
        let r1 = Vec3::new(1.0, 0.0, 0.0);
        let r2 = Vec3::new(5.0, 0.0, 0.0);
        let distance = (r2 - r1).nearest_image(6.0).norm2();
        assert_relative_eq!(distance, 2.0)
    }

    #[test]
    fn periodic_boundary_conditions() {
        let mut r1 = Vec3::new(6.0, 6.0, 6.0);
        r1.apply_pbc(5.0);
        assert_relative_eq!(r1.x, 1.0);
        assert_relative_eq!(r1.y, 1.0);
        assert_relative_eq!(r1.y, 1.0);
    }

    #[test]
    fn lennard_jones() {
        let lj: Box<dyn Potential> = Box::new(LennardJones::new(1.0, 1.0, 3.0, false));
        let r_min = 2.0f64.powf(1.0 / 6.0);
        assert_relative_eq!(lj.energy(r_min.powi(2)), -1.0);
        assert_relative_eq!(lj.virial(r_min.powi(2)), 0.0);
    }

    #[test]
    fn nist1() {
        let system = get_system("lj1.xyz", 3.0, 10.0);
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 1), -4351.5, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 2), -568.67, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 2), -198.49, max_relative = 1e-15);
    }

    #[test]
    fn nist2() {
        let system = get_system("lj2.xyz", 3.0, 8.0);
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 2), -690.00, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 2), -568.46, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 3), -24.230, max_relative = 1e-15);
    }

    #[test]
    fn nist3() {
        let system = get_system("lj3.xyz", 3.0, 10.0);
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 1), -1146.7, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 1), -1164.9, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 3), -49.622, max_relative = 1e-15);
    }

    #[test]
    fn nist4() {
        let system = get_system("lj4.xyz", 3.0, 8.0);
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 3), -16.790, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 3), -46.249, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 5), -0.54517, max_relative = 1e-15);
    }
}
