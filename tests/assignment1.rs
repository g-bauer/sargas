use sargas::configuration::Configuration;
use sargas::error::SargasError;
use sargas::lennard_jones::LennardJones;
use sargas::system::System;
use std::path::Path;
use std::rc::Rc;

#[cfg(chemfiles)]
pub fn get_system(name: &str, rc: f64, box_length: f64) -> Result<System, SargasError> {
    let path = Path::new(file!()).parent().unwrap().join(name);
    let potential = Rc::new(LennardJones::new(1.0, 1.0, rc, true));
    let configuration = Configuration::from_file(path, box_length, 800).unwrap();
    System::new(configuration, potential)
}

#[cfg(chemfiles)]
mod assignment1 {
    use super::get_system;
    use approx::assert_relative_eq;
    use sargas::error::SargasError;
    use sargas::lennard_jones::LennardJones;
    use sargas::vec::Vec3;

    fn round_to(x: f64, places: i32) -> f64 {
        let f = 10.0f64.powi(places);
        (x * f).round() / f
    }

    #[test]
    fn nearest_image_convention() {
        let r1 = Vec3::new(1.0, 0.0, 0.0);
        let r2 = Vec3::new(5.0, 0.0, 0.0);
        let distance = (r2 - r1).nearest_image(6.0).norm2();
        assert_relative_eq!(distance, 2.0);

        let r1 = Vec3::new(1.0, 0.0, 0.0);
        let r2 = Vec3::new(2.0, 0.0, 0.0);
        let distance = (r2 - r1).nearest_image(6.0).norm2();
        assert_relative_eq!(distance, 1.0)
    }

    #[test]
    fn periodic_boundary_condition() {
        let mut r1 = Vec3::new(6.0, 6.0, 6.0);
        r1.apply_pbc(5.0);
        assert_relative_eq!(r1.x, 1.0);
        assert_relative_eq!(r1.y, 1.0);
        assert_relative_eq!(r1.y, 1.0);

        let mut r1 = Vec3::new(3.0, 3.0, 3.0);
        r1.apply_pbc(5.0);
        assert_relative_eq!(r1.x, 3.0);
        assert_relative_eq!(r1.y, 3.0);
        assert_relative_eq!(r1.y, 3.0);
    }

    #[test]
    fn lennard_jones_virial() {
        let lj = LennardJones::new(1.0, 1.0, 3.0, false);
        let r_min = 2.0f64.powf(1.0 / 6.0);
        assert_relative_eq!(lj.energy(r_min.powi(2)), -1.0);
        assert_relative_eq!(lj.virial(r_min.powi(2)), 0.0);
        assert_relative_eq!(lj.virial(9.0), -0.03283149023127684);
    }

    #[test]
    fn lennard_jones_shift() {
        let lj = LennardJones::new_shifted(1.0, 1.0, 3.0, false);
        let r_min = 2.0f64.powf(1.0 / 6.0);
        assert_relative_eq!(lj.virial(r_min.powi(2)), 0.0);
        assert_relative_eq!(lj.virial(9.0), -0.03283149023127684);

        assert_relative_eq!(lj.energy(3.0_f64.powi(2)), 0.0);
        assert_relative_eq!(lj.energy(2.5_f64.powi(2)), -0.010837449391761223);
    }

    #[test]
    fn lennard_jones_energy_tail() {
        let density = 0.8;
        let nparticles = 1337;
        let sigma = 1.234;
        let epsilon = 5.67;
        let rc = 3.0 * sigma;

        let lj = LennardJones::new(sigma, epsilon, rc, true);

        assert_relative_eq!(
            lj.energy_tail(density, nparticles),
            -3534.32227313446,
            max_relative = 1e-12
        );
    }

    #[test]
    fn lennard_jones_pressure_tail() {
        let density = 0.8;
        let sigma = 1.234;
        let epsilon = 5.67;
        let rc = 3.0 * sigma;

        let lj = LennardJones::new(sigma, epsilon, rc, true);

        assert_relative_eq!(
            lj.pressure_tail(density),
            -4.227620612464192,
            max_relative = 1e-12
        );
    }

    #[test]
    fn nist1() -> Result<(), SargasError> {
        let system = get_system("lj1.xyz", 3.0, 10.0)?;
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 1), -4351.5, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 2), -568.67, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 2), -198.49, max_relative = 1e-15);
        Ok(())
    }

    #[test]
    fn nist2() -> Result<(), SargasError> {
        let system = get_system("lj2.xyz", 3.0, 8.0)?;
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 2), -690.00, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 2), -568.46, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 3), -24.230, max_relative = 1e-15);
        Ok(())
    }

    #[test]
    fn nist3() -> Result<(), SargasError> {
        let system = get_system("lj3.xyz", 3.0, 10.0)?;
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 1), -1146.7, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 1), -1164.9, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 3), -49.622, max_relative = 1e-15);
        Ok(())
    }

    #[test]
    fn nist4() -> Result<(), SargasError> {
        let system = get_system("lj4.xyz", 3.0, 8.0)?;
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        let (u, v) = system.energy_virial();
        assert_relative_eq!(round_to(u - u_tail, 3), -16.790, max_relative = 1e-15);
        assert_relative_eq!(round_to(v, 3), -46.249, max_relative = 1e-15);
        assert_relative_eq!(round_to(u_tail, 5), -0.54517, max_relative = 1e-15);
        Ok(())
    }
}
