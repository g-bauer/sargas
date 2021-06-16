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

mod tests {
    use super::get_system;
    use approx::assert_ulps_eq;
    #[test]
    fn nist1() {
        let system = get_system("lj1.xyz", 3.0, 10.0);
        let u_tail = system
            .potential
            .energy_tail(system.density(), system.configuration.nparticles);
        assert_ulps_eq!(system.energy() - u_tail, -4.3515E+03, max_ulps = 0);
    }
}
