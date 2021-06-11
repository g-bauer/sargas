pub trait Integrator {
    fn apply(&mut self, system: &mut System);
}

pub trait Thermostat {
    fn apply(&self, system: &mut System);
}

struct MolecularDynamics {
    pub integrator: Box<dyn Integrator>,
    pub thermostat: Option<Box<dyn Thermostat>>
}