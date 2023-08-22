use std::collections::HashMap;
use std::hash::Hash;

pub struct Objective<Variable> {
    pub(crate) coefficients: HashMap<Variable, f64>,
    constant_term: f64,
    minimize: bool,
}

impl<Variable: Eq + Hash> Objective<Variable> {
    pub fn new(coefficients: HashMap<Variable, f64>, constant_term: f64, minimize: bool) -> Self {
        let constant_term = if minimize { -constant_term } else { constant_term };
        let coefficients =
            if minimize {
                coefficients.into_iter()
                    .map(|(k, v)| {
                        (k, -v)
                    }).collect()
            } else { coefficients };
        Self { coefficients, constant_term, minimize }
    }

    pub fn constraint_term(&self) -> f64 {
        if self.minimize { -self.constant_term } else { self.constant_term }
    }
}