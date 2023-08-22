use std::collections::HashMap;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ConstraintType {
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
    Equal,
}

#[derive(Clone)]
pub struct Constraint<Variable> {
    pub(crate) coefficients: HashMap<Variable, f64>,
    pub(crate) value: f64,
    pub(crate) constraint_type: ConstraintType,
}

impl<Variable> Constraint<Variable> {
    pub fn new(coefficients: HashMap<Variable, f64>, value: f64, constraint_type: ConstraintType) -> Self {
        Self { coefficients, value, constraint_type }
    }

    pub fn geq(coefficients: HashMap<Variable, f64>, value: f64) -> Self {
        Self::new(coefficients, value, ConstraintType::GreaterOrEqual)
    }

    pub fn leq(coefficients: HashMap<Variable, f64>, value: f64) -> Self {
        Self::new(coefficients, value, ConstraintType::LessOrEqual)
    }
}