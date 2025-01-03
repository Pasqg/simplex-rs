mod matrix;
pub mod constraints;
pub mod objective;

use constraints::{Constraint, ConstraintType};
use log::{debug, info};
use matrix::Matrix;
use objective::Objective;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::time::Instant;

const M: f64 = 1e15;
const CONSTRAINT_TRANSFORMATION_EPSILON: f64 = 0.001;

pub struct SimplexSolver<Variable: Eq + Hash + Copy> {
    variables: Vec<Variable>,
    constraints: Vec<Constraint<Variable>>,
    objective: Objective<Variable>,
}

impl<Variable: Eq + Hash + Copy + Debug> SimplexSolver<Variable> {
    pub fn new(variables: HashSet<Variable>,
               constraints: Vec<Constraint<Variable>>,
               objective: Objective<Variable>) -> Self {
        Self::validate_all_variables_are_defined(&variables, &constraints);

        Self { variables: variables.into_iter().collect(), constraints: Self::transform_constraint(constraints), objective }
    }

    pub fn solve(&self, epsilon: f64) -> HashMap<Variable, f64> {
        let constraints_size = self.constraints.len();
        let variables_size = self.variables.len();

        let constraint_matrix = self.build_constraint_matrix();

        let mut greater_equal_constraints_num = 0;
        let mut is_greater_or_equal = Vec::with_capacity(constraints_size);
        for c in 0..constraints_size {
            let constraint = &self.constraints[c];
            if constraint.constraint_type == ConstraintType::GreaterOrEqual {
                is_greater_or_equal.push(true);
                greater_equal_constraints_num += 1;
            } else {
                is_greater_or_equal.push(false);
            }
        }

        // Big-M can only be applied if the origin (all variables equal to 0) is a feasible point.
        // This condition is satisfied only when all the constraints (except non-negativity) are
        // less-than constraints and with positive constant on the right-hand side.
        let table_cols = variables_size + constraints_size + greater_equal_constraints_num + 2;
        let mut table: Matrix<f64> =
            Matrix::with_default(constraints_size + 1, table_cols, 0.0);

        for c in 0..constraints_size {
            for v in 0..variables_size {
                table[c][v] = constraint_matrix[c][v];
            }
            table[c][table_cols - 1] = constraint_matrix[c][variables_size];
        }
        for v in 0..variables_size {
            table[constraints_size][v] = self.objective.coefficients.get(&self.variables[v])
                .map_or(0.0, |x| *x);
        }

        table[constraints_size][table_cols - 1] = self.objective.constraint_term();

        let optimum = Self::big_m(&self.variables, variables_size, constraints_size, greater_equal_constraints_num, &is_greater_or_equal, table);

        Self::validate_solution(&self.variables, &self.constraints, &optimum, epsilon);
        self.make_solution(&optimum)
    }

    fn make_solution(&self, optimum: &[f64]) -> HashMap<Variable, f64> {
        let mut optimal_solution: HashMap<Variable, f64> = HashMap::new();
        for (i, optimum_i) in optimum.iter().enumerate() {
            optimal_solution.insert(self.variables[i], *optimum_i);
        }
        optimal_solution
    }

    fn build_constraint_matrix(&self) -> Matrix<f64> {
        let constraints_size = self.constraints.len();
        let variables_size = self.variables.len();

        let mut constraint_matrix: Matrix<f64> =
            Matrix::with_default(constraints_size, variables_size + 1, 0.0);
        for c in 0..constraints_size {
            let constraint = &self.constraints[c];
            let coefficients = &constraint.coefficients;
            for v in 0..variables_size {
                constraint_matrix[c][v] = coefficients.get(&self.variables[v])
                    .map_or(0.0, |x| *x);
            }
            constraint_matrix[c][variables_size] = constraint.value;
        }
        constraint_matrix
    }

    /**
     * Transforms greater/less than constraints in greater/less or equal by multiplying/dividing
     * the constraint value by CONSTRAINT_TRANSFORMATION_EPSILON.
     * Equal constraints become two greater or equal and less or equal constraints.
     */
    fn transform_constraint(constraints: Vec<Constraint<Variable>>) -> Vec<Constraint<Variable>> {
        let mut transformed_constraints = Vec::new();
        for constraint in constraints {
            let constraint_type = constraint.constraint_type;
            match constraint_type {
                ConstraintType::LessThan => transformed_constraints.push(Constraint {
                    constraint_type: constraint.constraint_type,
                    coefficients: constraint.coefficients,
                    value: constraint.value / (1.0 + CONSTRAINT_TRANSFORMATION_EPSILON),
                }),
                ConstraintType::GreaterThan => transformed_constraints.push(Constraint {
                    constraint_type: constraint.constraint_type,
                    coefficients: constraint.coefficients,
                    value: constraint.value * (1.0 + CONSTRAINT_TRANSFORMATION_EPSILON),
                }),
                ConstraintType::Equal => {
                    transformed_constraints.push(Constraint {
                        constraint_type: ConstraintType::GreaterOrEqual,
                        coefficients: constraint.coefficients.clone(),
                        value: constraint.value,
                    });
                    transformed_constraints.push(Constraint {
                        constraint_type: ConstraintType::LessOrEqual,
                        coefficients: constraint.coefficients,
                        value: constraint.value,
                    });
                }
                _ => transformed_constraints.push(constraint),
            }
        }
        transformed_constraints
    }

    fn big_m(variables_list: &[Variable], variables_num: usize, constraints_num: usize, geq_constraints_num: usize, is_greater: &[bool], original_table: Matrix<f64>) -> Vec<f64> {
        let all_variables_num = variables_num + constraints_num + geq_constraints_num;
        let objective_index = constraints_num;
        let mut table = original_table;

        let mut geq_constraint_index = 0;
        for c in 0..constraints_num {
            //surplus, slack and artificial variables transform inequalities into equalities
            if is_greater[c] {
                table[c][variables_num + c] = -1.0; //surplus variable
                table[c][variables_num + constraints_num + geq_constraint_index] = 1.0; //artificial variable
                //penalize objective for artificial variables, so that they can never be in the optimum
                table[objective_index][variables_num + constraints_num + geq_constraint_index] = -M;
                geq_constraint_index += 1;
            } else {
                table[c][variables_num + c] = 1.0; //slack variable (positive surplus)
            }
        }
        table[objective_index][all_variables_num] = 1.0;

        debug!("Initial Big-M table: {:?}", table);

        //base return of artificial variables
        if geq_constraints_num > 0 {
            let mut geq_constraint_index = 0;
            for c in 0..constraints_num {
                if is_greater[c] {
                    let pivot = table[objective_index][variables_num + constraints_num + geq_constraint_index];
                    for i in 0..table.columns {
                        table[objective_index][i] += -pivot * table[c][i];
                    }
                    geq_constraint_index += 1;
                }
                debug!("Base return of artificial variables (constraint {}): {:?}", c, table);
            }
        }

        let mut iteration = 1;
        let max_iterations = variables_num * constraints_num * 10;
        let mut bases = Vec::with_capacity(constraints_num);
        for (i, is_greater_i) in is_greater.iter().enumerate() {
            bases.push(if *is_greater_i { variables_num + constraints_num + i } else { variables_num + i });
        }

        let now = Instant::now();
        while iteration <= max_iterations {
            let mut entering_base_index = 0;
            let mut optimum_found = true;
            for i in 0..all_variables_num {
                if table[objective_index][i] > 0.0 {
                    optimum_found = false;
                }
            }
            if optimum_found {
                break;
            }

            for i in 0..all_variables_num {
                if table[objective_index][i] > table[objective_index][entering_base_index] {
                    entering_base_index = i;
                }
            }

            let mut leaving = 0;
            let mut entering;
            let mut constraint_index = 0;
            while constraint_index < constraints_num {
                if table[constraint_index][entering_base_index] > 0.0 {
                    leaving = constraint_index;
                    break;
                }
                constraint_index += 1;
            }
            entering = leaving;

            if constraint_index >= constraints_num {
                panic!("Unbounded problem, iteration {}/{}, {:.2?}", iteration, max_iterations, now.elapsed());
            }

            for c in 0..constraints_num {
                if table[c][entering_base_index] > 0.0
                    && table[c][all_variables_num + 1] / table[c][entering_base_index]
                    < table[leaving][all_variables_num + 1] / table[leaving][entering_base_index] {
                    leaving = c;
                    //if (is_greater[c]) uscente+=m;
                    entering = c;
                }
            }
            leaving = bases[leaving];
            bases[entering] = entering_base_index;
            debug!("Variable {} entering, {} leaving",
                Self::format_basis(entering_base_index, variables_num, constraints_num, variables_list),
                Self::format_basis(leaving, variables_num, constraints_num, variables_list));

            let mut new_table = Matrix::with_default(table.rows, table.columns, 0.0);
            for i in 0..new_table.columns {
                new_table[entering][i] = table[entering][i] / table[entering][entering_base_index];
            }

            for v in 0..table.rows {
                if v != entering {
                    for i in 0..table.columns {
                        new_table[v][i] = table[v][i] - new_table[entering][i] * table[v][entering_base_index];
                    }
                }
            }

            table = new_table;

            debug!("Table for iteration {}: {:?}", iteration, table);

            iteration += 1;
        }
        let elapsed = now.elapsed();

        if iteration == max_iterations {
            panic!("Max iteration number reached: {:?}", max_iterations);
        } else {
            debug!("Final table: {:?}", iteration);
            info!("Done in {:?} iterations, {:.2?} ms", iteration, elapsed.as_millis());
        }

        let mut optimum = vec![0.0; variables_num];
        for (i, optimum_i) in optimum.iter_mut().enumerate() {
            let mut i_is_base = false;
            for basis in &bases {
                if *basis == i {
                    i_is_base = true;
                }
            }
            if i_is_base {
                let mut base_row = 0;
                for k in 0..constraints_num {
                    if table[k][i] == 1.0 {
                        base_row = k;
                    }
                }
                *optimum_i = table[base_row][all_variables_num + 1];
            }
        }
        optimum
    }

    fn format_basis(base: usize, variables_num: usize, constraints_num: usize, variables_list: &[Variable]) -> String {
        if base < variables_num {
            // Normal variable
            format!("{:?}", variables_list[base])
        } else if base >= variables_num && base < constraints_num + variables_num {
            // Surplus/slack
            format!("s{:?}", base - variables_num + 1)
        } else {
            // Artificial
            format!("a{:?}", base - variables_num - constraints_num + 1)
        }
    }

    //Todo: add check for variables that are neither in the objective nor in the constraints
    fn validate_all_variables_are_defined(variables: &HashSet<Variable>, constraints: &[Constraint<Variable>]) {
        for constraint in constraints {
            for variable in constraint.coefficients.keys() {
                variables.get(variable)
                    .unwrap_or_else(|| panic!("Undefined variable {:?} from constraint {:?}", variable, constraint.coefficients));
            }
        }
    }

    fn validate_solution(variables: &[Variable],
                         constraints: &[Constraint<Variable>],
                         solution: &[f64],
                         epsilon: f64) {
        for constraint in constraints {
            let mut objective_value = 0.0;
            for i in 0..solution.len() {
                objective_value += solution[i] * constraint.coefficients.get(&variables[i]).map_or(0.0, |&x| x);
            }
            match constraint.constraint_type {
                ConstraintType::GreaterThan => {
                    if objective_value <= constraint.value {
                        panic!("Validation failed for {:?} > {:?}, got {:?}", constraint.coefficients, constraint.value, objective_value);
                    }
                }
                ConstraintType::GreaterOrEqual => {
                    if objective_value - constraint.value < -epsilon {
                        panic!("Validation failed for {:?} >= {:?}, got {:?}", constraint.coefficients, constraint.value, objective_value);
                    }
                }
                ConstraintType::LessOrEqual => {
                    if objective_value - constraint.value > epsilon {
                        panic!("Validation failed for {:?} <= {:?}, got {:?}", constraint.coefficients, constraint.value, objective_value);
                    }
                }
                ConstraintType::LessThan => {
                    if objective_value >= constraint.value {
                        panic!("Validation failed for {:?} < {:?}, got {:?}", constraint.coefficients, constraint.value, objective_value);
                    }
                }
                ConstraintType::Equal => {
                    if (objective_value - constraint.value).abs() > epsilon {
                        panic!("Validation failed for {:?} = {:?}, got {:?}", constraint.coefficients, constraint.value, objective_value);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::objective::Objective;
    use crate::constraints::{Constraint, ConstraintType};
    use crate::SimplexSolver;
    use std::collections::{HashMap, HashSet};

    #[macro_export]
    macro_rules! assert_approx_eq {
        ($left:expr, $right:expr, $delta:expr) => {{
            let (left_val, right_val, delta_val) = (&$left, &$right, &$delta);
            assert!(
                (*left_val - *right_val).abs() < *delta_val,
                "{:?} is not equal to {:?} (with delta {:?})",
                *left_val,
                *right_val,
                *delta_val
            )
        }};
    }

    #[test]
    fn test_no_constraints() {
        let x = "x";
        let y = "y";
        let solver = SimplexSolver::new(
            HashSet::from([x, y]),
            vec![],
            Objective::new(HashMap::from([(x, 1.0), (y, 1.0)]), 0.0, true));

        let solution = solver.solve(1e-7);
        assert_eq!(*solution.get(x).unwrap(), 0.0);
        assert_eq!(*solution.get(y).unwrap(), 0.0);
    }

    #[test]
    fn test_bounded_max() {
        let x = "x";
        let y = "y";
        let solver = SimplexSolver::new(
            HashSet::from([x, y]),
            vec![
                Constraint::new(HashMap::from([(x, 1.0)]), 1.0, ConstraintType::LessOrEqual),
                Constraint::new(HashMap::from([(y, 1.0)]), 1.0, ConstraintType::LessOrEqual),
            ],
            Objective::new(HashMap::from([(x, 3.0), (y, 1.0)]), 0.0, false));

        let solution = solver.solve(1e-7);
        assert_eq!(*solution.get(x).unwrap(), 1.0);
        assert_eq!(*solution.get(y).unwrap(), 1.0);
    }

    #[test]
    fn test_max_and_min() {
        let x = "x";
        let y = "y";

        let variables = HashSet::from([x, y]);
        let constraints = vec![
            Constraint::new(HashMap::from([(x, 1.0)]), 10.0, ConstraintType::LessOrEqual),
            Constraint::new(HashMap::from([(x, 1.0), (y, 1.0)]), 1.0, ConstraintType::GreaterOrEqual),
            Constraint::new(HashMap::from([(x, -0.1), (y, 1.0)]), 8.0, ConstraintType::LessOrEqual),
        ];
        let objective_coefficients = HashMap::from([(x, 3.0), (y, 1.0)]);

        let objective_max = Objective::new(objective_coefficients.clone(), 0.0, false);
        let solver = SimplexSolver::new(variables.clone(), constraints.clone(), objective_max);
        let solution = solver.solve(1e-8);
        assert_eq!(*solution.get(x).unwrap(), 10.0);
        assert_eq!(*solution.get(y).unwrap(), 9.0);

        let objective_min = Objective::new(objective_coefficients, 0.0, true);
        let solver = SimplexSolver::new(variables, constraints, objective_min);
        let solution = solver.solve(1e-8);
        assert_eq!(*solution.get(x).unwrap(), 0.0);
        assert_eq!(*solution.get(y).unwrap(), 1.0);
    }

    #[test]
    fn test_equal_constraint() {
        let x = "x";
        let y = "y";
        let x_value = 7.0;
        let variable = HashSet::from([x, y]);
        let constraints = vec![
            Constraint::new(HashMap::from([(x, 1.0)]), 10.0, ConstraintType::LessOrEqual),
            Constraint::new(HashMap::from([(x, 1.0), (y, 1.0)]), 1.0, ConstraintType::GreaterOrEqual),
            Constraint::new(HashMap::from([(x, -0.1), (y, 1.0)]), 8.0, ConstraintType::LessOrEqual),
            Constraint::new(HashMap::from([(x, 1.0)]), x_value, ConstraintType::Equal),
        ];

        let solver = SimplexSolver::new(
            variable.clone(),
            constraints.clone(),
            Objective::new(HashMap::from([(x, 3.0), (y, 1.0)]), 0.0, false));
        let solution_max = solver.solve(1e-8);
        assert_eq!(*solution_max.get(x).unwrap(), x_value);

        let solver = SimplexSolver::new(
            variable,
            constraints,
            Objective::new(HashMap::from([(x, 3.0), (y, 1.0)]), 0.0, true));
        let solution_max = solver.solve(1e-8);
        assert_eq!(*solution_max.get(x).unwrap(), x_value);
    }
}