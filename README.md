# simplex-rs

A simple crate to calculate the optimal solution of linear programming
problems using the simplex (big-M) method.

## Usage examples

Take the following problem:

```
Find the max of f(x, y) = 3x + y

with:
    x <= 1
    y <= 1
```

The maximum of the function can be easily found by taking the maximum
values allowed for each variable `x=1`, `y=1` which yields `f(1, 1) = 4`.

In simplex-rs, the problem can be formulated as:
```
let x = "x";
let y = "y";

let variables = HashSet::from([x, y]);
let constraints = vec![
        Constraint::new(HashMap::from([(x, 1.0)]), 1.0, ConstraintType::LessOrEqual),
        Constraint::new(HashMap::from([(y, 1.0)]), 1.0, ConstraintType::LessOrEqual),
    ]'
let objective = Objective::new(HashMap::from([(x, 3.0), (y, 1.0)]), 0.0, false);
```

Then a solution can be find by calling ```SimplexSolver::solve``` with the
specified epsilon (used only for validation of the final solution).

```
let solver = SimplexSolver::new(variables, constraints, objective);
let solution = solver.solve(1e-7);
assert_eq!(*solution.get(x).unwrap(), 1.0);
assert_eq!(*solution.get(y).unwrap(), 1.0);
```

## License

This repository is licensed under [Apache 2.0 License](LICENSE).