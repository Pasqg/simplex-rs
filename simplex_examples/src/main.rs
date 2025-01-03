use simplex_rs::{
    constraints::{Constraint, ConstraintType},
    objective::Objective,
    SimplexSolver,
};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

//Per gram
#[derive(Debug, Copy, Clone, PartialEq)]
struct NutritionInfo {
    calories: f64,
    protein: f64,
    carbs: f64,
    fat: f64,
    fiber: f64,
}

impl NutritionInfo {
    pub fn normalize(&self, grams: f64) -> Self {
        NutritionInfo {
            calories: self.calories / grams,
            protein: self.protein / grams,
            carbs: self.carbs / grams,
            fat: self.fat / grams,
            fiber: self.fiber / grams,
        }
    }
}

fn create_constraints<Var>(
    ingredients: &HashMap<Var, NutritionInfo>,
    min_quantities: &HashMap<Var, f64>,
    meal_info: &NutritionInfo,
) -> Vec<Constraint<Var>>
where
    Var: Eq + Hash + Clone + Debug,
{
    let mut constraints = Vec::new();
    let epsilon = 0.04; //3%
    let nutrient_fns = vec![
        |x: &NutritionInfo| x.calories,
        |x: &NutritionInfo| x.protein,
        |x: &NutritionInfo| x.carbs,
        |x: &NutritionInfo| x.fat,
        |x: &NutritionInfo| x.fiber,
    ];

    for func in nutrient_fns {
        let mut weights = HashMap::new();

        for (ingredient, info) in ingredients {
            weights.insert(ingredient.to_owned(), func(info));
        }

        constraints.push(Constraint::new(
            weights.clone(),
            func(meal_info) * (1.0 - epsilon),
            ConstraintType::GreaterOrEqual,
        ));
        constraints.push(Constraint::new(
            weights,
            func(meal_info) * (1.0 + epsilon),
            ConstraintType::LessOrEqual,
        ));
    }

    for (var, min_q) in min_quantities {
        constraints.push(Constraint::new(
            HashMap::from([(var.to_owned(), 1.0)]),
            min_q.to_owned(),
            ConstraintType::GreaterOrEqual,
        ));
    } 

    return constraints;
}

fn main() {
    let ingredients = [
        (
            "fried chicken",
            NutritionInfo {
                calories: 283.4,
                protein: 20.5,
                carbs: 7.5,
                fat: 19.0,
                fiber: 0.0,
            }
            .normalize(127.0),
        ),
        (
            "rice",
            NutritionInfo {
                calories: 349.0,
                protein: 7.0,
                carbs: 78.0,
                fat: 0.2,
                fiber: 0.8,
            }
            .normalize(100.0),
        ),
        (
            "oil",
            NutritionInfo {
                calories: 82.0,
                protein: 0.0,
                carbs: 0.0,
                fat: 9.1,
                fiber: 0.0,
            }
            .normalize(10.0),
        ),
        (
            "tuna",
            NutritionInfo {
                calories: 86.0,
                protein: 19.4,
                carbs: 0.0,
                fat: 1.0,
                fiber: 0.0,
            }
            .normalize(100.0),
        ),
        (
            "noodles",
            NutritionInfo {
                calories: 254.0,
                protein: 6.2,
                carbs: 40.0,
                fat: 7.4,
                fiber: 1.0,
            }
            .normalize(62.0),
        ),
        (
            "white beans",
            NutritionInfo {
                calories: 105.0,
                protein: 6.9,
                carbs: 14.0,
                fat: 0.8,
                fiber: 3.8,
            }
            .normalize(100.0),
        ),
    ];

    let meal_nutrition = NutritionInfo {
        calories: 1039.0,
        protein: 34.3,
        carbs: 178.5,
        fat: 24.0,
        fiber: 11.2,
    };

    let min_quantities = HashMap::from([
        ("oil", 10.0),
        ("tuna", 20.0),
        ("rice", 30.0),
        ("noodles", 30.0),
        ("white beans", 100.0),
    ]);

    let solver = SimplexSolver::new(
        HashSet::from(ingredients.map(|(x, _)| x)),
        create_constraints(&HashMap::from(ingredients), &min_quantities, &meal_nutrition),
        Objective::new(HashMap::from(ingredients.map(|(x, _)| (x, 1.0))), 0.0, true),
    );
    let solution_max = solver.solve(1e-8);
    println!("Ingredient grams {:?}", solution_max);
}
