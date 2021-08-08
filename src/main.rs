use nalgebra::{Matrix2, SMatrix, Vector2};
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

fn sigmoid(n: f32) -> f32 {
    1.0 / (1.0 + E.powf(n))
}

fn generate_number(range: Range<f32>) -> f32 {
    rand::thread_rng().gen_range(range)
}

fn main() {
    let big_matrix = SMatrix::<f32, 4, 4>::from_fn(|_i, _j| generate_number(0.0..1.0));
    println!("{}", big_matrix);
    let sigmoid_matrix = big_matrix.map(sigmoid);
    println!("{}", sigmoid_matrix);

    let linear_transformation_matrix = Matrix2::new(1, 3, 2, 1);
    let original_vector = Vector2::new(4, 2);
    let transformed_vector = linear_transformation_matrix * original_vector;
    println!("{}", transformed_vector);

    let ninety_degree_spin = Matrix2::new(0, -1, 1, 0);
    let shear = Matrix2::new(1, 2, 0, 1);
    let combined_matrix = ninety_degree_spin * shear;
    println!("{}", combined_matrix);
}
