use nalgebra::{SMatrix, SVector};
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

type FLOAT = f32;
type Vector<const SIZE: usize> = SVector<FLOAT, SIZE>;
type Matrix<const ROWS: usize, const COLS: usize> = SMatrix<FLOAT, ROWS, COLS>;

fn sigmoid(n: FLOAT) -> FLOAT {
    1.0 / (1.0 + E.powf(n))
}

fn generate_number(range: Range<FLOAT>) -> FLOAT {
    rand::thread_rng().gen_range(range)
}

fn generate_matrix<const ROWS: usize, const COLS: usize>() -> Matrix<ROWS, COLS> {
    Matrix::from_fn(|_i, _j| generate_number(0.0..1.0))
}

fn generate_vector<const SIZE: usize>() -> Vector<SIZE> {
    Vector::from_fn(|_i, _j| generate_number(0.0..1.0))
}

fn activate_layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize>(
    inputs: Vector<INPUT_SIZE>,
    weights: Matrix<OUTPUT_SIZE, INPUT_SIZE>,
    biases: Vector<OUTPUT_SIZE>,
) -> Vector<OUTPUT_SIZE> {
    (weights * inputs + biases).map(sigmoid)
}

fn main() {
    const INPUT_SIZE: usize = 2;
    const HIDDEN_SIZE: usize = 3;
    const OUTPUT_SIZE: usize = 5;
    let input_to_hidden_weights = generate_matrix::<HIDDEN_SIZE, INPUT_SIZE>();
    let hidden_to_output_weights = generate_matrix::<OUTPUT_SIZE, HIDDEN_SIZE>();
    let hidden_biases = generate_vector::<HIDDEN_SIZE>();
    let output_biases = generate_vector::<OUTPUT_SIZE>();

    let inputs = generate_vector::<INPUT_SIZE>();

    let biases = activate_layer(inputs, input_to_hidden_weights, hidden_biases);
    let outputs = activate_layer(biases, hidden_to_output_weights, output_biases);
    println!("{}", outputs);
}
