use nalgebra::{SMatrix, SVector};
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

type FLOAT = f32;
type Vector<const SIZE: usize> = SVector<FLOAT, SIZE>;
type Matrix<const ROWS: usize, const COLS: usize> = SMatrix<FLOAT, ROWS, COLS>;

const INITIAL_VALUE_OFFSET: FLOAT = 1.0;
const INITIAL_VALUE_RANGE: Range<FLOAT> = 0.0 - INITIAL_VALUE_OFFSET..1.0 + INITIAL_VALUE_OFFSET;

fn sigmoid(n: FLOAT) -> FLOAT {
    1.0 / (1.0 + E.powf(n))
}

fn generate_number(range: Range<FLOAT>) -> FLOAT {
    rand::thread_rng().gen_range(range)
}

fn generate_matrix<const ROWS: usize, const COLS: usize>() -> Matrix<ROWS, COLS> {
    Matrix::from_fn(|_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

fn generate_vector<const SIZE: usize>() -> Vector<SIZE> {
    Vector::from_fn(|_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

fn get_neuron_values<const LAST_SIZE: usize, const NEXT_SIZE: usize>(
    last_layer: Vector<LAST_SIZE>,
    weights: Matrix<NEXT_SIZE, LAST_SIZE>,
    biases: Vector<NEXT_SIZE>,
) -> Vector<NEXT_SIZE> {
    weights * last_layer + biases
}

fn run_activation_function<const SIZE: usize>(neuron_values: Vector<SIZE>) -> Vector<SIZE> {
    neuron_values.map(sigmoid)
}

fn cost<const SIZE: usize>(actual: Vector<SIZE>, expected: Vector<SIZE>) -> FLOAT {
    expected
        .iter()
        .zip(actual.iter())
        .fold(0.0, |acc, (expected, actual)| {
            acc + (expected - actual).powf(2.0)
        })
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

    let hidden = run_activation_function(get_neuron_values(
        inputs,
        input_to_hidden_weights,
        hidden_biases,
    ));
    let outputs = run_activation_function(get_neuron_values(
        hidden,
        hidden_to_output_weights,
        output_biases,
    ));
    println!("{}", outputs);
}
