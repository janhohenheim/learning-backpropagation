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
    1.0 / (1.0 + E.powf(-n))
}

/// Requires that sigmoid is already sigmoid(x)
fn d_sigmoid(sigmoid: FLOAT) -> FLOAT {
    sigmoid * (1.0 - sigmoid)
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
    last_layer: &Vector<LAST_SIZE>,
    weights: &Matrix<NEXT_SIZE, LAST_SIZE>,
    biases: &Vector<NEXT_SIZE>,
) -> Vector<NEXT_SIZE> {
    weights * last_layer + biases
}

fn run_activation_function<const SIZE: usize>(neuron_values: Vector<SIZE>) -> Vector<SIZE> {
    neuron_values.map(sigmoid)
}

/// Cost function is (expected - actual)^2
fn del_cost_wrt_layer_activation_for_last_layer<const SIZE: usize>(
    actual: &Vector<SIZE>,
    expected: &Vector<SIZE>,
) -> Vector<SIZE> {
    (expected - actual).abs() * 2.0
}

/// Cost function is (expected - actual)^2
fn del_cost_wrt_layer_activation_for_not_last_layer<
    const INPUT_SIZE: usize,
    const OUTPUT_SIZE: usize,
>(
    next_weights: &Matrix<OUTPUT_SIZE, INPUT_SIZE>,
    next_neuron_values: &Vector<OUTPUT_SIZE>,
    next_del_cost_wrt_layer_activation: &Vector<OUTPUT_SIZE>,
) -> Vector<INPUT_SIZE> {
    (expected - actual).abs() * 2.0
}

fn del_activation_wrt_neuron_values<const SIZE: usize>(
    neuron_values: &Vector<SIZE>,
) -> Vector<SIZE> {
    neuron_values.map(d_sigmoid)
}

fn del_cost_wrt_neuron_values<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize>(
    actual: &Vector<OUTPUT_SIZE>,
    expected: &Vector<OUTPUT_SIZE>,
) -> Vector<OUTPUT_SIZE> {
    // Hadamard product
    del_cost_wrt_layer_activation(actual, expected)
        .component_mul(&del_activation_wrt_neuron_values(actual))
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
        &inputs,
        &input_to_hidden_weights,
        &hidden_biases,
    ));
    let outputs = run_activation_function(get_neuron_values(
        &hidden,
        &hidden_to_output_weights,
        &output_biases,
    ));

    let expected = Vector::<OUTPUT_SIZE>::zeros();
    let del_cost_wrt_neuron_values = del_cost_wrt_neuron_values(&outputs, &expected);
    let hidden_to_output_gradients =
        Matrix::from_fn(|i, j| del_cost_wrt_neuron_values[i] * hidden[j]);
    let input_to_hidden_gradients = Matrix::from_fn(|i, j| {
        hidden_to_output_gradients[(i, j)]
            * hidden_to_output_weights[(i, j)]
            * del_sigmoid(inputs[j])
    });

    const LEARNING_RATE: FLOAT = 0.05;
    let new_hidden_to_output_weights =
        hidden_to_output_weights - hidden_to_output_gradients * LEARNING_RATE;
    let new_input_to_hidden_weights =
        input_to_hidden_weights - input_to_hidden_gradients * LEARNING_RATE;
}
