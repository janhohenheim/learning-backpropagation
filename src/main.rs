use nalgebra::{SMatrix, SVector};
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

type FLOAT = f32;
type Vector<const SIZE: usize> = SVector<FLOAT, SIZE>;
type DVector = nalgebra::DVector<FLOAT>;
type DMatrix = nalgebra::DMatrix<FLOAT>;
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
fn del_cost_wrt_layer_activation_for_last_layer(actual: &DMatrix, expected: &DMatrix) -> DMatrix {
    (expected - actual).abs() * 2.0
}

fn del_activation_wrt_neuron_values<const SIZE: usize>(
    neuron_values: &Vector<SIZE>,
) -> Vector<SIZE> {
    neuron_values.map(d_sigmoid)
}

fn to_dynamic<const ROWS: usize, const COLS: usize>(matrix: Matrix<ROWS, COLS>) -> DMatrix {
    matrix.resize(ROWS, COLS, 0.0)
}

fn del_cost_wrt_neuron_activation(
    layer_count: usize,
    weights: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
) -> Vec<DMatrix> {
    let outputs = activations.last().unwrap();
    let expected = DMatrix::zeros(outputs.len(), 1);
    let del_cost_wrt_layer_activation_for_last_layer =
        del_cost_wrt_layer_activation_for_last_layer(outputs, &expected);
    (0..(layer_count - 1)).rev().fold(
        vec![del_cost_wrt_layer_activation_for_last_layer],
        |mut acc, layer| {
            let outgoing_weights = weights[layer].transpose();
            let next_activations = &activations[layer + 1];
            let last_layer_del_cost_wrt_neuron_activation = acc.first().unwrap().sum();
            acc.insert(
                0,
                outgoing_weights * next_activations * last_layer_del_cost_wrt_neuron_activation,
            );
            acc
        },
    )
}

fn main() {
    const INPUT_SIZE: usize = 2;
    const HIDDEN_SIZE: usize = 3;
    const OUTPUT_SIZE: usize = 5;
    const HIDDEN_LAYER_COUNT: usize = 1;
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

    let activations = vec![to_dynamic(inputs), to_dynamic(hidden), to_dynamic(outputs)];
    let biases = vec![to_dynamic(hidden_biases), to_dynamic(output_biases)];
    let weights = vec![
        to_dynamic(input_to_hidden_weights),
        to_dynamic(hidden_to_output_weights),
    ];
    const LAYER_COUNT: usize = 2 + HIDDEN_LAYER_COUNT;

    let del_cost_wrt_neuron_activation =
        del_cost_wrt_neuron_activation(LAYER_COUNT, &weights, &activations);
    let weight_gradient = (1..LAYER_COUNT)
        .rev()
        .fold(Vec::new(), |mut acc, layer| {
            let current_activations = &activations[layer];
            let last_activations = &activations[layer - 1];
            let layer_gradient = DMatrix::from_fn(
                weights[layer - 1].nrows(),
                weights[layer - 1].ncols(),
                |j, k| {
                    let del_cost_wrt_neuron_activation_for_current_layer =
                        if layer == LAYER_COUNT - 1 {
                            &del_cost_wrt_neuron_activation[layer][j]
                        } else {
                            &del_cost_wrt_neuron_activation[layer + 1][k]
                        };
                    &last_activations[k]
                        * d_sigmoid(current_activations[j])
                        * del_cost_wrt_neuron_activation_for_current_layer
                },
            );
            acc.push(layer_gradient);
            acc
        })
        .iter()
        .rev();
}
