use nalgebra::{SMatrix, SVector};
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

type FLOAT = f32;
type Vector<const SIZE: usize> = SVector<FLOAT, SIZE>;
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

fn get_neuron_values(last_layer: &DMatrix, weights: &DMatrix, biases: &DMatrix) -> DMatrix {
    weights * last_layer + biases
}

fn run_activation_function(neuron_values: DMatrix) -> DMatrix {
    neuron_values.map(sigmoid)
}

/// Cost function is (expected - actual)^2
fn del_cost_wrt_layer_activation_for_last_layer(actual: &DMatrix, expected: &DMatrix) -> DMatrix {
    2.0 * (expected - actual)
}

fn del_activation_wrt_neuron_values(neuron_values: &DMatrix) -> DMatrix {
    neuron_values.map(d_sigmoid)
}

fn to_dynamic<const ROWS: usize, const COLS: usize>(matrix: Matrix<ROWS, COLS>) -> DMatrix {
    matrix.resize(ROWS, COLS, 0.0)
}

fn del_cost_wrt_neuron_activation(
    layer_count: usize,
    weights: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
    expected: &DMatrix,
) -> Vec<DMatrix> {
    let outputs = activations.last().unwrap();
    let del_cost_wrt_layer_activation_for_last_layer =
        del_cost_wrt_layer_activation_for_last_layer(outputs, &expected);
    (0..(layer_count - 1)).rev().fold(
        vec![del_cost_wrt_layer_activation_for_last_layer],
        |mut acc, layer| {
            let outgoing_weights = weights[layer].transpose();
            let next_activations = &activations[layer + 1];
            let del_activation_wrt_neuron_values =
                del_activation_wrt_neuron_values(next_activations);
            let last_layer_del_cost_wrt_neuron_activation = acc.first().unwrap().sum();
            acc.insert(
                0,
                outgoing_weights
                    * del_activation_wrt_neuron_values
                    * last_layer_del_cost_wrt_neuron_activation,
            );
            acc
        },
    )
}

fn get_gradient(
    layer_count: usize,
    weights: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
    del_cost_wrt_neuron_activation: &Vec<DMatrix>,
    is_bias: bool,
) -> Vec<DMatrix> {
    (1..layer_count)
        .rev()
        .fold(Vec::new(), |mut acc, layer| {
            let current_activations = &activations[layer];
            let last_activations = &activations[layer - 1];
            let rows = weights[layer - 1].nrows();
            let columns = if is_bias {
                1
            } else {
                weights[layer - 1].ncols()
            };
            let layer_gradient = DMatrix::from_fn(rows, columns, |j, k| {
                let del_cost_wrt_neuron_activation_for_current_layer = if layer == layer_count - 1 {
                    &del_cost_wrt_neuron_activation[layer][j]
                } else {
                    &del_cost_wrt_neuron_activation[layer + 1][k]
                };
                let last_activation = if is_bias { 1.0 } else { last_activations[k] };
                last_activation
                    * d_sigmoid(current_activations[j])
                    * del_cost_wrt_neuron_activation_for_current_layer
            });
            acc.push(layer_gradient);
            acc
        })
        .into_iter()
        .rev()
        .collect()
}

fn get_activations(
    layer_count: usize,
    inputs: &DMatrix,
    weights: &Vec<DMatrix>,
    biases: &Vec<DMatrix>,
) -> Vec<DMatrix> {
    (0..layer_count - 1).fold(vec![inputs.clone()], |mut acc, layer| {
        let activation = run_activation_function(get_neuron_values(
            &acc[layer],
            &weights[layer],
            &biases[layer],
        ));
        acc.push(activation);
        acc
    })
}

fn get_gradients(
    layer_count: usize,
    weights: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
    expected: &DMatrix,
) -> (Vec<DMatrix>, Vec<DMatrix>) {
    let del_cost_wrt_neuron_activation =
        del_cost_wrt_neuron_activation(layer_count, &weights, &activations, expected);
    let weight_gradient = get_gradient(
        layer_count,
        &weights,
        &activations,
        &del_cost_wrt_neuron_activation,
        false,
    );
    let bias_gradient = get_gradient(
        layer_count,
        &weights,
        &activations,
        &del_cost_wrt_neuron_activation,
        true,
    );
    (weight_gradient, bias_gradient)
}

fn main() {
    const INPUT_SIZE: usize = 2;
    const HIDDEN_SIZE: usize = 256;
    const OUTPUT_SIZE: usize = 5;
    const HIDDEN_LAYER_COUNT: usize = 1;
    const LAYER_COUNT: usize = 2 + HIDDEN_LAYER_COUNT;
    const LEARNING_RATE: FLOAT = 0.001;

    let input_to_hidden_weights = to_dynamic(generate_matrix::<HIDDEN_SIZE, INPUT_SIZE>());
    let hidden_to_output_weights = to_dynamic(generate_matrix::<OUTPUT_SIZE, HIDDEN_SIZE>());

    let mut weights =
        (0..HIDDEN_LAYER_COUNT - 1).fold(vec![input_to_hidden_weights], |mut acc, _layer| {
            acc.push(to_dynamic(generate_matrix::<HIDDEN_SIZE, HIDDEN_SIZE>()));
            acc
        });
    weights.push(hidden_to_output_weights);

    let mut biases = (0..HIDDEN_LAYER_COUNT).fold(Vec::new(), |mut acc, _layer| {
        acc.push(to_dynamic(generate_vector::<HIDDEN_SIZE>()));
        acc
    });
    biases.push(to_dynamic(generate_vector::<OUTPUT_SIZE>()));
    let inputs = to_dynamic(generate_vector::<INPUT_SIZE>());
    let expected = DMatrix::from_fn(OUTPUT_SIZE, 1, |i, _j| i as FLOAT);
    let mut outputs = Vec::new();
    for _epoch in 0..100000 {
        let activations = get_activations(LAYER_COUNT, &inputs, &weights, &biases);
        outputs.push(activations.last().unwrap().clone());
        let (weight_gradient, bias_gradient) =
            get_gradients(LAYER_COUNT, &weights, &activations, &expected);
        for (weight, gradient) in weights.iter_mut().zip(weight_gradient) {
            *weight += &gradient * LEARNING_RATE;
        }
        for (bias, gradient) in biases.iter_mut().zip(bias_gradient) {
            *bias += &gradient * LEARNING_RATE;
        }
    }
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
}
