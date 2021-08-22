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

fn get_gradients(
    layer_count: usize,
    weights: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
    expected: &DMatrix,
) -> Vec<(FLOAT, FLOAT)> {
    let outputs = activations.last().unwrap();
    let del_cost_wrt_layer_activation_for_last_layer =
        del_cost_wrt_layer_activation_for_last_layer(outputs, &expected);
    let del_cost_wrt_neuron_values_for_last_layer =
        del_cost_wrt_layer_activation_for_last_layer.component_mul(&del_activation_wrt_neuron_values(outputs));
    let penultimate_layer_activations = &activations[layer_count - 2];
    println!("{}", penultimate_layer_activations);
    println!("{}", del_cost_wrt_neuron_values_for_last_layer);
    let del_cost_wrt_weight_for_last_layer =
        del_cost_wrt_neuron_values_for_last_layer.dot(&penultimate_layer_activations);
    let del_cost_wrt_bias_for_last_layer = del_cost_wrt_neuron_values_for_last_layer.sum();
    let (_, gradients) = (1..layer_count - 1).rev().fold(
        (
            del_cost_wrt_neuron_values_for_last_layer,
            vec![(
                del_cost_wrt_weight_for_last_layer,
                del_cost_wrt_bias_for_last_layer,
            )],
        ),
        |(last_del_cost_wrt_neuron_values, mut gradients), layer| {
            let outgoing_weights = weights[layer].transpose();
            let neuron_activations = &activations[layer];
            let del_cost_wrt_neuron_values = last_del_cost_wrt_neuron_values.dot(&outgoing_weights)
                * del_activation_wrt_neuron_values(neuron_activations);
            let last_activations = &activations[layer - 1];
            let del_cost_wrt_weight = del_cost_wrt_neuron_values.dot(&last_activations.transpose());
            let del_cost_wrt_bias = del_cost_wrt_neuron_values.sum();
            gradients.push((del_cost_wrt_weight, del_cost_wrt_bias));
            (del_cost_wrt_neuron_values, gradients)
        },
    );
    gradients.into_iter().rev().collect()
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

fn main() {
    const INPUT_SIZE: usize = 2;
    const HIDDEN_SIZE: usize = 3;
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
        let gradients = get_gradients(LAYER_COUNT, &weights, &activations, &expected);
        for ((weight, bias),(gradient_weight, gradient_bias)) in weights.iter_mut().zip(biases.iter_mut()).zip(gradients.iter()) {
            weight.add_scalar_mut(gradient_weight * LEARNING_RATE);
            bias.add_scalar_mut(gradient_bias * LEARNING_RATE);
        }
    }
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
}
