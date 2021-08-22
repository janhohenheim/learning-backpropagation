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

fn get_dc_dz(
    weights: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
    expected: &DMatrix,
) -> Vec<DMatrix> {
    let layer_count = weights.len() + 1;
    let outputs = activations.last().unwrap();
    let dc_da = del_cost_wrt_layer_activation_for_last_layer(outputs, &expected);
    let dc_dz = dc_da.component_mul(&del_activation_wrt_neuron_values(outputs));
        (1..layer_count - 1)
            .rev()
            .fold(vec![dc_dz], |mut dc_dzs, layer| {
                let outgoing_weights = &weights[layer];
                let neuron_activations = &activations[layer];
                let next_dc_dz = dc_dzs.last().unwrap();
                let dc_da = DMatrix::from_column_slice(
                    outgoing_weights.ncols(),
                    1,
                    &outgoing_weights
                        .column_iter()
                        .map(|weights| weights.dot(&next_dc_dz))
                        .collect::<Vec<FLOAT>>(),
                );
                let da_dz = del_activation_wrt_neuron_values(neuron_activations);
                let dc_dz = dc_da.component_mul(&da_dz);
                dc_dzs.push(dc_dz);
                dc_dzs
            })
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
    dc_dzs: &Vec<DMatrix>,
    activations: &Vec<DMatrix>,
    weights: &Vec<DMatrix>,
) -> Vec<(DMatrix, DMatrix)> {
    let last_activations = activations.iter().rev().skip(1);
    dc_dzs.iter().zip(last_activations).zip(weights).fold(
        Vec::new(),
        |mut gradients, ((dc_dz, last_activation), weights)| {
            let weight_gradient = weights.map_with_location(|row, col, _| {
                dc_dz[row] * last_activation[col]
            });
            let bias_gradient = dc_dz.clone();
            gradients.push((weight_gradient, bias_gradient));
            gradients
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
        let dc_dz = get_dc_dz(&weights, &activations, &expected);
        let gradients = get_gradients(&dc_dz,&activations, &weights);
        for ((layer_weights, layer_biases), (gradient_weight, gradient_bias)) in weights
            .iter_mut()
            .zip(biases.iter_mut())
            .zip(gradients.iter())
        {
            *layer_weights += gradient_weight * LEARNING_RATE;
            *layer_biases += gradient_bias * LEARNING_RATE;
        }
    }
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
}
