use nalgebra::DMatrix;
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

type Float = f32;
type Matrix = DMatrix<Float>;

const INITIAL_VALUE_OFFSET: Float = 1.0;
const INITIAL_VALUE_RANGE: Range<Float> = 0.0 - INITIAL_VALUE_OFFSET..1.0 + INITIAL_VALUE_OFFSET;

fn sigmoid(n: Float) -> Float {
    1.0 / (1.0 + E.powf(-n))
}

/// Requires that sigmoid is already sigmoid(x)
fn d_sigmoid(sigmoid: Float) -> Float {
    sigmoid * (1.0 - sigmoid)
}

fn generate_number(range: Range<Float>) -> Float {
    rand::thread_rng().gen_range(range)
}

fn generate_matrix(rows: usize, cols: usize) -> Matrix {
    Matrix::from_fn(rows, cols, |_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

fn get_neuron_values(last_layer: &Matrix, weights: &Matrix, biases: &Matrix) -> Matrix {
    weights * last_layer + biases
}

fn run_activation_function(neuron_values: Matrix) -> Matrix {
    neuron_values.map(sigmoid)
}

/// Cost function is (expected - actual)^2
fn dc_da_for_last_layer(actual: &Matrix, expected: &Matrix) -> Matrix {
    2.0 * (expected - actual)
}

fn da_dz(neuron_values: &Matrix) -> Matrix {
    neuron_values.map(d_sigmoid)
}

fn get_dc_dz(weights: &[Matrix], activations: &[Matrix], expected: &Matrix) -> Vec<Matrix> {
    let layer_count = weights.len() + 1;
    let outputs = activations.last().unwrap();
    let dc_da = dc_da_for_last_layer(outputs, expected);
    let da_dz = da_dz(outputs);
    let dc_dz = dc_da.component_mul(&da_dz);
    (1..layer_count - 1)
        .rev()
        .fold(vec![dc_dz], |mut dc_dzs, layer| {
            let outgoing_weights = &weights[layer];
            let neuron_activations = &activations[layer];
            let next_dc_dz = dc_dzs.last().unwrap();
            let dc_da = Matrix::from_column_slice(
                outgoing_weights.ncols(),
                1,
                &outgoing_weights
                    .column_iter()
                    .map(|weights| weights.dot(next_dc_dz))
                    .collect::<Vec<Float>>(),
            );
            let da_dz = crate::da_dz(neuron_activations);
            let dc_dz = dc_da.component_mul(&da_dz);
            dc_dzs.push(dc_dz);
            dc_dzs
        })
}

fn get_activations(
    layer_count: usize,
    inputs: &Matrix,
    weights: &[Matrix],
    biases: &[Matrix],
) -> Vec<Matrix> {
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

fn get_gradients_from_dc_dz(dc_dzs: &[Matrix], activations: &[Matrix]) -> Vec<(Matrix, Matrix)> {
    let last_activations = activations.iter().rev().skip(1);
    dc_dzs.iter().zip(last_activations).fold(
        Vec::new(),
        |mut gradients, (dc_dz, last_activation)| {
            // [Outer product](https://en.wikipedia.org/wiki/Outer_product). Same shape as incoming weights.
            // Think of last_activation as the *from* and dc_dz as the *to* of the weight.
            let weight_gradient = dc_dz * last_activation.transpose();
            let bias_gradient = dc_dz.clone();
            gradients.push((weight_gradient, bias_gradient));
            gradients
        },
    )
}

fn get_gradients(
    weights: &[Matrix],
    activations: &[Matrix],
    expected: &Matrix,
) -> Vec<(Matrix, Matrix)> {
    let dc_dzs = get_dc_dz(weights, activations, expected);
    get_gradients_from_dc_dz(&dc_dzs, activations)
}

fn main() {
    const INPUT_SIZE: usize = 2;
    const HIDDEN_SIZE: usize = 10;
    const OUTPUT_SIZE: usize = 5;
    const HIDDEN_LAYER_COUNT: usize = 2;
    const LAYER_COUNT: usize = 2 + HIDDEN_LAYER_COUNT;
    const LEARNING_RATE: Float = 0.3;

    let input_to_hidden_weights = generate_matrix(HIDDEN_SIZE, INPUT_SIZE);
    let hidden_to_output_weights = generate_matrix(OUTPUT_SIZE, HIDDEN_SIZE);

    let mut weights =
        (0..HIDDEN_LAYER_COUNT - 1).fold(vec![input_to_hidden_weights], |mut acc, _layer| {
            acc.push(generate_matrix(HIDDEN_SIZE, HIDDEN_SIZE));
            acc
        });
    weights.push(hidden_to_output_weights);

    let mut biases = (0..HIDDEN_LAYER_COUNT).fold(Vec::new(), |mut acc, _layer| {
        acc.push(generate_matrix(HIDDEN_SIZE, 1));
        acc
    });
    biases.push(generate_matrix(OUTPUT_SIZE, 1));
    let inputs = generate_matrix(INPUT_SIZE, 1);
    let expected = Matrix::from_fn(OUTPUT_SIZE, 1, |i, _j| i as Float / OUTPUT_SIZE as Float);
    let mut outputs = Vec::new();
    for _epoch in 0..1000 {
        let activations = get_activations(LAYER_COUNT, &inputs, &weights, &biases);
        outputs.push(activations.last().unwrap().clone());
        let gradients = get_gradients(&weights, &activations, &expected);
        for ((layer_weights, layer_biases), (gradient_weight, gradient_bias)) in weights
            .iter_mut()
            .zip(biases.iter_mut())
            .zip(gradients.iter().rev())
        {
            *layer_weights += gradient_weight * LEARNING_RATE;
            *layer_biases += gradient_bias * LEARNING_RATE;
        }
    }
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
    println!("Expected output: {}", expected);
}
