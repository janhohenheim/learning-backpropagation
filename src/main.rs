use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::f32::consts::E;
use std::ops::Range;

type Float = f32;
type Matrix = DMatrix<Float>;
type Vector = DVector<Float>;

const INITIAL_VALUE_OFFSET: Float = 1.0;
const INITIAL_VALUE_RANGE: Range<Float> = 0.0 - INITIAL_VALUE_OFFSET..1.0 + INITIAL_VALUE_OFFSET;

/// Computes the sigmoid function.
fn sigmoid(n: Float) -> Float {
    1.0 / (1.0 + E.powf(-n))
}

/// Requires that input is already in the form of sigmoid(x)
fn d_sigmoid(sigmoid: Float) -> Float {
    sigmoid * (1.0 - sigmoid)
}

/// Generates a random number in the range `range`
fn generate_number(range: Range<Float>) -> Float {
    rand::thread_rng().gen_range(range)
}

/// Generates a random matrix of size `rows` x `cols`
fn generate_matrix(rows: usize, cols: usize) -> Matrix {
    Matrix::from_fn(rows, cols, |_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

/// Generates a random vector of size `size`
fn generate_vector(size: usize) -> Vector {
    Vector::from_fn(size, |_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

/// Runs the neural network for a single layer
fn activate_layer(last_layer: &Vector, weights: &Matrix, biases: &Vector) -> Vector {
    (weights * last_layer + biases).map(sigmoid)
}

/// Gets the derivative of the cost function with respect to the neuron activations.
/// Cost function is (expected - actual)^2
fn dc_da_for_last_layer(actual: &Vector, expected: &Vector) -> Vector {
    2.0 * (expected - actual)
}

/// Gets the derivative of the cost function with respect to the neuron values.
fn da_dz(neuron_values: &Vector) -> Vector {
    neuron_values.map(d_sigmoid)
}

/// Gets the derivative of the cost function with respect to the neuron values from back to front
fn get_dc_dz(weights: &[Matrix], activations: &[Vector], expected: &Vector) -> Vec<Vector> {
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
            let dc_da = Vector::from(
                outgoing_weights
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

/// Runs the neuron network forward and returns the activations of the last layer
fn get_activations(
    layer_count: usize,
    inputs: &Vector,
    weights: &[Matrix],
    biases: &[Vector],
) -> Vec<Vector> {
    (0..layer_count - 1).fold(vec![inputs.clone()], |mut activations, layer| {
        let activation = activate_layer(&activations[layer], &weights[layer], &biases[layer]);
        activations.push(activation);
        activations
    })
}

/// The gradients for a single layer
struct Gradients {
    /// The gradients for the weights
    weights: Matrix,
    /// The gradients for the biases
    biases: Vector,
}

/// Calculates the gradients for a all layers.
/// dc_dzs is the vector of derivatives of the cost function with respect to the neuron values from back to front.
fn get_gradients_from_dc_dz(dc_dzs: Vec<Vector>, activations: &[Vector]) -> Vec<Gradients> {
    let last_activations = activations.iter().rev().skip(1);
    dc_dzs
        .into_iter()
        .zip(last_activations)
        .map(|(dc_dz, last_activation)| {
            // [Outer product](https://en.wikipedia.org/wiki/Outer_product). Same shape as incoming weights.
            // Think of last_activation as the *from* and dc_dz as the *to* of the weight.
            let weight_gradient = &dc_dz * last_activation.transpose();
            let bias_gradient = dc_dz;
            Gradients {
                weights: weight_gradient,
                biases: bias_gradient,
            }
        })
        .rev()
        .collect()
}

/// Runs backpropagation on the neural network and returns the gradients for each layer
fn get_gradients(weights: &[Matrix], activations: &[Vector], expected: &Vector) -> Vec<Gradients> {
    let dc_dzs = get_dc_dz(weights, activations, expected);
    get_gradients_from_dc_dz(dc_dzs, activations)
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
        acc.push(generate_vector(HIDDEN_SIZE));
        acc
    });
    biases.push(generate_vector(OUTPUT_SIZE));
    let inputs = generate_vector(INPUT_SIZE);
    let expected = Vector::from_fn(OUTPUT_SIZE, |i, _j| i as Float / OUTPUT_SIZE as Float);
    let mut outputs = Vec::new();
    for _epoch in 0..1000 {
        let activations = get_activations(LAYER_COUNT, &inputs, &weights, &biases);
        outputs.push(activations.last().unwrap().clone());
        let gradients = get_gradients(&weights, &activations, &expected);
        for ((layer_weights, layer_biases), gradients) in
            weights.iter_mut().zip(biases.iter_mut()).zip(gradients)
        {
            *layer_weights += &gradients.weights * LEARNING_RATE;
            *layer_biases += &gradients.biases * LEARNING_RATE;
        }
    }
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
    println!("Expected output: {}", expected);
}
