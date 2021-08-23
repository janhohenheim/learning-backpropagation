use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::generation::{generate_parameters, generate_vector, Parameters};
use learning_backpropagation::linear_algebra::{Float, Matrix, Vector};
use std::f32::consts::E;
use std::iter;

/// Computes the sigmoid function.
fn sigmoid(n: Float) -> Float {
    1.0 / (1.0 + E.powf(-n))
}

/// Requires that input is already in the form of sigmoid(x)
fn d_sigmoid(sigmoid: Float) -> Float {
    sigmoid * (1.0 - sigmoid)
}

/// Runs the neural network for a single layer
fn activate_layer(last_layer: &Vector, weights: &Matrix, biases: &Vector) -> Vector {
    (weights * last_layer + biases).map(sigmoid)
}

/// Gets the derivative of the cost function with respect to the neuron activations.
/// Cost function is (expected - actual)^2
fn dc_da_for_last_layer(actual: &Vector, labels: &Vector) -> Vector {
    2.0 * (labels - actual)
}

/// Gets the derivative of the cost function with respect to the neuron values.
fn da_dz(neuron_values: &Vector) -> Vector {
    neuron_values.map(d_sigmoid)
}

/// Gets the derivative of the cost function with respect to the neuron values from back to front
fn get_dc_dz(weights: &[Matrix], activations: &[Vector], labels: &Vector) -> Vec<Vector> {
    let layer_count = weights.len() + 1;
    let outputs = activations.last().unwrap();
    let dc_da = dc_da_for_last_layer(outputs, labels);
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
fn get_activations(inputs: &Vector, parameters: &Parameters) -> Vec<Vector> {
    let non_input_layers = parameters.weights.len();
    (0..non_input_layers).fold(vec![inputs.clone()], |mut activations, layer| {
        let activation = activate_layer(
            &activations[layer],
            &parameters.weights[layer],
            &parameters.biases[layer],
        );
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
fn backpropagate(weights: &[Matrix], activations: &[Vector], labels: &Vector) -> Vec<Gradients> {
    let dc_dzs = get_dc_dz(weights, activations, labels);
    get_gradients_from_dc_dz(dc_dzs, activations)
}

fn gradient_descent(
    parameters: &mut Parameters,
    gradients: &[Gradients],
    learning_configuration: &LearningConfiguration,
) {
    for ((layer_weights, layer_biases), gradients) in parameters
        .weights
        .iter_mut()
        .zip(parameters.biases.iter_mut())
        .zip(gradients)
    {
        *layer_weights += &gradients.weights * learning_configuration.learning_rate;
        *layer_biases += &gradients.biases * learning_configuration.learning_rate;
    }
}

struct TrainingData {
    inputs: Vector,
    labels: Vector,
}

fn generate_training_data(network_architecture: &NetworkArchitecture) -> TrainingData {
    let inputs = generate_vector(network_architecture.input_size);
    let labels = Vector::from_fn(network_architecture.output_size, |i, _j| {
        i as Float / network_architecture.output_size as Float
    });
    TrainingData { inputs, labels }
}

fn train(
    training_data: &TrainingData,
    mut parameters: &mut Parameters,
    learning_configuration: &LearningConfiguration,
) -> Vector {
    let activations = get_activations(&training_data.inputs, parameters);
    let gradients = backpropagate(&parameters.weights, &activations, &training_data.labels);
    gradient_descent(&mut parameters, &gradients, learning_configuration);
    activations.last().unwrap().clone()
}

fn main() {
    let network_architecture = NetworkArchitecture {
        input_size: 2,
        hidden_size: 10,
        output_size: 5,
        hidden_layer_count: 2,
    };
    let learning_configuration = LearningConfiguration { learning_rate: 0.3 };
    let mut parameters = generate_parameters(&network_architecture);
    let training_data = generate_training_data(&network_architecture);
    let outputs =
        iter::repeat_with(|| train(&training_data, &mut parameters, &learning_configuration))
            .take(10_000)
            .collect::<Vec<_>>();
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
    println!("Expected output: {}", training_data.labels);
}
