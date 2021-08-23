use crate::functions::sigmoid;
use crate::linear_algebra::{Matrix, Vector};

pub struct Parameters {
    pub weights: Vec<Matrix>,
    pub biases: Vec<Vector>,
}

/// Runs the neural network for a single layer
fn activate_layer(last_layer: &Vector, weights: &Matrix, biases: &Vector) -> Vector {
    (weights * last_layer + biases).map(sigmoid)
}

/// Runs the neuron network forward and returns the activations of the last layer
pub fn get_activations(inputs: &Vector, parameters: &Parameters) -> Vec<Vector> {
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
