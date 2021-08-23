use crate::functions::d_sigmoid;
use crate::linear_algebra::{Float, Matrix, Vector};

/// The gradients for a single layer
pub struct Gradients {
    /// The gradients for the weights
    pub weights: Matrix,
    /// The gradients for the biases
    pub biases: Vector,
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
            let da_dz = self::da_dz(neuron_activations);
            let dc_dz = dc_da.component_mul(&da_dz);
            dc_dzs.push(dc_dz);
            dc_dzs
        })
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
pub fn backpropagate(
    weights: &[Matrix],
    activations: &[Vector],
    labels: &Vector,
) -> Vec<Gradients> {
    let dc_dzs = get_dc_dz(weights, activations, labels);
    get_gradients_from_dc_dz(dc_dzs, activations)
}
