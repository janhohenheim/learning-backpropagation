use crate::configuration::NetworkArchitecture;
use crate::linear_algebra::{Float, Matrix, Vector};
use crate::neural_network::Parameters;
use rand::Rng;
use std::iter;
use std::ops::Range;

const INITIAL_VALUE_OFFSET: Float = 1.0;
const INITIAL_VALUE_RANGE: Range<Float> = 0.0 - INITIAL_VALUE_OFFSET..1.0 + INITIAL_VALUE_OFFSET;

/// Generates a random number in the range `range`
fn generate_number(range: Range<Float>) -> Float {
    rand::thread_rng().gen_range(range)
}

/// Generates a random matrix of size `rows` x `cols`
fn generate_matrix(rows: usize, cols: usize) -> Matrix {
    Matrix::from_fn(rows, cols, |_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

/// Generates a random vector of size `size`
pub fn generate_vector(size: usize) -> Vector {
    Vector::from_fn(size, |_i, _j| generate_number(INITIAL_VALUE_RANGE))
}

/// Generate random weights
fn generate_weights(network_architecture: &NetworkArchitecture) -> Vec<Matrix> {
    let input_to_hidden_weights = generate_matrix(
        network_architecture.hidden_size,
        network_architecture.input_size,
    );
    let hidden_to_hidden_weights = iter::repeat_with(|| {
        generate_matrix(
            network_architecture.hidden_size,
            network_architecture.hidden_size,
        )
    });
    let hidden_to_output_weights = generate_matrix(
        network_architecture.output_size,
        network_architecture.hidden_size,
    );
    iter::once(input_to_hidden_weights)
        .chain(hidden_to_hidden_weights)
        .take(network_architecture.hidden_layer_count)
        .chain(iter::once(hidden_to_output_weights))
        .collect()
}

/// Generate random biases
fn generate_biases(network_architecture: &NetworkArchitecture) -> Vec<Vector> {
    let hidden_biases = iter::repeat_with(|| generate_vector(network_architecture.hidden_size));
    let output_biases = generate_vector(network_architecture.output_size);
    hidden_biases
        .take(network_architecture.hidden_layer_count)
        .chain(iter::once(output_biases))
        .collect()
}

/// Generate random parameters
pub fn generate_parameters(network_architecture: &NetworkArchitecture) -> Parameters {
    Parameters {
        weights: generate_weights(network_architecture),
        biases: generate_biases(network_architecture),
    }
}
