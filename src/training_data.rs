use crate::configuration::NetworkArchitecture;
use crate::generation::generate_vector;
use crate::linear_algebra::{Float, Vector};

/// A pre-labeled training data set.
pub struct TrainingData {
    pub inputs: Vector,
    pub labels: Vector,
}

pub fn generate_training_data(network_architecture: &NetworkArchitecture) -> TrainingData {
    let inputs = generate_vector(network_architecture.input_size);
    let labels = Vector::from_fn(network_architecture.output_size, |i, _j| {
        i as Float / network_architecture.output_size as Float
    });
    TrainingData { inputs, labels }
}
