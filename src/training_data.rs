use crate::configuration::NetworkArchitecture;
use crate::generation::generate_vector;
use crate::linear_algebra::{Float, Vector};
use std::fmt;

/// A pre-labeled training data set.
#[derive(Debug)]
pub struct TrainingData {
    pub inputs: Vector,
    pub labels: Vector,
}

impl fmt::Display for TrainingData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "inputs: {}labels: {}", self.inputs, self.labels)
    }
}

pub fn generate_training_data(network_architecture: &NetworkArchitecture) -> Vec<TrainingData> {
    let inputs = generate_vector(network_architecture.input_size);
    let labels = Vector::from_fn(network_architecture.output_size, |i, _j| {
        i as Float / network_architecture.output_size as Float
    });
    vec![TrainingData { inputs, labels }]
}
