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

fn generate_training_set(
    network_architecture: &NetworkArchitecture,
    expected_outputs: &[Float],
) -> TrainingData {
    let inputs = generate_vector(network_architecture.input_size);
    let labels = Vector::from_fn(network_architecture.output_size, |i, _j| {
        expected_outputs[i]
    });
    TrainingData { inputs, labels }
}

pub fn generate_training_data(network_architecture: &NetworkArchitecture) -> Vec<TrainingData> {
    vec![
        generate_training_set(network_architecture, &[0.0, 0.1, 0.2, 0.3, 0.4]),
        generate_training_set(network_architecture, &[0.0, 0.2, 0.4, 0.6, 0.8]),
    ]
}
