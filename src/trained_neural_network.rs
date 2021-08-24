use crate::linear_algebra::Vector;
use crate::neural_network::get_activations;
use crate::neural_network::Parameters;

/// An already trained neural network.
pub struct TrainedNeuralNetwork {
    parameters: Parameters,
}

impl TrainedNeuralNetwork {
    /// Creates a new `TrainedNeuralNetwork` from the given parameters.
    pub(crate) fn new(parameters: Parameters) -> Self {
        Self { parameters }
    }
    /// Run the neural network on the given input and return the output of the last layer.
    pub fn run(&self, inputs: &Vector) -> Vector {
        let activations = get_activations(inputs, &self.parameters);
        activations.last().unwrap().clone()
    }
}
