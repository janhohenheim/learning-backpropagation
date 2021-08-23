use crate::linear_algebra::Vector;
use crate::neural_network::get_activations;
use crate::neural_network::Parameters;

pub struct TrainedNeuralNetwork {
    parameters: Parameters,
}

impl TrainedNeuralNetwork {
    pub fn new(parameters: Parameters) -> Self {
        Self { parameters }
    }
    pub fn run(&self, inputs: &Vector) -> Vector {
        let activations = get_activations(inputs, &self.parameters);
        activations.last().unwrap().clone()
    }
}
