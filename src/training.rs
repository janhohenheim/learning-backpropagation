use crate::backpropagation::{backpropagate, Gradients};
use crate::configuration::{LearningConfiguration, NetworkArchitecture};
use crate::generation::generate_parameters;
use crate::gradient_descent::gradient_descent;
use crate::linear_algebra::Vector;
use crate::neural_network::{get_activations, Parameters};
use crate::trained_neural_network::TrainedNeuralNetwork;
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

fn generate_empty_gradients(parameters: &Parameters) -> Vec<Gradients> {
    parameters
        .weights
        .iter()
        .zip(parameters.biases.iter())
        .map(|(weights, biases)| Gradients::zeros_like(weights, biases))
        .collect::<Vec<_>>()
}

fn add_gradients(mut gradient_sum: Vec<Gradients>, summand: Vec<Gradients>) -> Vec<Gradients> {
    gradient_sum
        .iter_mut()
        .zip(summand.into_iter())
        .for_each(|(gradient_sum, current_gradient)| *gradient_sum += current_gradient);
    gradient_sum
}

fn train_mini_batch(
    training_data: &[TrainingData],
    mut parameters: &mut Parameters,
    learning_configuration: &LearningConfiguration,
) {
    let empty_gradients = generate_empty_gradients(parameters);
    let gradients = training_data
        .iter()
        .map(|training_data| {
            let activations = get_activations(&training_data.inputs, parameters);
            backpropagate(&parameters.weights, &activations, &training_data.labels)
        })
        .fold(empty_gradients, |gradient_sum, current_gradients| {
            add_gradients(gradient_sum, current_gradients)
        });
    gradient_descent(&mut parameters, &gradients, learning_configuration);
}

fn train_epoch(
    training_data: &[TrainingData],
    mut parameters: &mut Parameters,
    learning_configuration: &LearningConfiguration,
) {
    training_data
        .chunks(learning_configuration.mini_batch_size)
        .for_each(|mini_batch| {
            train_mini_batch(mini_batch, &mut parameters, learning_configuration);
        });
}

pub fn train(
    training_data: &[TrainingData],
    network_architecture: &NetworkArchitecture,
    learning_configuration: &LearningConfiguration,
) -> TrainedNeuralNetwork {
    let mut parameters = generate_parameters(network_architecture);
    for _ in 0..learning_configuration.epochs {
        train_epoch(training_data, &mut parameters, learning_configuration);
    }
    TrainedNeuralNetwork::new(parameters)
}
