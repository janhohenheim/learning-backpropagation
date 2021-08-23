use crate::backpropagation::backpropagate;
use crate::configuration::LearningConfiguration;
use crate::gradient_descent::gradient_descent;
use crate::linear_algebra::Vector;
use crate::neural_network::{get_activations, Parameters};
use crate::training_data::TrainingData;

pub fn train(
    training_data: &TrainingData,
    mut parameters: &mut Parameters,
    learning_configuration: &LearningConfiguration,
) -> Vector {
    let activations = get_activations(&training_data.inputs, parameters);
    let gradients = backpropagate(&parameters.weights, &activations, &training_data.labels);
    gradient_descent(&mut parameters, &gradients, learning_configuration);
    activations.last().unwrap().clone()
}
