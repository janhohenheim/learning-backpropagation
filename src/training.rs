use crate::backpropagation::{backpropagate, Gradients};
use crate::configuration::LearningConfiguration;
use crate::gradient_descent::gradient_descent;
use crate::neural_network::{get_activations, Parameters};
use crate::training_data::TrainingData;

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

pub fn train(
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
