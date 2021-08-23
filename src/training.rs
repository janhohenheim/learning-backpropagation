use crate::backpropagation::{backpropagate, Gradients};
use crate::configuration::LearningConfiguration;
use crate::gradient_descent::gradient_descent;
use crate::neural_network::{get_activations, Parameters};
use crate::training_data::TrainingData;

pub fn train(
    training_data: &[TrainingData],
    mut parameters: &mut Parameters,
    learning_configuration: &LearningConfiguration,
) {
    let empty_gradients = parameters
        .weights
        .iter()
        .zip(parameters.biases.iter())
        .map(|(weights, biases)| Gradients::zeros_like(weights, biases))
        .collect::<Vec<_>>();
    let gradients = training_data
        .iter()
        .map(|training_data| {
            let activations = get_activations(&training_data.inputs, parameters);
            backpropagate(&parameters.weights, &activations, &training_data.labels)
        })
        .fold(empty_gradients, |mut all_gradients, current_gradients| {
            all_gradients
                .iter_mut()
                .zip(current_gradients.into_iter())
                .for_each(|(total_gradient, current_gradient)| *total_gradient += current_gradient);
            all_gradients
        });
    gradient_descent(&mut parameters, &gradients, learning_configuration);
}
