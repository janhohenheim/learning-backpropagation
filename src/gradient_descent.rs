use crate::backpropagation::Gradients;
use crate::configuration::LearningConfiguration;
use crate::neural_network::Parameters;

pub fn gradient_descent(
    parameters: &mut Parameters,
    gradients: &[Gradients],
    last_gradients: &[Gradients],
    learning_configuration: &LearningConfiguration,
) {
    parameters
        .weights
        .iter_mut()
        .zip(parameters.biases.iter_mut())
        .zip(gradients)
        .zip(last_gradients)
        .for_each(
            |(((layer_weights, layer_biases), gradients), last_gradients)| {
                *layer_weights += (&gradients.weights
                    + &last_gradients.weights * learning_configuration.momentum)
                    * learning_configuration.learning_rate;
                *layer_biases += (&gradients.biases
                    + &last_gradients.biases * learning_configuration.momentum)
                    * learning_configuration.learning_rate;
            },
        );
}
