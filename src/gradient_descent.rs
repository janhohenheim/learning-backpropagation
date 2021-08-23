use crate::backpropagation::Gradients;
use crate::configuration::LearningConfiguration;
use crate::neural_network::Parameters;

pub fn gradient_descent(
    parameters: &mut Parameters,
    gradients: &[Gradients],
    learning_configuration: &LearningConfiguration,
) {
    for ((layer_weights, layer_biases), gradients) in parameters
        .weights
        .iter_mut()
        .zip(parameters.biases.iter_mut())
        .zip(gradients)
    {
        *layer_weights += &gradients.weights * learning_configuration.learning_rate;
        *layer_biases += &gradients.biases * learning_configuration.learning_rate;
    }
}
