use learning_backpropagation::backpropagation::backpropagate;
use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::generation::{generate_parameters, generate_vector};
use learning_backpropagation::gradient_descent::gradient_descent;
use learning_backpropagation::linear_algebra::{Float, Vector};
use learning_backpropagation::neural_network::{get_activations, Parameters};
use std::iter;

struct TrainingData {
    inputs: Vector,
    labels: Vector,
}

fn generate_training_data(network_architecture: &NetworkArchitecture) -> TrainingData {
    let inputs = generate_vector(network_architecture.input_size);
    let labels = Vector::from_fn(network_architecture.output_size, |i, _j| {
        i as Float / network_architecture.output_size as Float
    });
    TrainingData { inputs, labels }
}

fn train(
    training_data: &TrainingData,
    mut parameters: &mut Parameters,
    learning_configuration: &LearningConfiguration,
) -> Vector {
    let activations = get_activations(&training_data.inputs, parameters);
    let gradients = backpropagate(&parameters.weights, &activations, &training_data.labels);
    gradient_descent(&mut parameters, &gradients, learning_configuration);
    activations.last().unwrap().clone()
}

fn main() {
    let network_architecture = NetworkArchitecture {
        input_size: 2,
        hidden_size: 10,
        output_size: 5,
        hidden_layer_count: 2,
    };
    let learning_configuration = LearningConfiguration { learning_rate: 0.3 };
    let mut parameters = generate_parameters(&network_architecture);
    let training_data = generate_training_data(&network_architecture);
    let outputs =
        iter::repeat_with(|| train(&training_data, &mut parameters, &learning_configuration))
            .take(10_000)
            .collect::<Vec<_>>();
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
    println!("Expected output: {}", training_data.labels);
}
