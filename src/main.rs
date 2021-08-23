use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::generation::generate_parameters;
use learning_backpropagation::training::train;
use learning_backpropagation::training_data::generate_training_data;
use std::iter;

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
