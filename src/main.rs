use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::generation::generate_parameters;
use learning_backpropagation::training::train;
use learning_backpropagation::training_data::generate_training_data;
use std::iter;

const EPOCHS: usize = 10_000;

const NETWORK_ARCHITECTURE: NetworkArchitecture = NetworkArchitecture {
    input_size: 2,
    hidden_size: 10,
    output_size: 5,
    hidden_layer_count: 2,
};

const LEARNING_CONFIGURATION: LearningConfiguration = LearningConfiguration { learning_rate: 0.3 };

fn main() {
    let mut parameters = generate_parameters(&NETWORK_ARCHITECTURE);
    let training_data = generate_training_data(&NETWORK_ARCHITECTURE);
    let outputs =
        iter::repeat_with(|| train(&training_data, &mut parameters, &LEARNING_CONFIGURATION))
            .take(EPOCHS)
            .collect::<Vec<_>>();
    println!("First output: {}", outputs.first().unwrap());
    println!("Last output: {}", outputs.last().unwrap());
    println!("Expected output: {}", training_data.labels);
}
