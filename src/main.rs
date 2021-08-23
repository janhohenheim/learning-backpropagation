use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::neural_network::get_activations;
use learning_backpropagation::training::train;
use learning_backpropagation::training_data::generate_training_data;

const NETWORK_ARCHITECTURE: NetworkArchitecture = NetworkArchitecture {
    input_size: 2,
    hidden_size: 10,
    output_size: 5,
    hidden_layer_count: 2,
};

const LEARNING_CONFIGURATION: LearningConfiguration = LearningConfiguration {
    learning_rate: 0.3,
    mini_batch_size: 10,
    epochs: 10_000,
};

fn main() {
    let training_data = generate_training_data(&NETWORK_ARCHITECTURE);

    let parameters = train(
        &training_data,
        &NETWORK_ARCHITECTURE,
        &LEARNING_CONFIGURATION,
    );

    for training_data in training_data.iter() {
        let activations = get_activations(&training_data.inputs, &parameters);
        let outputs = activations.last().unwrap();
        println!("{}", training_data);
        println!("output: {}", outputs);
    }
}
