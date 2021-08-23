use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::mnist::load_training_data;
use learning_backpropagation::training::train;
use std::error::Error;

const NETWORK_ARCHITECTURE: NetworkArchitecture = NetworkArchitecture {
    input_size: 28 * 28,
    hidden_size: 10,
    output_size: 1,
    hidden_layer_count: 2,
};

const LEARNING_CONFIGURATION: LearningConfiguration = LearningConfiguration {
    learning_rate: 0.3,
    mini_batch_size: 10,
    epochs: 10_000,
};

fn main() -> Result<(), Box<dyn Error>> {
    let training_data = load_training_data("./mnist_handwritten_train.json")?;
    let neural_network = train(
        &training_data,
        &NETWORK_ARCHITECTURE,
        &LEARNING_CONFIGURATION,
    );

    for training_data in training_data.iter() {
        let outputs = neural_network.run(&training_data.inputs);
        println!("{}", training_data);
        println!("output: {}", outputs);
    }
    Ok(())
}
