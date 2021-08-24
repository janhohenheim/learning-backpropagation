use learning_backpropagation::configuration::{LearningConfiguration, NetworkArchitecture};
use learning_backpropagation::mnist::load_training_data;
use learning_backpropagation::training::train;
use std::error::Error;

const NETWORK_ARCHITECTURE: NetworkArchitecture = NetworkArchitecture {
    input_size: 28 * 28,
    hidden_size: 20,
    output_size: 1,
    hidden_layer_count: 2,
};

const LEARNING_CONFIGURATION: LearningConfiguration = LearningConfiguration {
    learning_rate: 0.3,
    mini_batch_size: 10,
    epochs: 30,
};

fn main() -> Result<(), Box<dyn Error>> {
    let training_data = load_training_data("./mnist_handwritten_train.json")?;
    let neural_network = train(
        &training_data,
        &NETWORK_ARCHITECTURE,
        &LEARNING_CONFIGURATION,
    );

    let testing_data = load_training_data("./mnist_handwritten_test.json")?;
    for testing_data in testing_data.iter() {
        let outputs = neural_network.run(&testing_data.inputs);
        println!("label: {}", &testing_data.labels * 10.0);
        println!("output: {}", &outputs * 10.0);
        println!("------");
    }
    Ok(())
}
