use crate::linear_algebra::Float;

/// The network's architecture
pub struct NetworkArchitecture {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub hidden_layer_count: usize,
}

/// Settings for the learning step
pub struct LearningConfiguration {
    pub learning_rate: Float,
    pub mini_batch_size: usize,
    pub epochs: usize,
}
