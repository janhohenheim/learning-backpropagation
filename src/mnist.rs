use crate::linear_algebra::{Float, Vector};
use crate::training::TrainingData;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Debug)]
struct Mnist {
    pub image: Vec<Float>,
    pub label: Float,
}

fn load_mnist(path: &str) -> Result<Vec<Mnist>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mnist = serde_json::from_reader(reader)?;
    Ok(mnist)
}

pub fn load_training_data(path: &str) -> Result<Vec<TrainingData>, Box<dyn Error>> {
    let mnist = load_mnist(path)?;
    let training_data = mnist
        .into_iter()
        .map(|mnist| TrainingData {
            inputs: Vector::from_vec(mnist.image) / 255.0,
            labels: Vector::from_vec(vec![mnist.label / 10.0]),
        })
        .collect::<Vec<_>>();
    Ok(training_data)
}
