use crate::linear_algebra::Float;
use std::f32::consts::E;

/// Computes the sigmoid function.
pub fn sigmoid(n: Float) -> Float {
    1.0 / (1.0 + E.powf(-n))
}

/// Requires that input is already in the form of sigmoid(x)
pub fn d_sigmoid(sigmoid: Float) -> Float {
    sigmoid * (1.0 - sigmoid)
}
