#include "Dense.h"

// Constructor
Dense::Dense(const Matrix& weights, const Matrix& bias,
             Matrix (*activation)(const Matrix&))
    : _weights(weights), _bias(bias), _activation(activation) {}

// Getters
const Matrix& Dense::get_weights() const {
    return _weights;
}

const Matrix& Dense::get_bias() const {
    return _bias;
}

Matrix (*Dense::get_activation() const)(const Matrix&) {
    return _activation;
}

// Apply layer: activation(weights * input + bias)
Matrix Dense::operator()(const Matrix& input) const {
    return _activation((_weights * input) + _bias);
}