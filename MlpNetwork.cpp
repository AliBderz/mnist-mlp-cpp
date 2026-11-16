#include "MlpNetwork.h"
#include "Activation.h"  // Needed for relu and softmax

// Constructor: initializes the 4 Dense layers with proper activations
MlpNetwork::MlpNetwork(const Matrix weights[MLP_SIZE],
                       const Matrix biases[MLP_SIZE])
        : _layers{
            Dense(weights[0], biases[0], activation::relu),
            Dense(weights[1], biases[1], activation::relu),
            Dense(weights[2], biases[2], activation::relu),
            Dense(weights[3], biases[3], activation::softmax)
    } {}

// Applies all 4 layers sequentially and returns predicted digit
digit MlpNetwork::operator()(const Matrix &input) const {
    Matrix output = input;
    for (const auto & _layer : _layers) {
        output = _layer(output);
    }

    unsigned int i = output.argmax();
    float probability = output[i];

    digit f = {i, probability};
    return f;
}
