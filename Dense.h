// Dense.h
#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

// Insert Dense class here...
class Dense {
private:
    Matrix _weights;
    Matrix _bias;
    Matrix (*_activation)(const Matrix&);
public:
    Dense(const Matrix& weights, const Matrix& bias,
          Matrix (*activation)(const Matrix&));

    // Getters
    const Matrix& get_weights() const;
    const Matrix& get_bias() const;
    Matrix (*get_activation() const)(const Matrix&);

    // Applies the layer transformation to input matrix (forward pass)
    Matrix operator()(const Matrix& input) const;

};









#endif //DENSE_H