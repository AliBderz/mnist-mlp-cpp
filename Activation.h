#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

/**
 * @namespace Activation
 * @brief Contains activation functions for neural network layers.
 */
namespace activation {

 /**
  * Applies the ReLU (Rectified Linear Unit) activation function.
  * Sets each negative element to 0, leaves others unchanged.
  * @param m Input matrix
  * @return Matrix after ReLU is applied
  */
 Matrix relu(const Matrix& m);

 /**
  * Applies the Softmax activation function.
  * Assumes the input is a column vector.
  * Converts values into a probability distribution (sums to 1).
  * @param m Input matrix
  * @return Matrix after Softmax is applied
  */
 Matrix softmax(const Matrix& m);
}

// Aliases for activation function pointers (used in Dense constructor)
typedef Matrix (*activation_func)(const Matrix& m);

#endif //ACTIVATION_H
