#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"

#define MLP_SIZE 4

/**
 * @struct digit
 * @brief Identified (by MLP network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit {
	unsigned int value;
	float probability;
} digit;

// Expected matrix dimensions for input, weights, and biases
const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
									{64,  128},
									{20,  64},
									{10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
								 {64,  1},
								 {20,  1},
								 {10,  1}};

/**
 * @class MlpNetwork
 * @brief Implements a 4-layer neural network (MLP) with predefined structure.
 */
class MlpNetwork {
private:
	Dense _layers[MLP_SIZE];  // Array of 4 Dense layers

public:
	/**
	 * Constructor: builds 4 Dense layers from weights and biases arrays.
	 * @param weights Array of 4 weight matrices
	 * @param biases  Array of 4 bias matrices
	 */
	MlpNetwork(const Matrix weights[MLP_SIZE],
			   const Matrix biases[MLP_SIZE]);

	/**
	 * Applies the network to an input matrix.
	 * @param input The input image as a Matrix (should be 784×1)
	 * @return digit struct with predicted class and probability
	 */
	digit operator()(const Matrix &input) const;
};

#endif // MLPNETWORK_H
