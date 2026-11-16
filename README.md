# MNIST digit recogniser in plain C++

A 4-layer multilayer perceptron that classifies handwritten digits, implemented from scratch in
C++ with no libraries — the course exercise for C/C++ at the Hebrew University of Jerusalem.
The point of the exercise is the C++ itself: a value-semantics `Matrix` class with full
operator overloading, the rule of three, and layers that compose through `operator()`.

## Structure

| File | Role |
|---|---|
| `Matrix.h/.cpp` | Heap-allocated float matrix. Copy ctor / assignment / destructor, `+`, `+=`, `*` (matrix and scalar, both sides), `()` and `[]` element access with const and non-const overloads, `transpose`, `vectorize`, `dot`, `norm`, `argmax`, `rref`, stream `<<` (ASCII-art render) and binary `>>` (reads raw weights/images). |
| `Activation.h/.cpp` | `relu` and `softmax` as free functions in a namespace, passed to layers as function pointers. |
| `Dense.h/.cpp` | One layer: `weights · x + bias` followed by its activation, exposed as `operator()`. |
| `MlpNetwork.h/.cpp` | Chains four `Dense` layers (784 → 128 → 64 → 20 → 10) and returns the predicted digit with its softmax probability. |

Input images are 28×28 grey-scale floats; weights and biases are loaded from the binary files
supplied with the course and are not included here.

## Building

```bash
g++ -std=c++14 -Wall -Wextra -o mlp *.cpp main.cpp   # main.cpp from the course handout
./mlp w1 w2 w3 w4 b1 b2 b3 b4
```

## Author

Ali Bder.
