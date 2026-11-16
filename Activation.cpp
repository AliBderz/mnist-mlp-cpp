#include "Activation.h"
#include <cmath>

namespace activation {

    Matrix relu(const Matrix& m) {
        Matrix mat(m);
        int size = m.get_rows() * m.get_cols();
        for (int i = 0; i < size; ++i) {
            mat[i] = (mat[i] > 0) ? mat[i] : 0.0f;
        }
        return mat;
    }

    Matrix softmax(const Matrix& m) {
        Matrix mat(m);
        int len = m.get_rows() * m.get_cols();

        float max_val = mat[0];
        for (int i = 1; i < len; ++i) {
            if (mat[i] > max_val) {
                max_val = mat[i];
            }
        }

        float sum0 = 0;
        for (int i = 0; i < len; ++i) {
            mat[i] = std::exp(mat[i] - max_val);
            sum0 += mat[i];
        }

        for (int i = 0; i < len; ++i) {
            mat[i] /= sum0;
        }

        return mat;
    }

}
