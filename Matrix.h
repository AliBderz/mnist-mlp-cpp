// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H
#include <iosfwd>


// You don't have to use the struct. Can help you with MlpNetwork.h
struct matrix_dims {
    int rows, cols;
};


// Insert Matrix class here...
class Matrix {
private:
    float *data = nullptr;
    int _rows, _cols;
public:
    Matrix(int rows, int cols);
    Matrix();
    Matrix(const Matrix &other);
    ~Matrix();

    int get_rows() const;
    int get_cols() const;

    Matrix &transpose();
    Matrix &vectorize();

    void plain_print() const;
    Matrix dot(const Matrix &other) const;
    float norm() const;
    Matrix rref() const;
    int argmax() const;
    float sum() const;

    Matrix& operator+=(const Matrix &b);
    Matrix operator+(const Matrix &b) const;
    friend void swap(Matrix& m0, Matrix& m1);
    Matrix& operator=(const Matrix &b);

    Matrix operator*(const Matrix &b) const;
    Matrix operator*(float b) const;
    friend Matrix operator*(float a, const Matrix &b);


    float operator()(int row, int col) const;
    float &operator()(int row, int col);
    float operator[](int index) const;
    float &operator[](int index);

    friend std::ostream &operator<<(std::ostream &ostream, const Matrix &m);
    friend std::istream &operator>>(std::istream &istream0, Matrix &m);

};

#endif //MATRIX_H