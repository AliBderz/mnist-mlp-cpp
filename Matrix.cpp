#include "Matrix.h"

#include <iostream>
#include <stdexcept>
#include <valarray>

// CONSTRUCTORS/DESTRUCTOR

Matrix::Matrix(int rows, int cols) : _rows(rows), _cols(cols)
{
    if(_rows <= 0 || _cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    data = new float[_rows * _cols];

    for(int i=0; i < _rows * _cols; i++) {
        data[i] = 0;
    }
}

Matrix::Matrix() : Matrix(1, 1)
{}

Matrix::Matrix(const Matrix &other) : _rows(other._rows), _cols(other._cols)
{
    data = new float[_rows * _cols];

    for(int i=0; i < _rows * _cols; i++) {
        data[i] = other.data[i];
    }
}

Matrix::~Matrix()
{
    delete[] data;
    data = nullptr;
}

// GETTERS
int Matrix::get_cols() const
{
    return this->_cols;
}

int Matrix::get_rows() const
{
    return this->_rows;
}

// FUNCTIONS
Matrix &Matrix::transpose()
{
    auto *transposed = new float[_rows * _cols];
    for(int i=0; i < _rows; i++) {
        for(int j=0; j < _cols; j++) {
            transposed[i + j * _rows] = data[i * _cols + j];
        }
    }

    std::swap(_rows, _cols);

    delete[] data;
    data = transposed;

    return *this;
}

Matrix &Matrix::vectorize()
{
    _rows = _rows * _cols;
    _cols = 1;
    return *this;
}

void Matrix::plain_print() const
{
    for(int i=0; i < _rows; i++) {
        for(int j=0; j < _cols; j++) {
            std::cout << data[i * _cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::dot(const Matrix &other) const
{
    if(this->_rows != other._rows || this->_cols != other._cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix dot(this->_rows, this->_cols);
    for(int i=0; i < this->_rows*this->_cols; i++) {
        dot.data[i] = this->data[i] * other.data[i];
    }

    return dot;
}

float Matrix::norm() const
{
    float sigma = 0;
    for(int i=0; i < _rows*_cols; i++) {
        sigma += data[i] * data[i];
    }
    return std::sqrt(sigma);
}

Matrix Matrix::rref() const
{
    Matrix rref0(*this);

    int leading = 0;
    for(int i=0; i < rref0._rows; i++) {
        if(leading >= rref0._cols) {
            break;
        }

        int j = i;
        while(std::fabs(rref0.data[j * rref0._cols + leading]) < 1e-6) {
            j++;
            if(j == rref0._rows) {
                j = i;
                leading++;
                if(leading == rref0._cols) {
                    return rref0;
                }
            }
        }

        if(j != i) {
            for(int l=0; l < rref0._cols; l++) {
                std::swap(rref0.data[i*rref0._cols + l],
                          rref0.data[j*rref0._cols + l]);
            }
        }

        float helper = rref0.data[i*rref0._cols + leading];
        if(std::fabs(helper) > 1e-6) {
            for(int l=0; l < rref0._cols; l++) {
                rref0.data[i*rref0._cols + l] /= helper;
            }
        }

        for(int l=0; l < rref0._rows; l++) {
            if(l != i) {
                float factor = rref0.data[l * rref0._cols + leading];
                if (std::fabs(factor) > 1e-6f) {
                    for (int a = 0; a < rref0._cols; a++) {
                        rref0.data[l * rref0._cols + a] -=
                                factor * rref0.data[i * rref0._cols + a];
                    }
                }
            }
        }
        leading++;
    }



    return rref0;
}

int Matrix::argmax() const
{
    float max_v = data[0];
    int max_i = 0;
    for(int i=1; i < _rows*_cols; i++) {
        if(data[i] > max_v) {
            max_v = data[i];
            max_i = i;
        }
    }
    return max_i;
}

float Matrix::sum() const
{
    float sum = data[0];
    for(int i=1; i < _rows*_cols; i++) {
        sum += data[i];
    }

    return sum;
}

Matrix& Matrix::operator+=(const Matrix &other)
{
    if(this->_rows != other._rows || this->_cols != other._cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    for(int i=0; i < this->_rows*this->_cols; i++) {
        this->data[i] += other.data[i];
    }

    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if(this->_rows != other._rows || this->_cols != other._cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix m(this->_rows, this->_cols);
    for(int i=0; i < this->_rows*this->_cols; i++) {
        m.data[i] = this->data[i] + other.data[i];
    }

    return m;
}

void swap(Matrix& m0, Matrix& m1){
    std::swap(m0._rows, m1._rows);
    std::swap(m0._cols, m1._cols);
    std::swap(m0.data, m1.data);
}

Matrix& Matrix::operator=(const Matrix &other)
{
    if(this != &other) {
        Matrix a(other);
        swap(*this, a);
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if(this->_cols != other._rows) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix m(this->_rows, other._cols);

    for(int i=0; i < this->_rows; i++) {
        for(int j=0; j < other._cols; j++) {
            float sum = 0;
            for(int l=0; l < this->_cols; l++) {
                sum += this->data[i*this->_cols + l] *
                        other.data[l*other._cols + j];
            }
            m.data[i*other._cols + j] = sum;
        }
    }
    return m;
}

Matrix Matrix::operator*(float b) const
{
    Matrix m(_rows, _cols);
    for(int i=0; i < _rows*_cols; i++) {
        m.data[i] = this->data[i] * b;
    }

    return m;
}

Matrix operator*(float b, const Matrix &other) {
    return other*b;
}

float Matrix::operator()(int row, int col) const
{
    if(row >= _rows || col >= _cols  || row < 0 || col < 0) {
        throw std::out_of_range("(row,col) out of range");
    }

    return data[row * _cols + col];
}

float& Matrix::operator()(int row, int col)
{
    if(row >= _rows || col >= _cols  || row < 0 || col < 0) {
        throw std::out_of_range("(row,col) out of range");
    }

    return data[row * _cols + col];
}

float Matrix::operator[](int index) const
{
    if(index >= _rows*_cols || index < 0) {
        throw std::out_of_range("(index) out of range");
    }

    return data[index];
}

float& Matrix::operator[](int index)
{
    if(index >= _rows*_cols || index < 0) {
        throw std::out_of_range("(index) out of range");
    }

    return data[index];
}


std::ostream& operator<<(std::ostream& ostream0, const Matrix& m) {
    for (int i = 0; i < m._rows; ++i) {
        for (int j = 0; j < m._cols; ++j) {
            if (m.data[i * m._cols + j] > 0.1) {
                ostream0 << "**";
            } else {
                ostream0 << "  ";
            }
        }
        ostream0 << std::endl;
    }
    return ostream0;
}

std::istream& operator>>(std::istream& istream0, Matrix& m)
{
    std::size_t a = static_cast<std::size_t>(m._rows) * m._cols;

    if (!istream0.read(reinterpret_cast<char*>(m.data), a * sizeof(float)))
    {
        throw std::runtime_error("operator>>: failed!");
    }
    return istream0;
}