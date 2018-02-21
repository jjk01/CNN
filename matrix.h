#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include "tensor.h"

/*
Matrix class designed for our needs (neural network), without more advanced linear
algebra functions (spectral decomposition etc).
n-dimensional vectors are simply nx1 matrices.
*/


class matrix {
public:

  	matrix();
	matrix(int rows, int cols);

    static matrix zeros(int n1, int n2);
    static matrix ones(int n1, int n2);
    static matrix random(int n1, int n2);
    static matrix identity(int n);

    double & operator()(int ind);
  	double operator()(int ind) const;
    double & operator()(int row, int col);
  	double operator()(int row, int col) const;

  	int rows() const;
  	int cols() const;
    int size() const;

  	void add_row(int row_ind, const matrix & row);
  	void add_col(int col_ind, const matrix & col);
  	void h_concatenate(const matrix & m);
  	void v_concatenate(const matrix & m);

  	void remove_row(int row_ind);
  	void remove_col(int col_ind);

    std::vector<double> return_data() const;

  	matrix& operator+=(const matrix& m);
  	matrix& operator*=(const double& f);
    matrix& operator/=(const double& f);

    matrix transpose() const;

    // function to set a block in a matrix
    void block(int row_ind, int col_ind, const matrix& m);

    // get a particular block
    matrix block(int r0, int c0, int r_length, int c_length);

    matrix row(int row_num);
    matrix col(int col_num);

    // c = a.hadamard(b) => c_{ij} = a_{ij} b_{ij}
    matrix hadamard(const matrix &) const;

    // a.dot(b) = \sum_{ij} a_{ij} b_{ij}
    double dot(const matrix &) const;

    void print() const;
    double norm() const;

protected:

  	int row_num;
  	int col_num;

  	// the matrix is 'flattened out' rowwise. i.e:
  	// data = [a11,a12,...,a1n,a21,...,a2n,......,an1,...,ann]

  	std::vector<double> data;

};


class vector: public matrix {
public:

    vector(): matrix(){};
    vector(int rows);
    vector(const matrix &m);
    vector(const tensor & t);
};


// standard matrix binary operations.

matrix operator+(const matrix&, const matrix&);
matrix operator-(const matrix&, const matrix&);
matrix operator*(const matrix&, const matrix&);
matrix operator*(double, const matrix&);
matrix operator*(const matrix&, double);
matrix operator/(const matrix&, double);

matrix outer_product(vector,vector);


#endif /* MATRIX_H */
