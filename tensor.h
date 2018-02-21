#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include "exception.h"


enum class index {
    x,
    y,
    z
};

/*
Class for 3D array type objects. It is only able to store doubles.
*/

class tensor {

public:

	tensor();
	tensor(int n1, int n2, int n3);

    static tensor zeros(int n1, int n2, int n3);
    static tensor ones(int n1, int n2, int n3);
    static tensor random(int n1, int n2, int n3);

    int size() const;
    int size(enum index) const;

    double& operator()(int ind);
	double operator()(int ind) const;
    double& operator()(int n1, int n2, int n3 );
	double operator()(int n1, int n2, int n3) const;

    tensor& operator+=(const tensor& rhs);
    tensor& operator+=(double s);
    tensor& operator*=(double s);
    tensor& operator/=(double s);

    tensor block(int,int,int,int,int,int) const;
    void block(int,int,int, const tensor&);

    tensor flip_axis(bool,bool,bool) const;

	tensor convolve(int Sx, int Sy, int Sz, int Px, int Py, int Pz, const tensor &t) const;
    tensor convolve(int S, int P, const std::vector<tensor> &t) const;

    tensor correlation(int Sx, int Sy, int Sz, int Px, int Py, int Pz, const tensor &t) const;
    std::vector<tensor> correlation(int Sx, int Sy, int Px, int Py, const tensor &t) const;

    double dot(const tensor &t) const;
    double dot(int,int,int, const tensor &t) const;
    tensor hadamard(const tensor &);

    std::vector<double> return_data() const;
    void set_data(const std::vector<double> &);
    double max();
    double max(int &, int &, int&);
    void print(int precision = 3) const;

private:

	int Lx;
	int Ly;
	int Lz;

	std::vector<double> data;
};

tensor operator+(const tensor &, const tensor &);
tensor operator*(double, const tensor &);
tensor operator*(const tensor &, double);
tensor operator/(const tensor&, double);

#endif /* TENSOR_H */
