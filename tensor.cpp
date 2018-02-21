#include "tensor.h"
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <ctime>

tensor::tensor(): Lx(0), Ly(0), Lz(0){}


tensor::tensor(int n1, int n2, int n3)
: Lx(n1), Ly(n2), Lz(n3), data(std::vector<double>(n1*n2*n3)){}



int tensor::size() const {
    return (Lx*Ly*Lz);
}



int tensor::size(enum index ind) const {
    switch (ind) {
        case index::x : return Lx;
        case index::y : return Ly;
        case index::z : return Lz;
    }
}

tensor tensor::zeros(int n1, int n2, int n3){
    tensor t(n1,n2,n3);

    for (int k = 0; k < t.size(); k++){
        t.data[k] = 0.0;
    }

    return t;
}

tensor tensor::ones(int n1, int n2, int n3){
    tensor t(n1,n2,n3);

    for (int k = 0; k < t.size(); k++){
        t.data[k] = 1.0;
    }

    return t;
}


tensor tensor::random(int n1, int n2, int n3){
    tensor t(n1,n2,n3);

    for (int k = 0; k < t.size(); k++){
        t.data[k] = (std::rand() % 50)/25.0 - 1.0;
    }

    return t;
}


double& tensor::operator()(int n){
    return data.at(n);
}


double tensor::operator()(int n) const {
    return data[n];
}



double& tensor::operator()(int n1, int n2, int n3){
    return data.at(n3*(Lx*Ly) + n2*Lx + n1);
}


double tensor::operator()(int n1, int n2, int n3) const {
    return data[n3*(Lx*Ly) + n2*Lx + n1];
}


tensor& tensor::operator+=(const tensor& t){

    if (Lx != t.size(index::x) || Ly != t.size(index::y) || Lz != t.size(index::z)){
        throw Exception("tensor error: dimension mismatch when using += ");
    }

    for (int k=0; k < data.size(); k++){
        data[k] += t.data[k];
    }

    return *this;
}


tensor& tensor::operator*=(double f){

    for (int k=0; k < data.size(); k++){
        data[k] *= f;
    }

    return *this;
}


tensor& tensor::operator/=(double f){

    for (int k=0; k < data.size(); k++){
        data[k] /= f;
    }

    return *this;
}


tensor& tensor::operator+=(double f){

    for (int k=0; k < data.size(); k++){
        data[k] *= f;
    }

    return *this;
}


tensor tensor::block(int x0, int y0, int z0,
    int x_size, int y_size, int z_size) const {

    if (Lx < x0 + x_size || Ly < y0 + y_size || Lz < z0 + z_size){
        throw Exception("tensor error: dimension mismatch when using tensor block(int,int,int,int,int,int)");
    }

    tensor t(x_size,y_size,z_size);
    int area = Lx*Ly;

    for (int n1 = 0, m1 = x0; n1 < x_size; n1++, m1++){
        for (int n2 = 0, m2 = y0; n2 < y_size; n2++, m2++){
            for (int n3 = 0, m3 = z0; n3 < z_size; n3++, m3++){

                t(n1,n2,n3)= data[m3*area + m2*Lx + m1];

            }
        }
    }
    return t;
}


void tensor::block(int x0, int y0, int z0, const tensor& t){

    if (Lx < x0 + t.size(index::x) || Ly < y0 + t.size(index::y) || Lz < z0 + t.size(index::z)){
        throw Exception("tensor error: input tensor is too large for void block(int,int,int,tensor)");
    }
    int area = Lx*Ly;

    for (int n1 = 0, m1 = x0; n1 < t.size(index::x); n1++, m1++){
        for (int n2 = 0, m2 = y0; n2 < t.size(index::y); n2++, m2++){
            for (int n3 = 0, m3 = z0; n3 < t.size(index::z); n3++, m3++){

                data[m3*area + m2*Lx + m1] = t(n1,n2,n3);
            }
        }
    }
}


double tensor::dot(const tensor &t) const {

    if (Lx != t.size(index::x) || Ly != t.size(index::y) || Lz != t.size(index::z)){
        throw Exception("tensor error: dimension mismatch for double dot(tensor)");
    }

    double sum = 0;

    for (int k = 0; k < data.size(); k++){
        sum += (t.data[k]*data[k]);
    }

    return sum;
}


double tensor::dot(int x0, int y0, int z0, const tensor &t) const {

    int x_size =  t.size(index::x);
    int y_size =  t.size(index::y);
    int z_size =  t.size(index::z);

    if (Lx < x0 + x_size || Ly < y0 + y_size || Lz < z0 + z_size){
        throw Exception("tensor error: dimension mismatch for double dot(int,int,int,tensor)");
    }

    double sum = 0;
    int area = Lx*Ly;

    for (int n1 = 0, m1 = x0; n1 < x_size; n1++, m1++){
        for (int n2 = 0, m2 = y0; n2 < y_size; n2++, m2++){
            for (int n3 = 0, m3 = z0; n3 < z_size; n3++, m3++){

                sum += data[m3*area + m2*Lx + m1]*t(n1,n2,n3);

            }
        }
    }
    return sum;
}


tensor tensor::flip_axis(bool x, bool y, bool z) const {

    int a1 = x ? -1:1;
    int a2 = y ? -1:1;
    int a3 = z ? -1:1;

    int m1 = x ? (Lx-1):0;
    int m2 = y ? (Ly-1):0;
    int m3 = z ? (Lz-1):0;

    tensor t(Lx,Ly,Lz);

    int area = Lx*Ly;

    for (int n1 = 0; n1 < Lx; n1++){
        int i1 = m1 + a1*n1;
        for (int n2 = 0; n2 < size(index::y); n2++){
            int i2 = m2 + a2*n2;
            for (int n3 = 0; n3 < size(index::z); n3++){
                int i3 = m3 + a3*n3;
                t(n1,n2,n3) = data[i3*area + i2*Lx + i1];
            }
        }
    }

    return t;
}


/*
The correlation function takes stride and padding (one for each dimension) together with a tensor as input. With this is
calculates the correlation of the current tensor with the input (which must have dimension smaller than the
current + padding).
*/

tensor tensor::correlation(int Sx, int Sy, int Sz, int Px, int Py, int Pz, const tensor &t) const {

    tensor T = tensor::zeros(Lx + 2*Px, Ly + 2*Py, Lz + 2*Pz);
    T.block(Px,Py,Pz,*this);
    
    if (T.Lx < t.Lx || T.Ly < t.Ly || T.Lz < t.Lz){
        throw Exception("tensor error: input matrix too large for correlation");
    }

    int L1 = 1 + (T.Lx - t.size(index::x))/Sx;
    int L2 = 1 + (T.Ly - t.size(index::y))/Sy;
    int L3 = 1 + (T.Lz - t.size(index::z))/Sz;

    tensor s(L1, L2, L3);

    for (int n1 = 0; n1 < L1; n1++){
        for (int n2 = 0; n2 < L2; n2++){
            for (int n3 = 0; n3 < L3; n3++){
                s(n1,n2,n3) = T.dot(Sx*n1,Sy*n2,Sz*n3,t);
            }
        }
    }
    return s;
}



//dC/d(W_{r,x,y,z}^{l}) = sum_{i,j} epsilon_{ijr}^{l} a_{i*S_x + x, j*S_y + y, z}^{l-1}
// This is to calculate the gradient of the conv weights given the layer error.


std::vector<tensor> tensor::correlation(int Sx, int Sy, int Px, int Py, const tensor &t) const {

    std::vector<tensor> R(t.size(index::z));
    
    // this tensor with the padding added.
    
    tensor T = tensor::zeros(Lx + 2*Px, Ly + 2*Py, Lz);

    if (T.Lx < t.Lx || T.Ly < t.Ly ){
        throw Exception("tensor error: input matrix too large for correlation");
    }

    T.block(Px,Py,0,*this);
    
    // dimensions of the final tensor
    
    int L1 = 1 + (T.Lx - t.size(index::x))/Sx;
    int L2 = 1 + (T.Ly - t.size(index::y))/Sy;
    int L3 = T.Lz;

    tensor s(L1, L2, L3);

    for (int m1 = 0; m1 < t.Lz; m1++){
        tensor S = tensor::zeros(L1, L2, L3);
        for (int m2 = 0; m2 < T.Lz; m2++){

            tensor g1 = t.block(0,0,m1, t.Lx, t.Ly,1);
            tensor g2 = T.block(0,0,m2, T.Lx, T.Ly,1);

            for (int n1 = 0; n1 < L1; n1++){
                for (int n2 = 0; n2 < L2; n2++){

                    S(n1,n2,m2) = g2.dot(Sx*n1,Sy*n2,0,t);

                }
            }
        }
        R[m1] = S;
    }
    return R;
}

/*
A fairly general form of convolution (an analog of correlation).
*/

tensor tensor::convolve(int Sx, int Sy, int Sz, int Px, int Py, int Pz, const tensor &t) const {

    int Wx = t.size(index::x);
    int Wy = t.size(index::y);
    int Wz = t.size(index::z);

    int L1 = Sx*(Lx - 1) + Wx - 2*Px;
    int L2 = Sy*(Ly - 1) + Wy - 2*Py;
    int L3 = Sz*(Lz - 1) + Wz - 2*Pz;

    tensor s(L1, L2, L3);

    int area = L1*L2;


    for (int n = 0; n < L1*L2*L3; n++){

        int n3 = n/area;
        int n2 = (n - area*n3)/L1;
        int n1 = n - area*n3 - L1*n2;


        for (int x = std::max(0, (n1 + Px - Wx + 1)/Sx); x < std::min(1 + (n1+Px)/Sx,L1); x++){
            for (int y = std::max(0, (n2 + Py - Wy + 1)/Sy); y < std::min(1 + (n2+Py)/Sy,L2); y++){
                for (int z = std::max(0, (n3 + Pz - Wz + 1)/Sz); z < std::min(1 + (n3+Pz)/Sz,L3); z++){
                    s(n1,n2,n3) += ((*this)(x,y,z))*t(n1 + Px - Sx*x, n2 + Py - Sy*y, n3 + Pz - Sz*z);
                }
            }
        }

    }
    return s;
}

/*
 This form of correlation function is similar to that above, but specifically adapted to the needs for backpropagation
 in the convolutional layers in a neural network. In these layers (in feed forward), the current tensor is correlated with
 a vector of weight tensors (each with depth equal to that of the current tensor). Each weight produces a 2D array, which
 form a tensor when concatenated.
 */


tensor tensor::convolve(int S, int P, const std::vector<tensor> &t) const {

    int Wx = t[0].size(index::x);
    int Wy = t[0].size(index::y);
    int Wz = t[0].size(index::z);

    for (auto it = t.begin(); it != t.end(); it++){
        if (it->size(index::x) != Wx || it->size(index::y) != Wy || it->size(index::z) != Wz){
            throw Exception("tensor convolve: tensors must be consistant size.");
        }
    }

    if (t.size() != Lz){
        throw Exception("tensor convolve: inconsistant depth");
    }

    int L1 = S*(Lx - 1) + Wx - 2*P;
    int L2 = S*(Ly - 1) + Wy - 2*P;
    int L3 = Wz;

    tensor s(L1, L2, L3);

    int area = L1*L2;


    for (int n = 0; n < L1*L2*L3; n++){

        int n3 = n/area;
        int n2 = (n - area*n3)/L1;
        int n1 = n - area*n3 - L1*n2;


        for (int x = std::max(0, (n1 + P - Wx + 1)/S); x < std::min(1 + (n1+P)/S,L1); x++){
            for (int y = std::max(0, (n2 + P - Wy + 1)/S); y < std::min(1 + (n2+P)/S,L2); y++){
                for (int z = 0; z < t.size() ; z++){
                    s(n1,n2,n3) += ((*this)(x,y,z))*t[z](n1 + P - S*x, n2 + P - S*y,n3);
                }
            }
        }

    }
    return s;
}



tensor tensor::hadamard(const tensor & t){

    if (Lx != t.size(index::x) || Ly != t.size(index::y) || Lz != t.size(index::z)){
        throw Exception("tensor error: dimension mismatch with hadamard");
    }

    tensor y = tensor::zeros(Lx,Ly,Lz);

    for (int k = 0; k < t.size(); k++){
         y.data[k] = data[k]*t.data[k];
    }

    return y;
}

tensor operator+(const tensor& t1, const tensor& t2){
    tensor t(t1);
    t += t2;
    return t;
}


tensor operator*(double f, const tensor& t){
    tensor _t(t);
    _t *= f;
    return _t;
}



tensor operator*(const tensor& t, double f){
    return f*t;
}



tensor operator/(const tensor& t, double f){
    return (1/f)*t;
}


double tensor::max(){
    return *std::max_element(data.begin(),data.end());
}


double tensor::max(int & x, int & y, int & z){

    auto ind = std::max_element(data.begin(),data.end());
    int dist = std::distance(data.begin(),ind);

    int area = Lx*Ly;

    z = dist/area;
    y = (dist - area*z)/Lx;
    x = dist - area*z - Lx*y;

    return *ind;
}


std::vector<double> tensor::return_data() const {
    return data;
}


void tensor::set_data(const std::vector<double> & v) {

    if (this->size() != v.size()){
        throw Exception("data size not equal to tensor size");
    }

    data = v;
}



void tensor::print(int precision) const {

    int area = Lx*Ly;
    double N = pow(10,precision);

    for (int k3 = 0; k3 < Lz; k3++){
        for (int k1 = 0; k1 < Lx; k1++){
            for (int k2 = 0; k2 < Ly; k2++){
                std::cout << round(N*data[k3*area + k2*Lx + k1])/N << ",";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
