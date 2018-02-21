#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "neural_layer.h"

/*
struct net_sizes {
    std::vector<pair> W0;
    std::vector<int> K;
    std::vector<pair> B0;
    std::vector<pair> W1;
    std::vector<int> B1;
    pair W2;
    int B2;
};*/



class neural_net {
public:

    neural_net() = default;

    void add_input_layer(int W_in, int D);
    void add_convolution_layer(FunctionType, int W_out, int K, int W_filter, int Stride);
    void add_pooling_layer(int pooling_width);
    void add_fully_connected_layer(FunctionType, int N_out);

    vector action(const tensor &);
    void forward_propagate(const tensor &);
    
    const std::vector<convolutional_layer> * convolution_ptr() const;
    const std::vector<fully_connected_layer> * full_ptr() const;
    const input_layer * input_ptr() const;

private:

    input_layer inpt;
    std::vector<convolutional_layer> conv;
    std::vector<fully_connected_layer> full;
};

/*

z_{pqr}^{l} = sum_{x,y,z} W_{r,x,y,z}^{l} a_{x+p,y+q,z}^{l-1} + b_{pqr}^{l} = a.correlation(W) + b

epsilon_{i,j,k}^{l} = sigma'(z_{i,j,k}) sum_{x=0}^{min{i,n_x-1} sum_{y=0}^{min{j,n_y-1} sum_{z} W_{z,x,y,k}^{l+1} epsilon_{i-x,j-y,k}^{l+1}

epsilon_{i,j,k}^{l} = sigma'(z_{i,j,k}) epsilon^{l+1}.convolve(W_{k}^{l+1})

dC/d(b_{ijk}^{l}) = epsilon_{ijk}^{l}

dC/d(W_{r,x,y,z}^{l}) = sum_{i,j} epsilon_{ijr}^{l} a_{i+x,j+y,z}^{l-1}

*/


#endif /* NEURAL_NET_H */
