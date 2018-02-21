#include <random>
#include <iostream>
#include "neural_layer.h"



input_layer::input_layer(int W_in, int W_out, int D):
input_width(W_in), a(tensor::zeros(W_out,W_out,D)), pooling(false), pooling_with(0){}
    

tensor input_layer::feed_forward(const tensor & t){
    
    if (t.size(index::x) != a.size(index::x) || t.size(index::y) != a.size(index::y) || t.size(index::z) != a.size(index::z)){
        throw Exception("wrong input vector size.");
    }
    
    if (pooling){
        for (int x = 0; x < a.size(index::x); x++){
            for (int y = 0; y < a.size(index::y); y++){
                for (int z = 0; z < a.size(index::z); z++){
                    a(x,y,z) = t.block(x*pooling_with, y*pooling_with, z, pooling_with, pooling_with,1).max();
                }
            }
        }
    } else {
        a = t;
    }

    return a;
}

    
tensor input_layer::get_output() const {
    return a;
}


pair input_layer::output_size() const {
    return pair(a.size(index::x),a.size(index::z));
}



void input_layer::pooling_convert(int width){
    
    if (input_width%width != 0){
        throw Exception("pooling layer: the pooling width must divide the input width.");
    }
    
    pooling_with = width;
    pooling = true;
    a = tensor::zeros(a.size(index::x)/width,a.size(index::y)/width,a.size(index::z));
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



convolutional_layer::convolutional_layer(FunctionType FT, int W_in, int D, int W_out, int K, int W_filter, int S):
act_fn(ActivationFunction<tensor>(FT)), stride(S), depth(D), a(tensor::zeros(W_out,W_out,K)), b(tensor::zeros(W_out,W_out,K)),
pooled(false), pooling_width(0), ind(tensor()), output_width(W_out)
{

    if (S*(W_out-1) + W_filter < W_in){
        width = W_in - S*(W_out-1);
        padding = 0;
    } else {
        padding = (int) std::floor((S*(W_out-1) + W_filter - W_in)/2.0);
        width = 2*padding + W_in - S*(W_out-1);
    }

    for (int k = 0; k < K; k++){
        w.push_back(tensor::random(width,width,D));
    }
}




std::vector<tensor> convolutional_layer::return_weights() const {
    return w;
}



tensor convolutional_layer::return_bias() const {
    return b;
}



pair convolutional_layer::weight_size() const {
    return pair(width,depth);
}



pair convolutional_layer::bias_size() const {
    return pair(b.size(index::x),b.size(index::z));
}


pair convolutional_layer::output_size() const {
    return pair(a.size(index::x),a.size(index::z));
}


int convolutional_layer::return_stride() const {
    return stride;
}


int convolutional_layer::return_padding() const {
    return padding;
}


int convolutional_layer::return_pooling_width() const {
    return pooling_width;
}




tensor convolutional_layer::ouput() const{
    return a;
}



tensor convolutional_layer::pool_index() const {
    return ind;
}

void convolutional_layer::update(std::vector<tensor> dw , tensor db){

    b += db;

    for (int k = 0; k < w.size(); k++){
        w[k] += dw[k];
    }
}


void convolutional_layer::pooling_convert(int width){
    
    if (output_width%width != 0){
        throw Exception("pooling layer: the pooling width must divide the input width.");
    }
    
    pooling_width = width;
    pooled = true;
    ind = tensor::zeros(output_width, output_width, a.size(index::z));
    a = tensor::zeros(output_width/width,output_width/width,a.size(index::z));
}


FunctionType convolutional_layer::return_funcType() const {
    return act_fn.return_funcType();
}



tensor convolutional_layer::feed_forward(const tensor& t){
    
    tensor A = convolve(t);
    
    if (pooled){
        a = pool(A);
    } else {
        a = A;
    }
    
    return a;
}


tensor convolutional_layer::convolve(const tensor & t){

    tensor z = tensor::zeros(output_width,output_width,(int) w.size());
    for (int k = 0; k < w.size(); k++){
        z.block(0,0,k,t.correlation(stride,stride,1,padding,padding,0,w[k]));
    }
    z += b;
    return act_fn(z);
}


tensor convolutional_layer::pool(const tensor & t){

    tensor S = a;
    ind *= 0.0;
    int n1,n2,n3;
    
    for (int x = 0; x < a.size(index::x); x++){
        for (int y = 0; y < a.size(index::y); y++){
            for (int z = 0; z < a.size(index::z); z++){
                S(x,y,z) = t.block(x*pooling_width, y*pooling_width, z, pooling_width, pooling_width,1).max(n1,n2,n3);
                ind(x*pooling_width + n1, y*pooling_width + n2, z*pooling_width + n3) = 1.0;
            }
        }
    }
    return S;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////
/* Fully connected layers */
////////////////////////////

/*
fully_connected_base::fully_connected_base(int n_in, int n_layer):
a(vector(n_layer)), b(vector(n_layer)), w(matrix(n_layer,n_in)) {


    std::default_random_engine generator;
    std::normal_distribution<double> weight_dist(0,1/sqrt(n_in)),bias_dist(0,1) ;

    for (int row=0; row<n_layer; row++){
        b(row) = bias_dist(generator);
        for (int col=0; col<n_in; col++){
            w(row,col) = weight_dist(generator);
        }
    }

}


void fully_connected_base::update(const matrix& dw, const vector & db){

    w += dw;
    b += db;
}





template <class FuncType>
fully_connected_layer<FuncType>::fully_connected_layer(int n_in, int n_layer): fully_connected_base(n_in,n_layer){
    std::cout << "fully connected layer created \n";
    std::cout << "input size = " << n_in << ", output size = " << n_layer  <<  "\n\n";
    
}


template <class FuncType>
vector fully_connected_layer<FuncType>::feed_forward(const vector& x){
    
    vector z = w*x + b;
    a = act_fn(z);
    return a;
}
*/