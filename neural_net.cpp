#include "neural_net.h"
#include <memory>


void neural_net::add_input_layer(int W, int D){
    inpt.reset(new input_layer(W,D));
}


void neural_net::add_convolution_layer(HiddenType ft, int W_out, int K, int W_f, int S){
    
    int W_in, Depth;
    
    if (inpt == nullptr){
        throw Exception("Input must declared prior to hidden layers.");
    } else if (conv.empty()){
        pair prev_size = inpt -> output_size();
        W_in = prev_size.x;
        Depth = prev_size.y;
    } else {
        pair prev_size = conv.back().output_size();
        W_in = prev_size.x;
        Depth = prev_size.y;
    }
    
    conv.push_back(convolutional_layer(ft,W_in, Depth, W_out, K, W_f, S));
}



void neural_net::add_pooling_layer(int W_pooling){
    
    if (inpt == nullptr){
        throw Exception("Input must declared prior to hidden layers.");
    } else if (conv.empty()){
        inpt -> pooling_convert(W_pooling);
    } else {
        conv.back().pooling_convert(W_pooling);
    }
}


void neural_net::add_fully_connected_layer(HiddenType ft, int n_out){
    
    int n_in;
    
    if (inpt == nullptr){
        throw Exception("input must be declared before fully connected layer.");
    } else if (conv.empty() && full.empty()){
        pair P = inpt -> output_size();
        n_in = P.x*P.x*P.y;
    } else if (!conv.empty() && full.empty()) {
        pair P = conv.back().output_size();
        n_in = P.x*P.x*P.y;
    } else {
        n_in = full.back().output_size();
    }
    
    full.push_back(hidden_layer(ft,n_in,n_out));
}



void neural_net::add_output_layer(OutputType _fn, int n_out){
    int n_in;
    
    if (inpt == nullptr){
        throw Exception("input must be declared before output layer.");
    } else if (conv.empty() && full.empty()){
        pair P = inpt -> output_size();
        n_in = P.x*P.x*P.y;
    } else if (!conv.empty() && full.empty()) {
        pair P = conv.back().output_size();
        n_in = P.x*P.x*P.y;
    } else {
        n_in = full.back().output_size();
    }
    
    otp.reset(new output_layer(_fn,n_in,n_out));

}


vector neural_net::action(const tensor & t) {
    
    forward_propagate(t);
    return otp -> get_output();
    
}

void neural_net::forward_propagate(const tensor & t) {
    
    tensor x = inpt -> feed_forward(t);
    
    for (auto conv_itr = conv.begin(); conv_itr != conv.end(); ++conv_itr){
        x = conv_itr -> feed_forward(x);
    }
    
    vector y{x};
    
    for (auto full_itr = full.begin(); full_itr != full.end(); ++full_itr){
        y = full_itr -> feed_forward(y);
    }
    
    y = otp -> feed_forward(y);
}




const std::vector<convolutional_layer> * neural_net::convolution_ptr() const {
    return &conv;
}

const std::vector<hidden_layer> * neural_net::full_ptr() const {
    return &full;
}

const input_layer * neural_net::input_ptr() const {
    return inpt.get();
}


const output_layer * neural_net::output_ptr() const {
    return otp.get();
}