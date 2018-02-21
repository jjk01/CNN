#include "neural_net.h"


void neural_net::add_input_layer(int W, int D){
    inpt = input_layer(W,D);
}


void neural_net::add_convolution_layer(FunctionType ft, int W_out, int K, int W_f, int S){
    
    int W_in, Depth;
    
    if (conv.empty()){
        pair prev_size = inpt.output_size();
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
    
    if (conv.empty()){
        inpt.pooling_convert(W_pooling);
    } else {
        conv.back().pooling_convert(W_pooling);
    }
}



const std::vector<convolutional_layer> * neural_net::convolution_ptr() const {
    return &conv;
}

const std::vector<fully_connected_layer> * neural_net::full_ptr() const {
    return &full;
}

const input_layer * neural_net::input_ptr() const {
    return &inpt;
}


/*












void neural_net::add_fully_connected_layer(Activation::FunctionType func, int n_out){
    
    int n_in;
    
    if (inpt== nullptr){
        throw Exception("input must be declared before convolutional layer.");
    }
    
    if (conv.empty() && full.empty()){
        pair P = inpt -> output_size();
        n_in = P.x*P.x*P.y;
    } else if (!conv.empty() && full.empty()) {
        pair P = conv.back() -> output_size();
        n_in = P.x*P.x*P.y;
    } else {
        n_in = full.back() -> output_size();
    }

    switch (func) {
        case Activation::FunctionType::sigmoid:
            full.push_back(new fully_connected_layer<Activation::sigmoid_function>(n_in,n_out));
            break;
        case Activation::FunctionType::tanh:
            full.push_back(new fully_connected_layer<Activation::tanh_function>(n_in,n_out));
            break;
        case Activation::FunctionType::ReLU:
            full.push_back(new fully_connected_layer<Activation::ReLU_function>(n_in,n_out));
            break;
        case Activation::FunctionType::softmax:
            full.push_back(new fully_connected_layer<Activation::softmax_function>(n_in,n_out));
            break;
    };

    sizes.W1.push_back(full.back() -> weight_size());
    sizes.B1.push_back(full.back() -> bias_size());
}


vector neural_net::action(const tensor & t) {

    forward_propagate(t);
    return full.back() -> get_output();

}



void neural_net::forward_propagate(const tensor & t) {
    
    tensor x = inpt -> feed_forward(t);

    for (int n = 0; n < conv.size(); n++){
        x = conv[n] -> feed_forward(x);
    }

    vector y{x};

    for (int n = 0; n < full.size(); n++){
        y = full[n] -> feed_forward(y);
    }
}*/
