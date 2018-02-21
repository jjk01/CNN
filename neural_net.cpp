#include "neural_net.h"
/*
neural_net::neural_net() = default;

neural_net::~neural_net(){

    delete inpt;

    for (int n=0; n<conv.size(); n++){
        delete conv[n];
    }

    for (int n=0; n<full.size(); n++){
        delete full[n];
    }
}




neural_net::neural_net(neural_net const& arg) {

    sizes = arg.sizes;
    inpt = arg.inpt -> clone();

    for (int n=0; n< arg.conv.size(); n++){
        conv.push_back(arg.conv[n] -> clone());
    }

    for (int n=0; n< arg.full.size(); n++){
        full.push_back(arg.full[n] -> clone());
    }
}



neural_net& neural_net::operator=(const neural_net& arg){

    sizes = arg.sizes;

    input_layer * _inpt = arg.inpt -> clone();
    std::vector<convolutional_base*> _conv;
    std::vector<fully_connected_base*> _full;


    for (int n=0; n<arg.conv.size(); n++){
        _conv.push_back(arg.conv[n] -> clone());
    }

    for (int n=0; n<arg.full.size(); n++){
        _full.push_back(arg.full[n] -> clone());
    }

    delete inpt;

    for (int n=0; n<conv.size(); n++){
        delete conv[n];
    }

    for (int n=0; n<full.size(); n++){
        delete full[n];
    }

    inpt = _inpt;
    conv = _conv;
    full = _full;

    return * this;
}


void neural_net::add_input_layer(int W_in, int D){

    if (inpt == nullptr) {
        inpt = new input_layer(W_in,W_in,D);
    } else {
        throw Exception("input already declared");
    }

}

void neural_net::add_convolution_layer(Activation::FunctionType func, int W_out, int K, int W_f, int S){
    
    int W_in, Depth;
    
    if (inpt== nullptr || !full.empty()){
        throw Exception("Convolutional layer must be declared before fully connected layers and after the input.");
    }
    
    if (conv.empty()){
        pair prev_size = inpt -> output_size();
        W_in = prev_size.x;
        Depth = prev_size.y;
    } else {
        pair prev_size = conv.back() -> output_size();
        W_in = prev_size.x;
        Depth = prev_size.y;
    }

    switch (func) {
        case Activation::FunctionType::sigmoid:
            conv.push_back(new convolutional_layer<Activation::sigmoid_function>(W_in, Depth, W_out, K, W_f, S));
            break;
        case Activation::FunctionType::tanh:
            conv.push_back(new convolutional_layer<Activation::tanh_function>(W_in, Depth, W_out, K, W_f, S));
            break;
        case Activation::FunctionType::ReLU:
            conv.push_back(new convolutional_layer<Activation::ReLU_function>(W_in, Depth, W_out, K, W_f, S));
            break;
        case Activation::FunctionType::softmax:
            break;
    };

    sizes.W0.push_back(conv.back() -> weight_size());
    sizes.B0.push_back(conv.back() -> bias_size());
    sizes.K.push_back((conv.back() -> output_size()).y);
}



void neural_net::add_pooling_layer(int W_pooling){
    
    if (inpt == nullptr) {
        throw Exception("need to declare input layer before adding additional layers");
    } else if (conv.empty()){
        pair o = inpt -> output_size();
        delete inpt;
        inpt = new pooling_input(o.x,o.y,W_pooling);
    } else {
        auto L = conv.back() -> pooling_convert(W_pooling);
        conv.pop_back();
        conv.push_back(L->clone());
    }
}



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
