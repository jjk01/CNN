#include "parameters.h"



conv_parameters::conv_parameters(const convolutional_layer* layer){
    
    pair w_size = layer -> weight_size();
    pair b_size = layer -> bias_size();
    int kernals = layer -> num_kernals();
    
    tensor w = tensor::zeros(w_size.x,w_size.x,w_size.y);
    B = tensor::zeros(b_size.x,b_size.x,b_size.y);
    
    for (int n = 0; n < kernals; ++n){
        W.push_back(w);
    }
}


conv_parameters::conv_parameters(const std::vector<tensor> & _W, const tensor & _B): W(_W), B(_B){}


    
conv_parameters conv_parameters::operator+=(const conv_parameters& arg){
    
    if (W.size() != arg.W.size()){
        throw "convolutional parameters: number of weights do not agree.";
    }
    
    B += arg.B;
    for (int n = 0; n < W.size(); ++n){
        W[n] += arg.W[n];
    }
    
    return *this;
}


conv_parameters conv_parameters::operator*=(const double& d){
    
    B *= d;
    for (int n = 0; n < W.size(); ++n){
        W[n] *= d;
    }
    return *this;
}



conv_parameters conv_parameters::operator/=(const double& d){
    B /= d;
    for (int n = 0; n < W.size(); ++n){
        W[n] /= d;
    }
    return *this;
}



conv_parameters operator+(const conv_parameters& X, const conv_parameters& Y){
    conv_parameters Z(X);
    Z += Y;
    return Z;
}


conv_parameters operator-(const conv_parameters& X, const conv_parameters& Y){
    conv_parameters Z(Y);
    Z *= -1;
    Z += X;
    return Z;
}




full_parameters::full_parameters(const fully_connected_layer* arg){
    pair w_size = arg -> weight_size();
    int b_size = arg -> bias_size();
    
    W = matrix::zeros(w_size.x, w_size.y);
    B = matrix::zeros(b_size,1);

}

full_parameters::full_parameters(const matrix & _W, const vector & _B): W(_W), B(_B){}

full_parameters full_parameters::operator+=(const full_parameters& arg){
    
    B += arg.B;
    W += arg.W;
    return *this;
}


full_parameters full_parameters::operator*=(const double& m){
    B *= m;
    W *= m;
    return *this;
}


full_parameters full_parameters::operator/=(const double& d){
    B /= d;
    W /= d;
    return *this;
}
    


full_parameters operator+(const full_parameters& X, const full_parameters& Y){
    full_parameters Z(Y);
    Z += X;
    return Z;
}



full_parameters operator-(const full_parameters& X, const full_parameters& Y){
    full_parameters Z(Y);
    Z *= -1;
    Z += X;
    return Z;
}



net_parameters::net_parameters(const neural_net * NN){
    
    const std::vector<convolutional_layer> * conv_ptr = NN -> convolution_ptr();
    const std::vector<hidden_layer> * full_ptr = NN -> full_ptr();
    const output_layer * otp_ptr = NN -> output_ptr();
    
    for (auto itr = conv_ptr -> begin(); itr != conv_ptr -> end(); ++itr){
        conv.push_back(conv_parameters(itr -> get_pointer()));
    }
    
    for (auto itr = full_ptr -> begin(); itr != full_ptr -> end(); ++itr){
        full.push_back(full_parameters(itr -> get_pointer()));
    }
    
    otp = full_parameters(otp_ptr);
}


void net_parameters::zero(){
    for (int n = 0; n < conv.size(); ++n){
        conv[n] *= 0;
    }
    
    for (int n = 0; n < full.size(); ++n){
        full[n] *= 0;
    }
    
    otp *= 0;
}

net_parameters net_parameters::operator+=(const net_parameters& arg) {
    
    if (arg.conv.size() != conv.size() || arg.full.size() != full.size()){
        throw Exception("+= net parameter size mismatch.");
    }

    for (int n = 0; n < conv.size(); ++n){
        conv[n] += arg.conv[n];
    }
    
    for (int n = 0; n < full.size(); ++n){
        full[n] += arg.full[n];
    }
    
    otp += arg.otp;
    
    return *this;
}



net_parameters net_parameters::operator*=(const double& m) {
    
    for (int n = 0; n < conv.size(); ++n){
        conv[n] *= m;
    }
    
    for (int n = 0; n < full.size(); ++n){
        full[n] *= m;
    }
    
    otp *= m;
    
    return *this;
}


net_parameters net_parameters::operator/=(const double& d) {
    
    for (int n = 0; n < conv.size(); ++n){
        conv[n] /= d;
    }
    
    for (int n = 0; n < full.size(); ++n){
        full[n] /= d;
    }
    
    otp /= d;
    
    return *this;
}



net_parameters operator+(net_parameters& l, net_parameters& r) {
    net_parameters X{l};
    X += r;
    return X;
}




net_parameters operator-(net_parameters& l, net_parameters& r) {

    net_parameters X{r};
    X *= -1.0;
    X += l;
    return X;
}



net_parameters operator*(double m, net_parameters& r) {

    net_parameters X{r};
    X *= m;
    return X;
}


net_parameters operator/(net_parameters& r, double d) {

    net_parameters X{r};
    X /= d;
    return X;
}
