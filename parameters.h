#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include <stdio.h>
#include "neural_net.h"

class neural_net;



struct conv_parameters {
    
    conv_parameters() = default;
    conv_parameters(const std::vector<tensor>&, const tensor&);
    conv_parameters(const convolutional_layer*);
    
    conv_parameters operator+=(const conv_parameters&);
    conv_parameters operator*=(const double&);
    conv_parameters operator/=(const double&);
    
    std::vector<tensor> W;
    tensor B;
};

conv_parameters operator+(const conv_parameters&, const conv_parameters&);
conv_parameters operator-(const conv_parameters&, const conv_parameters&);




struct full_parameters {
    
    full_parameters() = default;
    full_parameters(const matrix &, const vector &);
    full_parameters(const fully_connected_layer*);
    
    full_parameters operator+=(const full_parameters&);
    full_parameters operator*=(const double&);
    full_parameters operator/=(const double&);
    
    matrix W;
    vector B;
};

full_parameters operator+(const full_parameters&, const full_parameters&);
full_parameters operator-(const full_parameters&, const full_parameters&);




struct net_parameters {
    
    net_parameters(const neural_net *);
    
    void zero();

    net_parameters operator+=(const net_parameters&);
    net_parameters operator*=(const double&);
    net_parameters operator/=(const double&);

    std::vector<conv_parameters> conv;
    std::vector<full_parameters> full;
    full_parameters otp;

};


net_parameters operator+(net_parameters&, net_parameters&);
net_parameters operator-(net_parameters&, net_parameters&);
net_parameters operator*(double, net_parameters&);
net_parameters operator/(net_parameters&, double);


#endif /* GRADIENT_HPP */
