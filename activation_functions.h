#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
#include "matrix.h"

/*
This file contains the activation functions to be used in the layers of the neural net.
*/


enum class FunctionType {
    sigmoid,
    tanh,
    ReLU,
    softmax,
};




template <class T>
static T sigmoid_function(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = 1/(1 + std::exp(-x(k)));
    }
    return y;
}



template <class T>
static T tanh_function(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = tanh(x(k));
    }
    return y;
}


template <class T>
static T ReLU_function (const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = std::max(0.0,x(k));
    }
    return y;
}




template <class T>
static T softmax_function (const T & x){
    T y(x);
    double N = 0;
    for (int k = 0; k < x.size(); k++){
        double z = std::exp(-x(k));
        N += z;
        y(k) = z;
    }
    return y;
}



template <class T>
class ActivationFunction {
public:
    
    ActivationFunction(FunctionType _fn): fn_type(_fn){
        switch (_fn){
            case FunctionType::sigmoid:
                fn = &sigmoid_function;
                break;
            case FunctionType::tanh:
                fn = &tanh_function;
                break;
            case FunctionType::ReLU:
                fn = &ReLU_function;
                break;
            case FunctionType::softmax:
                fn = &softmax_function;
                break;
        }
    };
    
    T operator() (const T & x) const {
        return fn(x);
    }
    
    FunctionType return_funcType() const {
        return fn_type;
    }
    
private:
    FunctionType fn_type;
    T (*fn)(const T&);
};

#endif /* ACTIVATION_FUNCTIONS_H */
