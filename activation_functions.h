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


enum class HiddenType {
    sigmoid,
    tanh,
    ReLU,
};


enum class OutputType {
    sigmoid,
    tanh,
    ReLU,
    softmax,
};




template <class T>
class ActivationFunction {
public:
 
    ActivationFunction(HiddenType);
    ActivationFunction(OutputType);
 
    T operator() (const T & x) const {
        return fn(x);
    }
    
    FunctionType return_funcType() const {
        return fn_type;
    }
    
protected:
    FunctionType fn_type;
    T (*fn)(const T&);
    
    static T sigmoid_function(const T & x);
    static T tanh_function(const T & x);
    static T ReLU_function(const T & x);
    static T softmax_function(const T & x);
};




template <class T>
ActivationFunction<T>::ActivationFunction(HiddenType _fn){
    switch (_fn){
        case HiddenType::sigmoid:
            fn = &ActivationFunction::sigmoid_function;
            fn_type = FunctionType::sigmoid;
            break;
        case HiddenType::tanh:
            fn = &ActivationFunction::tanh_function;
            fn_type = FunctionType::tanh;
            break;
        case HiddenType::ReLU:
            fn = &ActivationFunction::ReLU_function;
            fn_type = FunctionType::ReLU;
            break;
    }
}




template <class T>
ActivationFunction<T>::ActivationFunction(OutputType _fn){
    switch (_fn){
        case OutputType::sigmoid:
            fn = &ActivationFunction::sigmoid_function;
            fn_type = FunctionType::sigmoid;
            break;
        case OutputType::tanh:
            fn = &ActivationFunction::tanh_function;
            fn_type = FunctionType::tanh;
            break;
        case OutputType::ReLU:
            fn = &ActivationFunction::ReLU_function;
            fn_type = FunctionType::ReLU;
            break;
        case OutputType::softmax:
            fn = &ActivationFunction::softmax_function;
            fn_type = FunctionType::softmax;
            break;
    }
}


template <class T>
T ActivationFunction<T>::sigmoid_function(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = 1/(1 + std::exp(-x(k)));
    }
    return y;
}



template <class T>
T ActivationFunction<T>::tanh_function(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = tanh(x(k));
    }
    return y;
}



template <class T>
T ActivationFunction<T>::ReLU_function (const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = std::max(0.0,x(k));
    }
    return y;
}



template <class T>
T ActivationFunction<T>::softmax_function (const T & x){
    T y(x);
    double N = 0;
    for (int k = 0; k < x.size(); k++){
        double z = std::exp(-x(k));
        N += z;
        y(k) = z;
    }
    return y/N;
    
}



#endif /* ACTIVATION_FUNCTIONS_H */
