#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
#include "matrix.h"
/*
This file contains the activation functions to be used in the layers of the neural net.
*/


enum class HiddenType {
    sigmoid,
    tanh,
    ReLU,
};


enum class OutputType {
    sigmoid,
    softmax,
};



template <class T>
class ActivationFunction {
public:

    T operator() (const T & x) const {
        return fn(x);
    }

protected:

    ActivationFunction() = default;

    T (*fn)(const T&);

    static T sigmoid_function(const T & x);
    static T tanh_function(const T & x);
    static T ReLU_function(const T & x);
    static T softmax_function(const T & x);
};



template <class T>
class HiddenFunction: public ActivationFunction<T>{
public:
    HiddenFunction(HiddenType);
    HiddenType return_funcType() const;
private:
    HiddenType fn_type;
};



class OutputFunction: public ActivationFunction<vector> {
public:

    OutputFunction(OutputType _fn);
    OutputType return_funcType() const;
private:
    OutputType fn_type;
};



/////////////////////
// Implementations //
/////////////////////


template <class T>
HiddenFunction<T>::HiddenFunction(HiddenType _fn){
    switch (_fn){
        case HiddenType::sigmoid:
            this->fn = &ActivationFunction<T>::sigmoid_function;
            fn_type = HiddenType::sigmoid;
            break;
        case HiddenType::tanh:
            this->fn = &ActivationFunction<T>::tanh_function;
            fn_type = HiddenType::tanh;
            break;
        case HiddenType::ReLU:
            this->fn = &ActivationFunction<T>::ReLU_function;
            fn_type = HiddenType::ReLU;
            break;
    }
}



template <class T>
HiddenType HiddenFunction<T>::return_funcType() const{
    return fn_type;
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
