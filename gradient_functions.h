#ifndef GRADIENT_FUNCTIONS_H
#define GRADIENT_FUNCTIONS_H


#include "activation_functions.h"


template <class T>
class GradientFunction {
public:
    
    GradientFunction(FunctionType _fn): fn_type(_fn){
        switch (_fn){
            case FunctionType::sigmoid:
                fn = &GradientFunction::sigmoid_gradient;
                break;
            case FunctionType::tanh:
                fn = &GradientFunction::tanh_gradient;
                break;
            case FunctionType::ReLU:
                fn = &GradientFunction::ReLU_gradient;
                break;
            case FunctionType::softmax:
                throw Exception("cannot specify softmax as hidden activation gradient");
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
    
    static T sigmoid_gradient(const T & x);
    static T tanh_gradient(const T & x);
    static T ReLU_gradient(const T & x);
    static T softmax_gradient(const T & x);
};



template <class T>
T GradientFunction<T>::sigmoid_gradient(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = x(k)*(1-x(k));
    }
    return y;
}



template <class T>
T GradientFunction<T>::tanh_gradient(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = 1 - x(k)*x(k);
    }
    return y;
}



template <class T>
T GradientFunction<T>::ReLU_gradient (const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        if (0 < x(k)){
            y(k) = 1;
        } else {
            y(k) = 0;
        }
    }
    return y;
}


#endif /* GRADIENT_FUNCTIONS_H */
