#ifndef GRADIENT_FUNCTIONS_H
#define GRADIENT_FUNCTIONS_H


#include "activation_functions.h"

/*
This file contains.
*/



struct sigmoid_gradient: public base_gradient<sigmoid_gradient> {
    
    template <class T>
    T operator() (const T & x){
        T y(x);
        for (int k = 0; k < x.size(); k++){
            y(k) = x(k)*(1-x(k));
        }
        return y;
    }
    
    FunctionType return_funcType(){
        return FunctionType::sigmoid;
    }
    
};



struct tanh_gradient: public base_gradient<tanh_gradient> {
    
    template <class T>
    T operator() (const T & x){
        T y(x);
        for (int k = 0; k < x.size(); k++){
            y(k) = 1 - x(k)*x(k);
        }
        return y;
    }
    
    FunctionType return_funcType(){
        return FunctionType::tanh;
    }
};



struct ReLU_gradient: public base_gradient<ReLU_gradient>  {
    
    template <class T>
    T operator() (const T & x){
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
    
    FunctionType return_funcType(){
        return FunctionType::ReLU;
    }
};


#endif /* GRADIENT_FUNCTIONS_H */
