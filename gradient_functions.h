#ifndef GRADIENT_FUNCTIONS_H
#define GRADIENT_FUNCTIONS_H


#include "neural_net.h"


template <class T>
class HiddenGradient {
public:
    
    HiddenGradient(HiddenType);
    HiddenType return_funcType() const;
    
    T get_error() const {
        return err;
    }
    
    T operator() (const T & x) const {
        return fn(x);
    }
    
protected:
    
    HiddenType fn_type;
    T err;
    
    T (*fn)(const T&);
    
    static T sigmoid_gradient(const T & x);
    static T tanh_gradient(const T & x);
    static T ReLU_gradient(const T & x);
};



class ConvolutionGradient: public HiddenGradient<tensor>{
public:
    ConvolutionGradient(const convolutional_layer * _layer);
    
    const convolutional_layer * return_ptr() const;
    
    tensor pass_back(tensor X);
    tensor pass_back(vector X);
    
private:
    const convolutional_layer * layer;
};



class FullyConnectedGradient: public HiddenGradient<vector>{
public:
    FullyConnectedGradient(const hidden_layer * _layer);
    const hidden_layer * return_ptr() const;
    
    vector pass_back(vector X);
private:
    const hidden_layer * layer;
};




/////////////////////
// Implementations //
/////////////////////


template <class T>
HiddenGradient<T>::HiddenGradient(HiddenType _fn){
    switch (_fn){
        case HiddenType::sigmoid:
            this->fn = &HiddenGradient<T>::sigmoid_gradient;
            fn_type = HiddenType::sigmoid;
            break;
        case HiddenType::tanh:
            this->fn = &HiddenGradient<T>::tanh_gradient;
            fn_type = HiddenType::tanh;
            break;
        case HiddenType::ReLU:
            this->fn = &HiddenGradient<T>::ReLU_gradient;
            fn_type = HiddenType::ReLU;
            break;
    }
}



template <class T>
HiddenType HiddenGradient<T>::return_funcType() const{
    return fn_type;
}



template <class T>
T HiddenGradient<T>::sigmoid_gradient(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = x(k)*(1-x(k));
    }
    return y;
}



template <class T>
T HiddenGradient<T>::tanh_gradient(const T & x){
    T y(x);
    for (int k = 0; k < x.size(); k++){
        y(k) = 1 - x(k)*x(k);
    }
    return y;
}



template <class T>
T HiddenGradient<T>::ReLU_gradient (const T & x){
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
