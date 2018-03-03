#ifndef OUTPUT_FUNCTIONS_H
#define OUTPUT_FUNCTIONS_H

#include <cmath>
#include "activation_functions.h"
#include <iostream>




template <class F>
class output_activation {
public:
    
    output_activation(enum LossType loss){
        static_cast<F*>(this) -> set_loss(loss);
    }
    
    vector implement(const vector& x){
        return static_cast<F*>(this) -> implement(x);
    }
    vector return_error(const vector& a, const vector& y){
        return static_cast<F*>(this) -> return_error(a,y);
    }
    
protected:
    output_activation() = default;
    vector (*fn)(const vector &, const vector &) = nullptr;
};


class softmax_function: public output_activation<softmax_function> {
public:
    
    vector implement(const vector&);
    vector return_error(const vector&, const vector&);
    void set_loss(enum LossType);
    
private:
    softmax_function() = default;
    static vector quadratic(const vector &, const vector &);
    static vector cross_entropy(const vector &, const vector &);
};


class sigmoid_function: public output_activation<sigmoid_function> {
public:
    
    vector implement(const vector& );
    vector return_error(const vector&, const vector&);
    void set_loss(enum LossType);
    
private:
    sigmoid_function() = default;
    static vector quadratic(const vector &, const vector &);
    static vector cross_entropy(const vector &, const vector &);
};

#endif /* OUTPUT_FUNCTIONS_H */
