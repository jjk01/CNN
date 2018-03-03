#ifndef output_gradient_hpp
#define output_gradient_hpp


#include "activation_functions.h"



enum class LossType {
    cross_entropy,
    quadratic,
};



class OutputGradient {
public:
    
    OutputGradient(OutputType, LossType);
    
    OutputType return_funcType() const;
    LossType return_lossType() const;
    
    vector operator() (const vector& a, const vector& y){
        return fn(a,y);
    }
    
private:
    
    LossType loss;
    OutputType fn_type;
    
    vector (*fn)(const vector &, const vector &) = nullptr;
    
    static vector cross_entropy_sigmoid(const vector & , const vector &);
    static vector cross_entropy_softmax(const vector & , const vector &);
    
    static vector quadratic_sigmoid(const vector & , const vector &);
    static vector quadratic_softmax(const vector & , const vector &);
};

#endif /* output_gradient_hpp */
