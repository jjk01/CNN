#ifndef output_gradient_hpp
#define output_gradient_hpp


#include "neural_net.h"



enum class LossType {
    cross_entropy,
    quadratic,
};



class OutputGradient {
public:
    
    OutputGradient(LossType, const output_layer *);
    
    OutputType return_funcType() const;
    LossType return_lossType() const;
    
    vector get_error() const;
    vector pass_back(const vector& a, const vector& y);
    
private:
    
    LossType loss;
    OutputType fn_type;
    vector err;
    const output_layer * layer;
    
    vector (*fn)(const vector &, const vector &) = nullptr;
    
    static vector cross_entropy_sigmoid(const vector & , const vector &);
    static vector cross_entropy_softmax(const vector & , const vector &);
    
    static vector quadratic_sigmoid(const vector & , const vector &);
    static vector quadratic_softmax(const vector & , const vector &);
};

#endif /* output_gradient_hpp */
