#ifndef NEUTRON_LAYER_H
#define NEUTRON_LAYER_H

#include "activation_functions.h"


struct pair {

    pair(): x(0),y(0){};
    pair(int _x, int _y): x(_x), y(_y){};

    int x;
    int y;
};



//////////////////
/* Input Layers */
//////////////////


class input_layer {
public:

    input_layer(int W_in, int W_out, int D);
    void pooling_convert(int Pw);
    tensor feed_forward(const tensor &);

    tensor get_output() const;
    pair output_size() const;

protected:
    bool pooling;
    int input_width;
    int pooling_with;
    tensor a;
};




////////////////////////
/* Convolution Layers */
////////////////////////


class convolutional_layer {
public:

    convolutional_layer(FunctionType ft, int W_in, int Depth, int W_out, int Kernals, int W_filter, int Stride);
    
    tensor feed_forward(const tensor&);
    void pooling_convert(int Pw);

    FunctionType return_funcType() const;
    bool pooling() const;
    
    std::vector<tensor> return_weights() const;
    tensor return_bias() const;

    int return_padding() const;
    int return_stride() const;
    int return_pooling_width() const;
    
    pair weight_size() const;
    pair bias_size() const;
    pair output_size() const;
    tensor ouput() const;
    tensor pool_index() const;

    void update(std::vector<tensor>, tensor);

private:

    ActivationFunction<tensor> act_fn;
    
    tensor convolve(const tensor &);
    tensor pool(const tensor &);

    int padding;
    int stride;
    int width;
    int depth;
    int output_width;

    std::vector<tensor> w;
    tensor b;
    tensor a;
    
    bool pooled;
    int pooling_width;
    tensor ind;
};






//////////////////////////
/* Fully connected part */
//////////////////////////
/*

class fully_connected_layer {
public:
    
    fully_connected_layer(int,int);
    vector feed_forward(const vector&);

    FunctionType return_funcType();

    void update(const matrix& dw, const vector & db);

    vector get_output() const {return a;};
    matrix get_weight(){return w;}
    vector get_bias(){return b;}

    pair weight_size(){return pair(w.rows(),w.cols());}
    int bias_size(){return b.size();}
    int output_size(){return a.size();}

private:

    matrix w;
    vector b;
    vector a;
    
    ActivationFunction<matrix> fn;
};
*/

#endif /* NEUTRON_LAYER_H */
