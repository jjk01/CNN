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
    input_layer(int W, int D);
    void pooling_convert(int Pw);
    tensor feed_forward(const tensor &);

    tensor get_output() const;
    pair output_size() const;

private:
    
    void pool(const tensor &);
    
    bool pooled;
    int input_width;
    int depth;
    int pooling_width;
    tensor a;
};




////////////////////////
/* Convolution Layers */
////////////////////////


class convolutional_layer {
public:

    convolutional_layer(HiddenType fn, int W_in, int Depth, int W_out, int Kernals, int W_filter, int Stride);
    
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


class fully_connected_layer {
public:
    
    fully_connected_layer(HiddenType, int,int);
    vector feed_forward(const vector&);

    FunctionType return_funcType();

    void update(const matrix& dw, const vector & db);

    vector get_output() const;
    matrix get_weight() const;
    vector get_bias() const;

    pair weight_size() const;
    int bias_size() const;
    int output_size() const;

protected:
    
    fully_connected_layer(OutputType,int,int);
    void initialise_parameters(int,int);

    matrix w;
    vector b;
    vector a;
    
    ActivationFunction<vector> fn;
};



class output_layer: public fully_connected_layer {
public:
    output_layer(OutputType, int,int);
};


#endif /* NEUTRON_LAYER_H */
