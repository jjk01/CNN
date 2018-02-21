#ifndef TRAINING_ALGORITHM_HPP
#define TRAINING_ALGORITHM_HPP

#include <stdio.h>
#include <set>
#include "neural_net.h"

using training_set = std::vector <std::pair<tensor,vector>>;

//using tensor_array = std::vector<std::vector<tensor>>;
//using tensor_vector = std::vector<tensor>;
//using matrix_vector = std::vector<matrix>;
//using vector_vector = std::vector<vector>;



class Backpropagation {
public:
    
    Backpropagation();
    Backpropagation(neural_net);
    

private:
    
    std::vector<FunctionType> conv_func;
    std::vector<bool> conv_pool;
    std::vector <tensor> conv_err;
    
    std::vector<FunctionType> full_func;
    std::vector<bool> full_pool;
    std::vector<vector> full_err;
    
    tensor conv_pass_back(tensor X, int ind);
    tensor conv_pass_back(vector X, int ind);
    tensor pool_pass_back(tensor X, int ind);
    tensor pool_pass_back(vector X, int ind);
    
    void output_error(vector y, vector a);

    void backpropagate(tensor x, vector y);
    void update_net();
    
    neural_net NN;
};



/*
class Trainer {
public:

    Trainer(neural_net, training_set);

    void train();

    //void set_momentum(double _p){p = _p;}
    //void set_max_iterations(int n){max_epochs = n;}
    //void set_rate(double _r){r = _r;}
    void set_batch_size(int _n){batch_size = _n;}

    double error(){return err;}
    neural_net return_NN(){return NN;}

protected:

    neural_net NN;
    training_set data;

    double p{0};
    int max_epochs{100};
    double r{0.2};
    double err;
    int batch_size;

    void backpropagate(tensor,vector);

    // These return the gradients of the Weight matrices/tensors.
    tensor_vector conv_gradient(tensor);
    tensor_vector conv_gradient(int);
    matrix connected_gradient(int);
    matrix output_gradient();

    // This error corresponds to the bias gradient
    vector get_output_error();
    vector get_connected_error(int);
    tensor get_conv_error(int);


    void update_conv(tensor_array, tensor_vector);
    void update_connected(matrix_vector, vector_vector);
    void update_output(matrix, vector);

};

 */





#endif /* TRAINING_ALGORITHM_HPP */
