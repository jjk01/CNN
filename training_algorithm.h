#ifndef TRAINING_ALGORITHM_HPP
#define TRAINING_ALGORITHM_HPP

#include <stdio.h>
#include <set>
#include "neural_net.h"
#include "gradient_functions.h"
#include "output_gradient.h"
#include "set"


using training_data = std::vector<std::pair<tensor,vector>>;

class TrainingMethod;



class TrainingStatergy {
public:
    
    TrainingStatergy(neural_net *, const training_data &, LossType);
    
    void train();
    
    void set_momentum(double _p){p = _p;}
    void set_max_iterations(int n){max_epochs = n;}
    void set_rate(double _r){r = _r;}
    void set_batch_size(int _n){batch_size = _n;}
    
    double get_error(){return err;}
    
private:
    
    void increment_epoch(std::vector<int> indices);
    std::vector<int> generate_batch();
    
    std::unique_ptr<TrainingMethod> method = nullptr;
    training_data data;
    neural_net * NN;
    
    double p{0};
    int max_epochs{100};
    double r{0.2};
    double err;
    int batch_size;
    
    bool print_progress{false};
};




class TrainingMethod {
public:
    
    TrainingMethod(neural_net *, LossType);
    virtual void iterate(vector y) = 0;
    net_parameters get_parameters();
    void zero_parameters();
    
protected:
    
    LossType loss;
    net_parameters params;
};




class Gradient_Descent: public TrainingMethod {
public:
    
    Gradient_Descent(neural_net *, LossType);
    void iterate(vector y);
    
private:
    
    void backpropagate(vector y);
    void add_params();
    
    const input_layer * inpt;
    std::vector<ConvolutionGradient> conv_grad;
    std::vector<FullyConnectedGradient> full_grad;
    std::unique_ptr<OutputGradient>  otp_grad = nullptr;
};



#endif /* TRAINING_ALGORITHM_HPP */
