#ifndef TRAINING_ALGORITHM_HPP
#define TRAINING_ALGORITHM_HPP

#include <stdio.h>
#include <set>
#include "neural_net.h"
#include "gradient_functions.h"
#include "output_gradient.h"



using training_data = std::vector<std::pair<tensor,vector>>;

class TrainingMethod;



class TrainingStatergy {
public:
    
    TrainingStatergy();
    
    void set_momentum(double _p){p = _p;}
    void set_max_iterations(int n){max_epochs = n;}
    void set_rate(double _r){r = _r;}
    void set_batch_size(int _n){batch_size = _n;}
    
    double get_error(){return err;}
    
private:
    
    std::unique_ptr<TrainingMethod> method = nullptr;
    training_data data;
    
    double p{0};
    int max_epochs{100};
    double r{0.2};
    double err;
    int batch_size;
};




class TrainingMethod {
public:
    
    TrainingMethod(neural_net *, LossType);
    
    void epoch_increment(training_data data);
    net_parameters get_parameters();
    void zero_parameters();
    
protected:
    
    virtual void iterate(tensor x, vector y) = 0;
    
    neural_net * NN;
    LossType loss;
    net_parameters params;
};




class Gradient_Descent: public TrainingMethod {
public:
    
    Gradient_Descent(neural_net *, LossType);
    
private:
    void iterate(tensor x, vector y);
    void backpropagate(tensor x, vector y);
    void add_params();

    std::vector<ConvolutionGradient> conv_grad;
    std::vector<FullyConnectedGradient> full_grad;
    std::unique_ptr<OutputGradient>  otp_grad = nullptr;
};



#endif /* TRAINING_ALGORITHM_HPP */
