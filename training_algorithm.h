#ifndef TRAINING_ALGORITHM_HPP
#define TRAINING_ALGORITHM_HPP

#include <stdio.h>
#include <set>
#include "neural_net.h"
#include "gradient_functions.h"
#include "output_gradient.h"


class Gradient_Descent {
public:
    
    Gradient_Descent(neural_net *, LossType);
    void Train(tensor x, vector y);
    
private:

    void backpropagate(tensor x, vector y);
    void add_params();
    
    neural_net * NN;
    LossType loss;

    std::vector<ConvolutionGradient> conv_grad;
    std::vector<FullyConnectedGradient> full_grad;
    std::unique_ptr<OutputGradient>  otp_grad = nullptr;
    
    net_parameters params;
};



/*
class TrainingStatergy {
public:

    Trainer(neural_net, training_set);

    void train();

    void set_momentum(double _p){p = _p;}
    void set_max_iterations(int n){max_epochs = n;}
    void set_rate(double _r){r = _r;}
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
};

 */





#endif /* TRAINING_ALGORITHM_HPP */
