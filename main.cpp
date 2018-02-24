#include <iostream>
#include "neural_net.h"
//#include "training_algorithm.hpp"
#include <ctime>
// check that the Weight gradient also holds true for the pooling layers.

int main(){

    tensor x = tensor::random(20,20,3);
    vector y(8);
    y(2) = 0.2;
    y(5) = 0.3;
    y(7) = 0.5;

    //training_set data;
    //data.push_back(std::make_pair(x,y));
    
    



    try {
        /*
        input_layer X(20,3);
        X.pooling_convert(2);
        
        
        convolutional_layer Y(FunctionType::sigmoid,10,3,8,1,3,1);
        Y.pooling_convert(2);
        
        tensor y = Y.feed_forward(X.feed_forward(x));
        y.print();*/
        

        neural_net NN;
        
        NN.add_input_layer(20,3);
        NN.add_pooling_layer(2);
        NN.add_convolution_layer(HiddenType::sigmoid,9,1,4,1);
        NN.add_pooling_layer(3);
        NN.add_fully_connected_layer(HiddenType::ReLU,8);
        NN.add_output_layer(OutputType::softmax,8);
        
        
        clock_t start = clock();
        vector a = NN.action(x);
        double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout <<  "Time taken = " << duration << "\n";
        a.print();
        
        /*
        
        Backpropagation BP(NN,data);

        BP.set_momentum(0.0);
        BP.set_max_iterations(100);
        BP.set_rate(0.2);
        BP.set_batch_size(1);

        clock_t start_1 = clock();
        BP.train();
        double duration_1 = ( std::clock() - start_1 ) / (double) CLOCKS_PER_SEC;
        std::cout <<  "Time taken = " << duration_1 << "\n";

        NN = BP.return_NN();

        a = NN.action(x);
        a.print();*/



        /*neural_net NN;

        NN.add_layer(convolutional_layer(FunctionType::sigmoid,27,3,9,1,4,1));
        NN.add_layer(pooled_convolutional_layer(FunctionType::sigmoid,9,1,5,1,3,1,2));
        NN.add_layer(hidden_layer(FunctionType::sigmoid,25,8));
        NN.add_layer(output_layer(OutputType::softmax,LossType::cross_entropy,8,8));

        clock_t start = clock();
        vector a = NN.action(x);
        double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;


        std::cout <<  "Time taken = " << duration << "\n";


        a.print();*/


    } catch( Exception e ) {
        e.what();
    }

    return 0;
}
