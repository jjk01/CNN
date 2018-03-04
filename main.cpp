#include <iostream>
#include "neural_net.h"
#include "training_algorithm.h"
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
     

        neural_net NN;

        NN.add_input_layer(20,3);
        //NN.add_input_layer(10,3);
        NN.add_pooling_layer(2);
        NN.add_convolution_layer(HiddenType::sigmoid,9,1,4,1);
        //NN.add_convolution_layer(HiddenType::sigmoid,9,1,4,1);
        NN.add_pooling_layer(3);
        NN.add_fully_connected_layer(HiddenType::sigmoid,8);
        NN.add_output_layer(OutputType::softmax,8);


        clock_t start = clock();
        vector a = NN.action(x);
        double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout <<  "Time taken = " << duration << "\n";
        a.print();
        
        Gradient_Descent GD(&NN, LossType::cross_entropy);
        
        for (int n = 0; n < 1e3; ++n){
            GD.Train(x,y);
        }
        
        a = NN.action(x);
        a.print();
        



    } catch( Exception e ) {
        Exception E("catch block",e);
        E.what();
    }

    return 0;
}
