#include <iostream>
#include "neural_net.h"
#include "training_algorithm.h"
#include "data.h"
#include <ctime>




void print_performance(const std::vector<tensor>& images, const std::vector<vector> labels, neural_net * NN){
    
    
    std::size_t data_size = images.size();
    int batch_size = 1e3;
    
    std::set<unsigned int> index_set;
    if (batch_size < std::ceil(data_size/2.0)){
        
        while (index_set.size() < batch_size){
            index_set.insert(rand() % data_size);
        }
        
    } else {
        
        for (int k = 0; k < data_size; k++){
            index_set.insert(k);
        }
        std::set<unsigned int> temp_ind;
        while (temp_ind.size() < data_size - batch_size){
            temp_ind.insert(rand() % data_size);
        }
        for (auto it = temp_ind.begin(); it != temp_ind.end(); it++){
            index_set.erase(*it);
        }
    }

    
    
    int successes = 0;
    
    for (auto itr = index_set.begin(); itr != index_set.end(); ++itr){
        int n  = *itr;
        tensor x = images[n];
        vector y = labels[n];
        vector a = NN -> action(x);
        
        int max_ind = 0;
        
        for (int n = 0; n < a.size(); ++n){
            max_ind = (a(max_ind) < a(n)) ? n:max_ind;
        }
        
        if (y(max_ind)==1){
            successes++;
        }
    }
    
    std::cout << "successes = " << successes << std::endl;
    std::cout << "total number = " << batch_size << std::endl;
}




int main(){
    
    //read MNIST label into double vector
    
    std::vector<tensor> images;
    std::vector<vector> labels;
    
    std::string fn_images = "/Users/klatzow/Downloads/train-images.idx3-ubyte";
    std::string fn_labels = "/Users/klatzow/Downloads/train-labels.idx1-ubyte";
    
    read_images(fn_images,images);
    read_label(fn_labels,labels);
    
    //Construct set.
    
    training_data data;

    
    for(int n=0; n<images.size(); n++){
        tensor x = images[n];
        vector y = labels[n];
        data.push_back(std::make_pair(x,y));
    }
    
    neural_net NN;
    
    NN.add_input_layer(28,1);
    NN.add_convolution_layer(HiddenType::sigmoid,28,32,5,1);
    NN.add_pooling_layer(2);
    NN.add_convolution_layer(HiddenType::sigmoid,14,64,5,1);
    NN.add_pooling_layer(2);
    NN.add_fully_connected_layer(HiddenType::sigmoid,1024);
    NN.add_output_layer(OutputType::softmax,10);
    

    
    
    TrainingStatergy Trainer(&NN,data,LossType::cross_entropy);
    Trainer.set_max_iterations(25);
    Trainer.set_rate(0.1);
    Trainer.set_batch_size(100);
    Trainer.set_momentum(0.8);
    
    std::cout <<  "Beginning training. \n";
    clock_t start = std::clock();
    Trainer.train();
    //vector a = NN.action(data[1].first);
    //a.print();
    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
       
    
    std::cout <<  "Training finished. Time taken = " << duration << std::endl << std::endl;
    
    //std::cout <<  "training performance:" << std::endl << std::endl;
    //print_performance(images,labels,&NN);
    
    std::vector<tensor> test_images;
    std::vector<vector> test_labels;
    
    
    
    std::string fn_timages = "/Users/klatzow/Downloads/t10k-images.idx3-ubyte";
    std::string fn_tlabels = "/Users/klatzow/Downloads/t10k-labels.idx1-ubyte";
    
    read_images(fn_timages,test_images);
    read_label(fn_tlabels,test_labels);
    
    std::cout <<  "test performance:" << std::endl << std::endl;
    print_performance(test_images,test_labels,&NN);

    return 0;
}




/*
 int main(){
 
 tensor x = tensor::random(20,20,3);
 vector y(8);
 y(2) = 0.2;
 y(5) = 0.3;
 y(7) = 0.5;
 
 training_data data(10,std::make_pair(x,y));
 
 try {
 
 
 neural_net NN;
 
 NN.add_input_layer(20,3);
 NN.add_pooling_layer(2);
 NN.add_convolution_layer(HiddenType::sigmoid,9,1,4,1);
 NN.add_pooling_layer(3);
 NN.add_fully_connected_layer(HiddenType::sigmoid,8);
 NN.add_output_layer(OutputType::softmax,8);
 
 clock_t start = clock();
 
 vector a = NN.action(x);
 a.print();
 
 
 TrainingStatergy Trainer(&NN,data,LossType::cross_entropy);
 Trainer.set_max_iterations(200);
 Trainer.set_rate(0.1);
 Trainer.set_batch_size(10);
 Trainer.train();
 
 
 a = NN.action(x);
 double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
 std::cout <<  "Time taken = " << duration << "\n";
 a.print();
 
 
 } catch( Exception e ) {
 Exception E("catch block",e);
 E.what();
 }
 
 return 0;
 }
*/

