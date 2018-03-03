#include <stdio.h>
#include <algorithm>
#include "training_algorithm.h"





Gradient_Descent::Gradient_Descent(neural_net * _NN, LossType _loss): NN(_NN), loss(_loss){
    
    const std::vector<convolutional_layer> * conv_ptr = NN -> convolution_ptr();
    const std::vector<hidden_layer> * full_ptr = NN -> full_ptr();
    const output_layer * otp_ptr = NN -> output_ptr();
    
    for (auto itr = conv_ptr -> begin(); itr != conv_ptr -> end(); ++itr){
        conv_pooling.push_back(itr->pooling());
        conv_grad.push_back(ConvolutionGradient(itr -> get_pointer()));
    }
    
    for (auto itr = full_ptr -> begin(); itr != full_ptr -> end(); ++itr){
        full_grad.push_back(FullyConnectedGradient(itr -> get_pointer()));
    }
    
    otp_grad.reset(new OutputGradient(otp_ptr->return_funcType(),loss));

}

    
void Gradient_Descent::backpropagate(tensor x, vector y){
    vector a = NN -> action(x);
}



/*
 tensor conv_pass_back(vector, HiddenGradient<tensor>);
 vector full_pass_back(vector, HiddenGradient<vector>);
 vector output_error(vector y, vector a);
 
 void backpropagate(tensor x, vector y);
 void update_net();
 
 neural_net * NN;
 
 std::vector<bool> conv_pooling;
 std::vector<HiddenGradient<tensor>> conv_grad;
 std::vector<HiddenGradient<vector>> full_grad;
 OutputGradient otp_grad;
 };

 
tensor Backpropagation::conv_pass_back(tensor T, ind){
    
    if (conv_pool[n]){
        
        conv_err[n] *= 0.0;
        int pooling_width = NN.conv[n] -> pooling_width();
        tensor ind = NN.conv[n] -> pool_index();
        
        
        for (int x = 0; x < T.size(index::x); x++){
            int X = pooling_width*x;
            for (int y = 0; y < T.size(index::y); y++){
                int Y = pooling_width*y;
                for (int z = 0; z < T.size(index::z); z++){
                    
                    if ( ind(X,Y,z) == 1 ){
                        conv_err[n](X,Y,z) = T(x,y,z);
                    }
                }
            }
        }
        
        return conv_err[n];
    }

    
}

tensor conv_pass_back(vector X, tensor err){
    
    if (pooling){
        tensor A = tensor::zeros(a.size(index::x), a.size(index::y), a.size(index::z));
        A.set_data(X.return_data());
        return pass_back(A);
    }
}


tensor pool_pass_back(tensor X);
tensor pool_pass_back(vector X);
    
void output_error(vector y, vector a);
    
void backpropagate(tensor x, vector y);
void update_net();
    
    neural_net NN;
};


TrainingAlgorithm::TrainingAlgorithm(neural_net _NN, training_set _data): NN(_NN), data(_data){};


void TrainingAlgorithm::train(){
    for( int j=0; j< max_epochs; j++){
        implement();
        std::cout << "Epoch:  " << j+1 << ", error:  " << err << "\n";
    }
}




void TrainingAlgorithm::update_conv(std::vector<std::vector<tensor>> W, std::vector<tensor> B){

    if (W.size() != NN.conv.size() || B.size() != NN.conv.size()){
        throw Exception("neural net update. Conv change in weight/bias number does not match net weight/bias.");
    }

    for (int k = 0; k < W.size(); k++){
        NN.conv[k] -> update(W[k], B[k]);
    }

}


void TrainingAlgorithm::update_connected(std::vector<matrix> W, std::vector<vector> B){

    if (W.size() != NN.full.size() || B.size() != NN.full.size()){
        throw Exception("neural net update. Connected change in weight/bias number does not match net weight/bias numbers.");
    }

    for (int k = 0; k < W.size(); k++){
        NN.full[k] -> update(W[k],B[k]);
    }
}



void TrainingAlgorithm::update_output(matrix W, vector B){
    NN.otp -> update(W,B);
}



Backpropagation::Backpropagation(neural_net _NN, training_set _data): TrainingAlgorithm(_NN,_data){};


void Backpropagation::set_batch_size(int n){
    batch_size = n;
}

void Backpropagation::implement(){

    parameters A = parameters::zeros(NN.return_sizes());

    std::set<unsigned int> indices;
    if (batch_size < std::ceil(sample_size/2.0)){

        while (indices.size() < batch_size){
            indices.insert(rand() % sample_size);
        }

    } else {

        for (int k = 0; k < sample_size; k++){
            indices.insert(k);
        }
        std::set<unsigned int> temp_ind;
        while (temp_ind.size() < sample_size - batch_size){
            temp_ind.insert(rand() % sample_size);
        }
        for (auto it = temp_ind.begin(); it != temp_ind.end(); it++){
            indices.erase(*it);
        }
    }

    for (auto it = indices.begin(); it != indices.end(); it++){

        parameters B;

        tensor x = data[*it].first;
        vector y = data[*it].second;

        backpropagate(x,y);

        for (int n = 0; n < NN.conv.size(); n++){

            if (n==0){
                B.dW_conv.push_back(conv_gradient(x));
            } else {
                B.dW_conv.push_back(conv_gradient(n));
            }
            B.dB_conv.push_back(get_conv_error(n));
        }

        for (int n = 0; n < NN.full.size(); n++){
            B.dW_full.push_back(connected_gradient(n));
            B.dB_full.push_back(get_connected_error(n));
        }

        B.dW_otp = output_gradient();
        dB_otp = get_output_error();

        B *= (-r/batch_size);
        A += B;
    }

    update_conv(A.dW_conv,A.dB_conv);
    update_connected(dW_full,dB_full);
    update_output(dW_otp,dB_otp);

    err = dB_otp.norm()/r;

}




void Backpropagation::backpropagate(tensor x, vector y){

    NN.forward_propagate(x);
    vector e1 = NN.otp -> pass_back(y);

    for (auto rit = NN.full.rbegin(); rit != NN.full.rend(); rit++){
        e1 = (*rit) -> pass_back(e1);
    }

    if (! NN.conv.empty()){

        tensor e2 = NN.conv.back() -> pass_back(e1);

        for (auto rit = NN.conv.rbegin()+1; rit != NN.conv.rend(); rit++){
            e2 = (*rit) -> pass_back(e2);
        }
    }
}


std::vector<tensor> Backpropagation::conv_gradient(int n){

    if (n<1){
        throw Exception("neural net: input conv gradient called incorrectly. call with input tensor instead");
    }

    tensor a = NN.conv[n-1] -> get_output();
    tensor err = NN.conv[n] -> get_error();
    int S = NN.conv[n] -> get_stride();
    int P = NN.conv[n] -> get_padding();

    return a.correlation(S,S,P,P,err);
}

std::vector<tensor> Backpropagation::conv_gradient(tensor x){

    tensor err = NN.conv.front() -> get_error();
    int S = NN.conv.front() -> get_stride();
    int P = NN.conv.front() -> get_padding();

    return x.correlation(S,S,P,P,err);
}

matrix Backpropagation::connected_gradient(int n){

    vector a;

    if (n > 0){
        a = NN.full[n-1] -> get_output();
    } else {
        a = NN.conv.back() -> get_output();
    }

    vector err = NN.full[n] -> get_error();
    return outer_product(err,a);
}


matrix Backpropagation::output_gradient(){

    vector a;

    if (! NN.full.empty()){
        a = NN.full.back() -> get_output();
    } else {
        a = NN.conv.back() -> get_output();
    }

    vector err = NN.otp -> get_error();
    return outer_product(err,a);
}


vector Backpropagation::get_output_error(){
    return NN.otp -> get_error();
}


vector Backpropagation::get_connected_error(int n){
    return NN.full[n] -> get_error();
}


tensor Backpropagation::get_conv_error(int n){
    return NN.conv[n] -> get_error();
}
*/





/*
 
 
 tensor convolutional_layer::pass_back(tensor X){
 
 err = X.hadamard(act_fn -> gradient(a));
 tensor Y = err.convolve(stride,padding,w);
 return Y;
 }
 
 
 
 tensor convolutional_layer::pass_back(vector X){
 tensor A = tensor::zeros(a.size(index::x), a.size(index::y), a.size(index::z));
 A.set_data(X.return_data());
 return pass_back(A);
 }
 
 
 
 
tensor pooling_layer::pass_back(tensor T){
 
err *= 0.0;
 
 for (int x = 0; x < T.size(index::x); x++){
    int X = pooling_width*x;
    for (int y = 0; y < T.size(index::y); y++){
        int Y = pooling_width*y;
        for (int z = 0; z < T.size(index::z); z++){
 
            if ( ind(X,Y,z) == 1 ){
                err(X,Y,z) = T(x,y,z);
            }
        }
    }
 }
 
 return err;
}
 
 
 
tensor pooling_layer::pass_back(vector X){
 
    tensor A = tensor::zeros(a.size(index::x), a.size(index::y), a.size(index::z));
    A.set_data(X.return_data());
    return pass_back(A);
 }

*/
