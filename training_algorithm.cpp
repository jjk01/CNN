#include <stdio.h>
#include <algorithm>
#include "training_algorithm.h"



TrainingMethod::TrainingMethod(neural_net * _NN, LossType _loss): NN(_NN), loss(_loss), params(_NN){}



void TrainingMethod::epoch_increment(training_data data){
    
    std::size_t data_size = data.size();
    for (auto itr = data.begin(); itr != data.end(); ++itr){
        iterate(itr->first, itr->second);
    }
    params /= data_size;
}




net_parameters TrainingMethod::get_parameters(){
    return params;
}



void TrainingMethod::zero_parameters(){
    params.zero();
}




Gradient_Descent::Gradient_Descent(neural_net * _NN, LossType _loss): TrainingMethod(_NN,_loss) {
    
    const std::vector<convolutional_layer> * conv_ptr = NN -> convolution_ptr();
    const std::vector<hidden_layer> * full_ptr = NN -> full_ptr();
    const output_layer * otp_ptr = NN -> output_ptr();
    
    for (auto itr = conv_ptr -> begin(); itr != conv_ptr -> end(); ++itr){
        conv_grad.push_back(ConvolutionGradient(itr -> get_pointer()));
    }
    
    for (auto itr = full_ptr -> begin(); itr != full_ptr -> end(); ++itr){
        full_grad.push_back(FullyConnectedGradient(itr -> get_pointer()));
    }
    
    otp_grad.reset(new OutputGradient(loss,otp_ptr));

}


void Gradient_Descent::iterate(tensor x, vector y){
    backpropagate(x,y);
    add_params();
}



/* Propagating the error back through the net. */
    
void Gradient_Descent::backpropagate(tensor x, vector y){

    vector a = NN -> action(x);
    vector e1 = otp_grad -> pass_back(a,y);
    
    for (auto rit = full_grad.rbegin(); rit != full_grad.rend(); rit++){
        e1 = rit -> pass_back(e1);
    }
    
    if (!conv_grad.empty()){
        tensor e2 = conv_grad.back().pass_back(e1);
        
        for (auto rit = conv_grad.rbegin()+1; rit != conv_grad.rend(); rit++){
            e2 = rit -> pass_back(e2);
        }
    }
    
}


/* This function uses the errors calculated in the previous backprop run to determine dC/dW and dC/B for a particular pass. Theses are then added to total parameter change. */

void Gradient_Descent::add_params(){
    
    const input_layer * inpt_ptr = NN -> input_ptr();
    const std::vector<convolutional_layer> * conv_ptr = NN -> convolution_ptr();
    const std::vector<hidden_layer> * full_ptr = NN -> full_ptr();
    
    tensor a = inpt_ptr -> get_output();

    
    for (int n = 0; n < conv_ptr -> size(); ++n){
        tensor err = conv_grad[n].get_error();
        int S = (*conv_ptr)[n].return_stride();
        int P = (*conv_ptr)[n].return_padding();
        
        conv_parameters dP(a.correlation(S,S,P,P,err),err);
        params.conv[n] += dP;
        a = (*conv_ptr)[n].get_output();
        
    }
    
    vector b{a};
    
    for (int n = 0; n < full_ptr -> size(); ++n){
        vector err = full_grad[n].get_error();
        full_parameters dP(outer_product(err,b),err);
        params.full[n] += dP;
        b = (*full_ptr)[n].get_output();
    }
    
    vector err = otp_grad -> get_error();
    full_parameters dP(outer_product(err,b),err);
    params.otp += dP;
}


/*




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










*/
