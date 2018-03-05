#include <stdio.h>
#include <algorithm>
#include "training_algorithm.h"



TrainingStatergy::TrainingStatergy(neural_net * _NN, const training_data & _data, LossType loss):
data(_data), NN(_NN){
    method.reset(new Gradient_Descent(NN,loss));
}


void TrainingStatergy::train(){

    for (int n = 0; n < max_epochs; ++n){
        std::vector<int> indices = generate_batch();
        increment_epoch(indices);
        if (!print_progress){
            std::cout << "epoch = " << n+1 << ", error = " << err << "\n";
        }
    }
}

void TrainingStatergy::increment_epoch(std::vector<int> indices){
    
    err *= 0;
    net_parameters prev = method -> get_parameters();
    method -> zero_parameters();
    for (int n = 0; n < indices.size(); ++n){
        vector a = NN -> action(data[n].first);
        method -> iterate(data[n].second);
        err += (a - data[n].second).norm();
    }
    
    err /= indices.size();
    
    net_parameters dP = method -> get_parameters();
    dP += p*prev;
    dP /= indices.size();
    dP *= -r;
    
    NN -> update(dP);
}




std::vector<int> TrainingStatergy::generate_batch(){
    
    std::size_t data_size = data.size();
    
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
    
    std::vector<int> indices;
    
    for (auto itr = index_set.begin(); itr != index_set.end(); ++itr){
        indices.push_back(*itr);
    }
    
    return indices;
}



TrainingMethod::TrainingMethod(neural_net * _NN, LossType _loss): loss(_loss), params(_NN){}



net_parameters TrainingMethod::get_parameters(){
    return params;
}



void TrainingMethod::zero_parameters(){
    params.zero();
}




Gradient_Descent::Gradient_Descent(neural_net * _NN, LossType _loss): TrainingMethod(_NN,_loss), inpt(_NN -> input_ptr()) {

    const std::vector<convolutional_layer> * conv_ptr = _NN -> convolution_ptr();
    const std::vector<hidden_layer> * full_ptr = _NN -> full_ptr();
    const output_layer * otp_ptr = _NN -> output_ptr();
    
    for (auto itr = conv_ptr -> begin(); itr != conv_ptr -> end(); ++itr){
        conv_grad.push_back(ConvolutionGradient(itr -> get_pointer()));
    }
    
    for (auto itr = full_ptr -> begin(); itr != full_ptr -> end(); ++itr){
        full_grad.push_back(FullyConnectedGradient(itr -> get_pointer()));
    }
    
    otp_grad.reset(new OutputGradient(loss,otp_ptr));

}


void Gradient_Descent::iterate(vector y){
    backpropagate(y);
    add_params();
}



/* Propagating the error back through the net. */
    
void Gradient_Descent::backpropagate(vector y){
    
    vector e1 = otp_grad -> pass_back(y);
    
    for (auto rit = full_grad.rbegin(); rit != full_grad.rend(); rit++){
        e1 = rit -> pass_back(e1);
    }
    
    for (auto rit = conv_grad.rbegin(); rit != conv_grad.rend(); rit++){
        e1 = rit -> pass_back(e1);
    }
    
}


/* This function uses the errors calculated in the previous backprop run to determine dC/dW and dC/B for a particular pass. Theses are then added to total parameter change. */

void Gradient_Descent::add_params(){
    
    tensor a = inpt -> get_output();
    
    for (auto itr = conv_grad.begin(); itr != conv_grad.end(); ++itr){
        
        const convolutional_layer * ptr = itr -> return_ptr();
        tensor err = itr -> get_error();
        int S = ptr -> return_stride();
        int P = ptr -> return_padding();
        
        conv_parameters dP(a.correlation(S,S,P,P,err),err);
        params.conv[std::distance(conv_grad.begin(),itr)] += dP;
        a = ptr -> get_output();
    }

    
    vector b{a};
    
    for (auto itr = full_grad.begin(); itr != full_grad.end(); ++itr){
        
        const hidden_layer * ptr = itr -> return_ptr();
        vector err = itr -> get_error();
        
        full_parameters dP(outer_product(err,b),err);
        params.full[std::distance(full_grad.begin(),itr)] += dP;
        b = ptr -> get_output();
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
