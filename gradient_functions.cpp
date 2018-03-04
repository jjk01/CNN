#include "gradient_functions.h"


ConvolutionGradient::ConvolutionGradient(const convolutional_layer * _layer):
HiddenGradient<tensor>(_layer->return_funcType()), layer(_layer){

    err = layer -> pool_index();
}




tensor ConvolutionGradient::pass_back(tensor T){
    
    tensor a = layer -> get_output();
    int stride = layer -> return_stride();
    int padding = layer -> return_padding();
    std::vector<tensor> w = layer -> return_weights();
    
    tensor E = T.hadamard(fn(a));

    if (layer -> pooling()){
        
        int pooling_width = layer -> return_pooling_width();
        tensor ind = layer -> pool_index();
        
        for (int x = 0; x < ind.size(index::x); x++){
            for (int y = 0; y < ind.size(index::y); y++){
                for (int z = 0; z < ind.size(index::z); z++){
                    err(x,y,z) = (ind(x,y,z) == 1 ) ? T(x/pooling_width,y/pooling_width,z):0;
                }
            }
        }
        
    } else {
        err = E;
    }

    tensor Y = err.convolve(stride,padding,w);
    
    return Y;
}



tensor ConvolutionGradient::pass_back(vector X){
    tensor a = layer -> get_output();
    tensor A = tensor::zeros(a.size(index::x), a.size(index::y), a.size(index::z));
    A.set_data(X.return_data());
    return pass_back(A);
}




FullyConnectedGradient::FullyConnectedGradient(const hidden_layer * _layer):
HiddenGradient<vector>(_layer->return_funcType()),layer(_layer){};
    

vector FullyConnectedGradient::pass_back(vector X){
    
    vector a = layer -> get_output();
    matrix w = layer -> get_weight();
    err = X.hadamard(fn(a));
    
    
    vector Y = (w.transpose())*err;
    return Y;
}