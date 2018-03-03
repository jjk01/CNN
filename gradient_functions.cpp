#include "gradient_functions.h"


ConvolutionGradient::ConvolutionGradient(const convolutional_layer * _layer):
HiddenGradient<tensor>(_layer->return_funcType()), layer(_layer){};




tensor ConvolutionGradient::pass_back(tensor X){
    
    tensor a = layer -> ouput();
    int stride = layer -> return_stride();
    int padding = layer -> return_padding();
    std::vector<tensor> w = layer -> return_weights();
    
    tensor E;

    if (layer -> pooling()){
        
        int pooling_width = layer -> return_pooling_width();
        tensor ind = layer -> pool_index();
        
        E = tensor::zeros(pooling_width*X.size(index::x), pooling_width*X.size(index::y), X.size(index::z));
        
        for (int x = 0; x < X.size(index::x); x++){
            int A = pooling_width*x;
            for (int y = 0; y < X.size(index::y); y++){
                int B = pooling_width*y;
                for (int z = 0; z < X.size(index::z); z++){
                    err(A,B,z) = (ind(A,B,z) == 1 ) ? X(x,y,z):0;
                }
            }
        }
        
    } else {
        E = X;
    }
    
    err = E.hadamard(fn(a));
    tensor Y = err.convolve(stride,padding,w);
    
    return Y;
}



tensor ConvolutionGradient::pass_back(vector X){
    tensor a = layer -> ouput();
    tensor A = tensor::zeros(a.size(index::x), a.size(index::y), a.size(index::z));
    A.set_data(X.return_data());
    return pass_back(A);
}





FullyConnectedGradient::FullyConnectedGradient(const hidden_layer * _layer):
    HiddenGradient<vector>(_layer->return_funcType()), layer(_layer){};
    

vector FullyConnectedGradient::pass_back(vector X){
    
    vector a = layer -> get_output();
    matrix w = layer -> get_weight();
    err = X.hadamard(fn(a));
    
    
    vector Y = (w.transpose())*err;
    return Y;
}