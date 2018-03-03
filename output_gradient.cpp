#include "output_gradient.h"


OutputGradient::OutputGradient(OutputType _fn, LossType _loss): loss(_loss), fn_type(_fn){
    
    if (_loss == LossType::quadratic){
        
        switch (_fn){
            case OutputType::sigmoid:
                fn = &quadratic_sigmoid;
                fn_type = OutputType::sigmoid;
                break;
            case OutputType::softmax:
                fn = &quadratic_softmax;
                fn_type = OutputType::softmax;
                break;
        }
        
    } else if (_loss == LossType::cross_entropy) {
        
        switch (_fn){
            case OutputType::sigmoid:
                fn = &cross_entropy_sigmoid;
                fn_type = OutputType::sigmoid;
                break;
            case OutputType::softmax:
                fn = &cross_entropy_softmax;
                fn_type = OutputType::softmax;
                break;
        }
        
    }
}



OutputType OutputGradient::return_funcType() const{
    return fn_type;
}


LossType OutputGradient::return_lossType() const{
    return loss;
}


vector OutputGradient::quadratic_softmax(const vector & a, const vector & y){
    int N = a.size();
    vector err(N);
    for (int m = 0; m < N; m++){
        double S = 0;
        for (int n = 0; n < N; n++){
            S += a(n)*(a(n) - y(n));
        }
        err(m) = a(m)*(S - a(m)*(a(m)-y(m)));
    }
    return err/N;
}

vector OutputGradient::quadratic_sigmoid(const vector & a, const vector & y){
    
    int N = a.size();
    vector z(N);
    
    for (int k = 0; k < N; k++){
        z(k) = a(k)*(1-a(k))*(a(k)-y(k));
    }
    return z/N;
}


vector OutputGradient::cross_entropy_sigmoid(const vector & a, const vector & y){
    
    int N = a.size();
    vector z(N);
    
    for (int k = 0; k < N; k++){
        z(k) = y(k)*(a(k)-1);
    }
    return z;
}


vector OutputGradient::cross_entropy_softmax(const vector & a, const vector & y){
    return (y-a);
}

