#include "output_functions.h"
/*
namespace Output {


    vector softmax_function::implement(const vector& x){

        vector y(x.size());
        double norm = 0;

        for (int k = 0; k < x.size(); k++){
            double otp = std::exp(-x(k));
            y(k) = otp;
            norm += otp;
        }
        return y/norm;
    }

    vector softmax_function::return_error(const vector& a, const vector& y){
        return fn(a,y);
    }


    void softmax_function::set_loss(enum LossType loss){

        switch (loss) {
            case LossType::cross_entropy:
                fn = &softmax_function::cross_entropy;
                break;
            case LossType::quadratic:
                fn = &softmax_function::quadratic;
                break;
        };
    }


    vector softmax_function::cross_entropy(const vector & a, const vector & y){
        return (y-a);
    }


    vector softmax_function::quadratic(const vector & a, const vector & y){
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


    vector sigmoid_function::implement(const vector& x){
        vector y(x.size());

        for (int k = 0; k < x.size(); k++){
            y(k) = 1/(1 + std::exp(-x(k)));
        }
        return y;
    }



    vector sigmoid_function::return_error(const vector& a, const vector& y){
        return fn(a,y);
    }


    void sigmoid_function::set_loss(enum LossType loss){

        switch (loss) {
            case LossType::cross_entropy:
                fn = &sigmoid_function::cross_entropy;
                break;
            case LossType::quadratic:
                fn = &sigmoid_function::quadratic;
                break;
        };
    }


    vector sigmoid_function::quadratic(const vector & a, const vector & y){

        int N = a.size();
        vector z(N);

        for (int k = 0; k < N; k++){
            z(k) = a(k)*(1-a(k))*(a(k)-y(k));
        }
        return z/N;
    }


    vector sigmoid_function::cross_entropy(const vector & a, const vector & y){

        int N = a.size();
        vector z(N);

        for (int k = 0; k < N; k++){
            z(k) = y(k)*(a(k)-1);
        }
        return z;
    }
}*/
