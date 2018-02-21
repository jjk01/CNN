#include "parameters.h"


static parameters zeros(const param_sizes& arg){

    for (auto w = arg.W_conv.begin(); w != arg.W_conv.end(); w++){
        std::vector<tensor> X;
        tensor X0 = tensor::zeros(w -> n1, w -> n2, w -> n3);
        for (int m = 0; m < arg.K_conv.size() ; m++){
            X.push_back(X0);
        }
        dW_conv.push_back(X);
    }

    for (auto b = arg.B_conv.begin(); b != arg.B_conv.end(); b++){
        dB_conv.push_back(tensor::zeros(b -> n1, b -> n2, b -> n3));
    }

    for (auto w = arg.W_full.begin(); w != arg.W_full.end(); w++){
        dW_full.push_back(matrix::zeros(w -> n1, w -> n2));
    }

    for (auto b = arg.B_full.begin(); b != arg.B_full.end(); w++){
        dB_full.push_back(vector(matrix::zeros(*b,1)));
    }

    dW_otp = matrix::zeros(W_otp.n1,W_otp.n2);
    vector dB_otp = vector(matrix::zeros(B_otp,1));
}


parameters parameters::operator+=(const parameters& arg) {

    for (int n = 0; n < arg.dW_conv.size(); n++){
        for (int m = 0; m < arg.dW_conv[n].size() ; m++){
            dW_conv[n][m] += arg.dW_conv[n][m]
        }
    }

    for (int n = 0; n < arg.dB_conv.size(); n++){
        dB_conv[n] += arg.dB_conv[n];
    }

    for (int n = 0; n < arg.dW_full.size(); n++){
        dW_full[n] += arg.dW_full[n];
    }

    for (int n = 0; n < arg.dB_full.size(); n++){
        dB_full[n] += arg.dB_full[n];
    }

    dW_otp += arg.dW_otp;
    dB_otp += arg.dB_otp;

    return *this;
}



parameters parameters::operator*=(const double& f) {

    for (int n = 0; n < arg.dW_conv.size(); n++){
        for (int m = 0; m < arg.dW_conv[n].size() ; m++){
            dW_conv[n][m] *= f;
        }
    }

    for (int n = 0; n < arg.dB_conv.size(); n++){
        dB_conv[n] *= f;
    }

    for (int n = 0; n < arg.dW_full.size(); n++){
        dW_full[n] *= f;
    }

    for (int n = 0; n < arg.dB_full.size(); n++){
        dB_full[n] *= f;
    }

    dW_otp *= f;
    dB_otp *= f;

    return *this;
}


parameters parameters::operator/=(const double& f) {

    for (int n = 0; n < arg.dW_conv.size(); n++){
        for (int m = 0; m < arg.dW_conv[n].size() ; m++){
            dW_conv[n][m] /= f;
        }
    }

    for (int n = 0; n < arg.dB_conv.size(); n++){
        dB_conv[n] /= f;
    }

    for (int n = 0; n < arg.dW_full.size(); n++){
        dW_full[n] /= f;
    }

    for (int n = 0; n < arg.dB_full.size(); n++){
        dB_full[n] /= f;
    }

    dW_otp /= f;
    dB_otp /= f;

    return *this;
}


parameters operator+(parameters& l, parameters& r) {

    parameters X{l};
    X += r;
    return X;
}


parameters operator-(parameters& l, parameters& r) {

    parameters X{r};
    X *= -1.0;
    X += l;
    return X;
}



parameters operator*(double m, parameters& r) {

    parameters X{r};
    X *= m;
    return X;
}


parameters operator/(parameters& r, double d) {

    parameters X{r};
    X /= d;
    return X;
}
