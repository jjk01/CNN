#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include <stdio.h>
#include "neural_net.h"

struct parameters {

    static parameters zeros(const dimensions&);

    parameters operator+=(const parameters&);
    parameters operator*=(const double&);
    parameters operator/=(const double&);

    std::vector<std::vector<tensor>> W0;
    std::vector<tensor> B0;
    std::vector<matrix> W1;
    std::vector<vector> B1;
    matrix W2;
    vector B2;
};

parameters operator+(parameters&, parameters&);
parameters operator-(parameters&, parameters&);
parameters operator*(double, parameters&);
parameters operator/(parameters&, double);



#endif /* GRADIENT_HPP */
