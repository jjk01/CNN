#include "activation_functions.h"



OutputFunction::OutputFunction(OutputType _fn){
    switch (_fn){
        case OutputType::sigmoid:
            this->fn = &sigmoid_function;
            fn_type = OutputType::sigmoid;
            break;
        case OutputType::softmax:
            this->fn = &softmax_function;
            fn_type = OutputType::softmax;
            break;
    }
}


OutputType OutputFunction::return_funcType() const {
    return fn_type;
}