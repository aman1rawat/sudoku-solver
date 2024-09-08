#include<math.h>

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}