#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include "matrix_functions.h"
#include<stdbool.h>

Matrix* createFilter(int size);

Matrix* forwardConvolution(Matrix *input, Matrix *filter, int stride, bool same_padding);

// Pooling Layer Functions
Matrix* forwardPooling(Matrix *input, int filter_size, int stride);

// Fully Connected Layer Functions
Matrix* forwardFullyConnected(Matrix *input, Matrix *weights, Matrix *biases);

Matrix* applyPadding(Matrix *input, int pad_height, int pad_width);

// Activation Functions
Matrix* applyReLU(Matrix *input);

Matrix* applySigmoid(Matrix *input);

Matrix* applySoftmax(Matrix *input);

// Loss Functions
double calculateMeanSquaredError(Matrix *output, Matrix *targets);

double calculateCrossEntropyLoss(Matrix *output, Matrix *targets);
// Training Functions


#endif