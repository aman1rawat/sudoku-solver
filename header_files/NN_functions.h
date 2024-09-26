#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include "matrix_functions.h"
#include<stdbool.h>
#include<string.h>

Matrix* createFilter(int size);

Matrix* forwardConvolution(Matrix *input, Matrix *filter, int stride, bool same_padding);
Matrix* forwardPooling(Matrix *input, int filter_size, int stride);
Matrix* forwardFullyConnected(Matrix *input, Matrix *weights, Matrix *biases);
Matrix* applyPadding(Matrix *input, int pad_height, int pad_width);
Matrix* applyReLU(Matrix *input);

Matrix* applySigmoid(Matrix *input);
Matrix* applySoftmax(Matrix *input);
double calculateMeanSquaredError(Matrix *output, Matrix *targets);
double calculateCrossEntropyLoss(Matrix *output, Matrix *targets);

Matrix** loadInputMatrices(char* file_string, int batch_size, int matrix_size);
Matrix** loadOutputMatrices(char* file_string, int batch_size, int matrix_size);
Matrix* gradientReLU(Matrix *input);
Matrix* gradientSigmoid(Matrix *input);
Matrix* gradientSoftmax(Matrix *output);
void backpropagateFullyConnected(Matrix *input, Matrix *weights, Matrix *biases, Matrix *output, Matrix *targets, double learning_rate);
void backpropagateConvolution(Matrix *input, Matrix *filter, Matrix *output, Matrix *dL_dY, int stride, bool same_padding, double learning_rate);
Matrix* argmax(Matrix *input);
Matrix* resize(Matrix* inputMatrix, int newRow, int newCol);
float checkAccuracy(Matrix* output, Matrix *act_output);

#endif