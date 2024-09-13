#include<stdio.h>
#include<stdlib.h>  
#include<stdbool.h>
#include<math.h>
#include<limits.h>
#include "matrix_functions.h"
#include "NN_functions.h"

Matrix* createFilter(int size) {
    Matrix *filter = createMatrix(size, size);
    initializeMatrix(filter);
    return filter;
}

Matrix* forwardConvolution(Matrix *input, Matrix *filter, int stride, bool same_padding) {
    int pad_height = 0;
    int pad_width = 0;

    if (same_padding) {
        pad_height = (filter->row - 1) / 2; // Calculate vertical padding
        pad_width = (filter->col - 1) / 2;  // Calculate horizontal padding
    }

    int output_rows = (input->row + 2 * pad_height - filter->row) / stride + 1;
    int output_cols = (input->col + 2 * pad_width - filter->col) / stride + 1;

    Matrix *output = createMatrix(output_rows, output_cols);

    Matrix *padded_input = (same_padding) ? applyPadding(input, pad_height, pad_width) : copyMatrix(input);

    // Perform convolution
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double sum = 0.0;
            for (int fi = 0; fi < filter->row; fi++) {
                for (int fj = 0; fj < filter->col; fj++) {
                    sum += padded_input->val[i * stride + fi][j * stride + fj] * filter->val[fi][fj];
                }
            }
            output->val[i][j] = sum;
        }
    }

    freeMatrix(padded_input);

    return output;
}

// Pooling Layer Functions
Matrix* forwardPooling(Matrix *input, int filter_size, int stride) {
    int output_rows = (input->row - filter_size) / stride + 1;
    int output_cols = (input->col - filter_size) / stride + 1;
    Matrix *output = createMatrix(output_rows, output_cols);

    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double max_val = INT_MIN;
            for (int fi = 0; fi < filter_size; fi++) {
                for (int fj = 0; fj < filter_size; fj++) {
                    double current_val = input->val[i * stride + fi][j * stride + fj];
                    if (current_val > max_val) {
                        max_val = current_val;
                    }
                }
            }
            output->val[i][j] = max_val;
        }
    }
    return output;
}

// Fully Connected Layer Functions
Matrix* forwardFullyConnected(Matrix *input, Matrix *weights, Matrix *biases) {
    Matrix *output = dot(weights, input);
    output = add(output, biases);
    return output;
}

Matrix* applyPadding(Matrix *input, int pad_height, int pad_width) {
    // Create a new matrix with padded dimensions
    Matrix *padded_matrix = createMatrix(input->row + 2 * pad_height, input->col + 2 * pad_width);
    
    // Initialize padded matrix with zeros
    for (int i = 0; i < padded_matrix->row; i++) {
        for (int j = 0; j < padded_matrix->col; j++) {
            if (i < pad_height || i >= padded_matrix->row - pad_height || 
                j < pad_width || j >= padded_matrix->col - pad_width) {
                padded_matrix->val[i][j] = 0; // Set padding to zero
            } else {
                padded_matrix->val[i][j] = input->val[i - pad_height][j - pad_width];
            }
        }
    }
    
    return padded_matrix;
}

// Activation Functions
Matrix* applyReLU(Matrix *input) {
    Matrix *output = copyMatrix(input);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            output->val[i][j] = fmax(0, output->val[i][j]);
        }
    }
    return output;
}

Matrix* applySigmoid(Matrix *input) {
    Matrix *output = copyMatrix(input);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            output->val[i][j] = 1 / (1 + exp(-output->val[i][j]));
        }
    }
    return output;
}

Matrix* applySoftmax(Matrix *input) {
    Matrix *output = copyMatrix(input);
    double sum = 0.0;
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            sum += exp(output->val[i][j]);
        }
    }
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            output->val[i][j] = exp(output->val[i][j]) / sum;
        }
    }
    return output;
}

// Loss Functions
double calculateMeanSquaredError(Matrix *output, Matrix *targets) {
    double error = 0.0;
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            double diff = output->val[i][j] - targets->val[i][j];
            error += diff * diff;
        }
    }
    return error / (output->row * output->col);
}

double calculateCrossEntropyLoss(Matrix *output, Matrix *targets) {
    double loss = 0.0;
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            loss += targets->val[i][j] * log(fmax(1e-10, output->val[i][j]));
        }
    }
    return -loss / (output->row * output->col);
}

