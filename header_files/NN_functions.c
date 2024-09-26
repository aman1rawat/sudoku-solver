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
    Matrix *output = copyMatrix(input); // Create a copy of the input matrix
    double max_val = output->val[0][0]; // Initialize max_val with the first element

    // Find the maximum value in the input for numerical stability
    for (int i = 0; i < output->row; i++) {
        if (output->val[i][0] > max_val) {
            max_val = output->val[i][0];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < output->row; i++) {
        output->val[i][0] = (double)exp(output->val[i][0] - max_val); // Subtract max_val for stability
        sum += output->val[i][0]; // Accumulate the sum of exponentials
    }

    for (int i = 0; i < output->row; i++) {
        output->val[i][0] /= sum; // Divide by the sum to normalize
    }
    return output; // Return the softmax probabilities
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


Matrix** loadInputMatrices(char* file_string, int batch_size, int matrix_size) {
    FILE* file = fopen(file_string, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    char line[256];
    int count = 0;

    // Allocate memory for the input matrices (array of pointers)
    Matrix **input_matrices = (Matrix **)malloc(batch_size * sizeof(Matrix*));

    while (count < batch_size && fgets(line, sizeof(line), file)) {
        // Find the comma separating scramble and solution
        char* comma_pos = strchr(line, ',');
        if (comma_pos == NULL) {
            fprintf(stderr, "Invalid file format: no comma found\n");
            fclose(file);
            exit(1);
        }

        // Replace the comma with a null terminator to isolate the scramble
        *comma_pos = '\0';
        char* scramble_data = line; // This is now the scramble part

        // Create a matrix for the scramble
        input_matrices[count] = createMatrix(matrix_size, matrix_size);

        // Fill the scramble matrix
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                input_matrices[count]->val[i][j] = scramble_data[i * matrix_size + j] - '0';
            }
        }

        count++; // Increment the count of loaded samples
    }

    fclose(file);
    printf("Successfully loaded %d input matrices from %s\n", count, file_string);
    return input_matrices; // Return the input matrices
}


Matrix** loadOutputMatrices(char* file_string, int batch_size, int matrix_size) {
    FILE* file = fopen(file_string, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    char line[256];
    int count = 0;

    // Allocate memory for the output matrices (array of pointers)
    Matrix **output_matrices = (Matrix **)malloc(batch_size * sizeof(Matrix*));

    while (count < batch_size && fgets(line, sizeof(line), file)) {
        // Find the comma separating scramble and solution
        char* comma_pos = strchr(line, ',');
        if (comma_pos == NULL) {
            fprintf(stderr, "Invalid file format: no comma found\n");
            fclose(file);
            exit(1);
        }

        // Move the pointer to the beginning of the solution part
        char* solution_data = comma_pos + 1; // This is the solution part

        // Create a matrix for the solution
        output_matrices[count] = createMatrix(matrix_size, matrix_size);

        // Fill the solution matrix
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                output_matrices[count]->val[i][j] = solution_data[i * matrix_size + j] - '0';
            }
        }

        count++; // Increment the count of loaded samples
    }

    fclose(file);
    printf("Successfully loaded %d output matrices from %s\n", count, file_string);
    return output_matrices; // Return the output matrices
}

// Gradient calculation for ReLU activation function
Matrix* gradientReLU(Matrix *input) {
    Matrix *gradient = createMatrix(input->row, input->col);
    for (int i = 0; i < input->row; i++) {
        for (int j = 0; j < input->col; j++) {
            gradient->val[i][j] = input->val[i][j] > 0 ? 1 : 0; // Derivative of ReLU
        }
    }
    return gradient;
}

// Gradient calculation for Sigmoid activation function
Matrix* gradientSigmoid(Matrix *input) {
    Matrix *gradient = createMatrix(input->row, input->col);
    for (int i = 0; i < input->row; i++) {
        for (int j = 0; j < input->col; j++) {
            double sig = 1 / (1 + exp(-input->val[i][j]));
            gradient->val[i][j] = sig * (1 - sig); // Derivative of Sigmoid
        }
    }
    return gradient;
}

// Gradient calculation for Softmax activation function
Matrix* gradientSoftmax(Matrix *output) {
    Matrix *gradient = createMatrix(output->row, output->col);
    for (int i = 0; i < output->row; i++) {
        for (int j = 0; j < output->col; j++) {
            double softmax_val = output->val[i][j];
            gradient->val[i][j] = softmax_val * (1 - softmax_val); // Derivative of Softmax
        }
    }
    return gradient;
}

void backpropagateFullyConnected(Matrix *input, Matrix *weights, Matrix *biases, Matrix *output, Matrix *targets, double learning_rate) {
    // Calculate the loss gradient
    Matrix *loss_gradient = subtract(output, targets); // dL/dY

    // Calculate gradients for weights and biases
    Matrix *input_transpose = transpose(input); // Transpose of input for weight gradient
    Matrix *weight_gradient = dot(loss_gradient, input_transpose); // dL/dW
    Matrix *bias_gradient = copyMatrix(loss_gradient); // dL/db

    // Update weights and biases
    for (int i = 0; i < weights->row; i++) {
        for (int j = 0; j < weights->col; j++) {
            weights->val[i][j] -= learning_rate * weight_gradient->val[i][j];
        }
    }
    for (int i = 0; i < biases->row; i++) {
        biases->val[i][0] -= learning_rate * bias_gradient->val[i][0];
    }

    // Free allocated matrices
    freeMatrix(loss_gradient);
    freeMatrix(input_transpose);
    freeMatrix(weight_gradient);
    freeMatrix(bias_gradient);
}

void backpropagateConvolution(Matrix *input, Matrix *filter, Matrix *output, Matrix *dL_dY, int stride, bool same_padding, double learning_rate) {
    int filter_height = filter->row;
    int filter_width = filter->col;
    int output_height = output->row;
    int output_width = output->col;

    // Initialize gradient for the filter
    Matrix *filter_gradient = createMatrix(filter_height, filter_width);

    // Iterate over the output dimensions to compute filter gradients
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            for (int fi = 0; fi < filter_height; fi++) {
                for (int fj = 0; fj < filter_width; fj++) {
                    // Calculate the position in the input matrix
                    int input_i = i * stride + fi;
                    int input_j = j * stride + fj;

                    // Update the filter gradient
                    filter_gradient->val[fi][fj] += dL_dY->val[i][j] * input->val[input_i][input_j];
                }
            }
        }
    }

    // Update the filter weights
    for (int fi = 0; fi < filter_height; fi++) {
        for (int fj = 0; fj < filter_width; fj++) {
            filter->val[fi][fj] -= learning_rate * filter_gradient->val[fi][fj];
        }
    }

    // Free allocated matrices
    freeMatrix(filter_gradient);
}

Matrix* resize(Matrix* inputMatrix, int newRow, int newCol) {
    // Check if the input matrix is of size 81x1
    if (inputMatrix->row != 81 || inputMatrix->col != 1) {
        printf("Input matrix must be of size 81x1.\n");
        exit(1);
    }

    // Create a new matrix with specified dimensions (9x9)
    Matrix* outputMatrix = createMatrix(newRow, newCol);
    
    // Fill the output matrix with values from the input matrix
    for (int i = 0; i < newRow; i++) {
        for (int j = 0; j < newCol; j++) {
            outputMatrix->val[i][j] = inputMatrix->val[i * newCol + j][0];
        }
    }

    return outputMatrix; // Return the resized output matrix
}

Matrix* argmax(Matrix *input) {
    if (input == NULL || input->row != 729 || input->col != 1) {
        return NULL; // Return NULL for invalid input
    }

    // Create output matrix with 81 rows and 1 column
    Matrix *output = malloc(sizeof(Matrix));
    output->row = 81;
    output->col = 1;
    output->val = malloc(output->row * sizeof(double*));
    for (int i = 0; i < output->row; i++) {
        output->val[i] = malloc(sizeof(double));
    }

    // Iterate through the input matrix in steps of 9
    for (int i = 0; i < 81; i++) {
        double max_value = input->val[i * 9][0]; // Start with the first value in the group
        int max_index = 0; // Initialize index of max value

        // Check the next 8 values
        for (int j = 1; j < 9; j++) {
            if (input->val[i * 9 + j][0] > max_value) {
                max_value = input->val[i * 9 + j][0]; // Update max value
                max_index = j; // Update index of max value
            }
        }

        // Store the index of the maximum value (1-9)
        output->val[i][0] = max_index + 1; // Store as numbers from 1 to 9
    }

    return output; // Return the output matrix
}


float checkAccuracy(Matrix* output, Matrix *act_output){
    if(output->row!=act_output->row || output->col!=act_output->col){
        printf("Dimension mismatch!\n");
        return 0.0;
    }
    float acc=0.0;
    int correct=0;
    for(int i=0;i<output->row;i++){
        if(output->val[i][0]==act_output->val[i][0]) correct++;
    }

    acc = (float)(correct/81)*100.0;

    return acc;
}