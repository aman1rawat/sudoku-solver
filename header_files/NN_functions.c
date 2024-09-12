#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits.h>
#include "matrix_functions.h"


typedef struct{
    int filter_size;
    int stride;
    Matrix *filter;
    Matrix *output;
}Convo_Layer;

typedef struct{
    int filter_size;
    int stride;
    Matrix *output;
}Pooling_Layer;

typedef struct {
    Matrix* data;  
} Input_Layer;

typedef struct {
    Matrix* weights; 
    Matrix* biases;   
    Matrix* output;   
} Hidden_Layer;

typedef struct {
    Matrix* data;    
    Matrix* targets;  
} Output_Layer;

Convo_Layer * createConvoLayer(int filter_size, int stride){
    Convo_Layer *layer = (Convo_Layer*)malloc(sizeof(Convo_Layer));
    layer->filter_size = filter_size;
    layer->stride = stride;
    layer->filter = createMatrix(filter_size, filter_size);
    initializeMatrix(layer->filter);

    return layer;
}

void freeConvoLayer(Convo_Layer *layer){
    if(layer){
        if(layer->filter){
            freeMatrix(layer->filter);
        }
        if(layer->output){
            freeMatrix(layer->output);
        }
        free(layer);
    }
}

void convolution(Convo_Layer *layer, Matrix *input) {
    int output_rows = (input->row - layer->filter_size) / layer->stride + 1;
    int output_cols = (input->col - layer->filter_size) / layer->stride + 1;


    if(!layer->output){
        layer->output = createMatrix(output_rows, output_cols);
        initializeMatrix(layer->output);
    }

    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < layer->filter_size; ki++) {
                for (int kj = 0; kj < layer->filter_size; kj++) {
                    int row = i * layer->stride + ki;
                    int col = j * layer->stride + kj;
                    sum += input->val[row][col] * layer->filter->val[ki][kj];
                }
            }
            layer->output->val[i][j] = sum;
        }
    }
}

Pooling_Layer * createPoolingLayer(int filter_size, int stride){
    Pooling_Layer * layer = (Pooling_Layer*)malloc(sizeof(Pooling_Layer));
    layer->filter_size = filter_size;
    layer->stride = stride;
    return layer;
}

void freePoolingLayer(Pooling_Layer *layer){
    if(layer){
        free(layer);
    }
}

void Pool(Pooling_Layer *pool, Matrix *input) {
    int output_rows = (input->row - pool->filter_size) / pool->stride + 1;
    int output_cols = (input->col - pool->filter_size) / pool->stride + 1;

    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double max_val = INT_MIN;  
            for (int ki = 0; ki < pool->filter_size; ki++) {
                for (int kj = 0; kj < pool->filter_size; kj++) {
                    int row = i * pool->stride + ki;
                    int col = j * pool->stride + kj;
                    if (row < input->row && col < input->col) {
                        double value = input->val[row][col];
                        if (value > max_val) {
                            max_val = value;
                        }
                    }
                }
            }
            pool->output->val[i][j] = max_val;
        }
    }
}

void createInputLayer(Input_Layer * layer){
    layer = (Input_Layer*)malloc(sizeof(Input_Layer));
}

void freeInputLayer(Pooling_Layer *layer){
    if(layer){
        if(layer->output){
            freeMatrix(layer->output);
        }
        free(layer);
    }
}



float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}