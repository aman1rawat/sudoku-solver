#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

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

Convo_Layer * createConvoLayer(int filter_size, int stride);

void freeConvoLayer(Convo_Layer *layer);

void convolution(Convo_Layer *layer, Matrix *input) ;

Pooling_Layer * createPoolingLayer(int filter_size, int );

void freePoolingLayer(Pooling_Layer *layer);

void Pool(Pooling_Layer *pool, Matrix *input);

void createInputLayer(Input_Layer * layer);

void freeInputLayer(Pooling_Layer *layer);

float sigmoid(float x);

float sigmoid_derivative(float x);

#endif