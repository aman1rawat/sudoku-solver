#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "header_files/NN_functions.h"

#define learning_rate  0.01
#define dataset_size 1000000
#define batch_size 1
#define same_padding true
#define convo_filter_size 3
#define hidden_layer1_size 25
#define hidden_layer2_size 200
#define output_layer_size 729

Matrix ** input_sample = NULL;
Matrix ** output_sample = NULL;
Matrix *convo1_filter = NULL;
Matrix *convo2_filter = NULL;
Matrix *convo3_filter = NULL;

Matrix *pooling_layer_output = NULL;


Matrix *hidden_layer1 = NULL;
Matrix *hidden_layer1_weights = NULL;
Matrix *hidden_layer1_bias = NULL;

Matrix *hidden_layer2 = NULL;
Matrix *hidden_layer2_weights = NULL;
Matrix *hidden_layer2_bias = NULL;

Matrix *output_layer = NULL;
Matrix *output_layer_bias = NULL;
Matrix *output_layer_weights = NULL;

int main(){
	if(!input_sample || !output_sample){
		input_sample = loadInputMatrices("D:\\Datasets\\sudoku scrambles\\chunk_1.csv", batch_size, 9);
		output_sample = loadOutputMatrices("D:\\Datasets\\sudoku scrambles\\chunk_1.csv",  batch_size, 9);
	}

	for(int epoch=0;epoch<batch_size;epoch++){
		if(!convo1_filter) convo1_filter =  createFilter(convo_filter_size);
		printf("convo1 filter created\n");
		printMatrix(convo1_filter);
		if(!convo2_filter) convo2_filter =  createFilter(convo_filter_size);
		printf("\nconvo2 filter created\n");
		printMatrix(convo2_filter);
		if(!convo3_filter) convo3_filter =  createFilter(convo_filter_size);
		printf("\nconvo3 filter created\n");
		printMatrix(convo3_filter);

		Matrix *convo1_output = forwardConvolution(input_sample[epoch], convo1_filter, 1, same_padding);
		printf("\nforward convo 1 done\n");
		printMatrix(convo1_output);

		Matrix *convo2_output = forwardConvolution(convo1_output, convo2_filter, 1, same_padding);
		printf("\nforward convo 2 done\n");
		printMatrix(convo2_output);
		Matrix *convo3_output = forwardConvolution(convo2_output, convo3_filter, 1, same_padding);
		printf("\nforward convo 3 done\n");
		printMatrix(convo3_output);

		printf("\n");

		convo3_output = applyPadding(convo3_output, 1, 1);
		printf("Padding applied \n");
		printMatrix(convo3_output);

		pooling_layer_output = forwardPooling(convo3_output, 2, 2);
		printf("\npooling done\n");
		printMatrix(pooling_layer_output);
		printf("\n");

		pooling_layer_output = flattenMatrix(pooling_layer_output);
		printf("flattening done\n");
		printMatrix(pooling_layer_output);
		printf("\n");



		hidden_layer1 = createMatrix(hidden_layer1_size, 1);
		printf("hidden layer 1 created\n");
		
		if(!hidden_layer1_weights){
			hidden_layer1_weights = createMatrix(hidden_layer1_size, pooling_layer_output->row);
			initializeMatrix(hidden_layer1_weights);
			printf("\nhidden layer 1 weights created\n");
			printMatrix(hidden_layer1_weights);
		}
		if(!hidden_layer1_bias){
			hidden_layer1_bias = createMatrix(hidden_layer1_size, 1);
			initializeMatrix(hidden_layer1_bias);
			printf("\nhidden layer 1 bias created\n");
			printMatrix(hidden_layer1_bias);
		}
		hidden_layer1 =  forwardFullyConnected(pooling_layer_output, hidden_layer1_weights, hidden_layer1_bias);
		printf("forward propagation 1 done\n");
		hidden_layer1 = applySigmoid(hidden_layer1);
		printf("\nactivation 1 done\n");
		printMatrix(hidden_layer1);



		hidden_layer2 = createMatrix(hidden_layer2_size, 1);
		printf("hidden layer 2 created\n");	
		printf("Row : %d    Columns : %d\n", hidden_layer2->row, hidden_layer2->col);

		if(!hidden_layer2_weights){
			hidden_layer2_weights = createMatrix(hidden_layer2_size, hidden_layer1_size);
			initializeMatrix(hidden_layer2_weights);
			printf("\nhidden layer 2 weights created\n");
			printMatrix(hidden_layer2_weights);
		}
		printf("\n");
		if(!hidden_layer2_bias){
			hidden_layer2_bias = createMatrix(hidden_layer2_size, 1);
			initializeMatrix(hidden_layer2_bias);
			printf("hidden layer 2 bias created\n");
			printMatrix(hidden_layer2_bias);
		}
		printf("\n");

		hidden_layer2 =  forwardFullyConnected(hidden_layer1, hidden_layer2_weights, hidden_layer2_bias);
		printf("forward propagation 2 done\n");
		hidden_layer2 = applySigmoid(hidden_layer2);
		printf("activation 2 done\n");
		printMatrix(hidden_layer2);


		output_layer = createMatrix(output_layer_size, 1);
		printf("output layer created\n");	
		printf("Row : %d    Columns : %d\n", output_layer->row, output_layer->col);

		if(!output_layer_weights){
			output_layer_weights = createMatrix(output_layer_size, hidden_layer2_size);
			initializeMatrix(output_layer_weights);
			printf("\noutput layer weights created\n");
			//printMatrix(output_layer_weights);
		}
		printf("\n");
		if(!output_layer_bias){
			output_layer_bias = createMatrix(output_layer_size, 1);
			initializeMatrix(output_layer_bias);
			printf("output layer bias created\n");
			printMatrix(output_layer_bias);
		}
		printf("\n");
		output_layer = forwardFullyConnected(hidden_layer2, output_layer_weights, output_layer_bias);
		printf("output layer forward propagation done\n");
		output_layer = applySoftmax(output_layer);
		printf("softmax done\n");
        printMatrix(output_layer);

		Matrix *final_matrix = argmax(output_layer);
		final_matrix = resize(final_matrix, 9, 9);
		printf("Final Matrix : \n");
		printMatrix(final_matrix);

		printf("\n\nOutput matrix for comparison :\n");
		printMatrix(output_sample[epoch]);

		printf("Accuracy = %.4f\n", checkAccuracy(final_matrix, output_sample[epoch]));
		printf("-------------------------------------------------------------------------------------------------");
	}
	return 0;
}