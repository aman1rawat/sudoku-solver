#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H
#include<math.h>

void initializeMatrix(float **m, int row, int col); //initialize matrix with random values

float * flattenMatrix(float **m, int row, int col); // flatten 2D matrix into a 1D array



#endif