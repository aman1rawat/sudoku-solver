#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H
#include<math.h>

typedef struct{
	double ** val;
	int row, col;
}Matrix;

Matrix * createMatrix(int row, int col);
Matrix * initializeMatrix(int row, int col); //initialize matrix with random values
void fillMatrix(Matrix *m, int n);
void freeMatrix(Matrix *m);
void printMatrix(Matrix* m);
Matrix* copyMatrix(Matrix* m);


#endif