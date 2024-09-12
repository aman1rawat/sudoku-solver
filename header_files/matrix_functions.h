#pragma once
#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H

typedef struct{
	double ** val;
	int row, col;
}Matrix;

Matrix * createMatrix(int row, int col);
void initializeMatrix(Matrix *m); //initialize matrix with random values
void fillMatrix(Matrix *m, int n);
void freeMatrix(Matrix *m);
void printMatrix(Matrix* m);
Matrix* copyMatrix(Matrix* m);
Matrix* loadMatrix(char* file_string);

Matrix* flattenMatrix(Matrix* m);
Matrix* multiply(Matrix *m1, Matrix *m2);
Matrix* add(Matrix *m1, Matrix *m2);

Matrix* subtract(Matrix *m1, Matrix *m2);

Matrix* dot(Matrix *m1, Matrix *m2);
Matrix* addScalar(Matrix* m, double n);
Matrix* transpose(Matrix* m);
#endif