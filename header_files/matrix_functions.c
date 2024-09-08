#include "matrix_functions.h"

#include<stdlib.h>
#include<math.h>

Matrix * createMatrix(int row, int col){
	Matrix *matrix = (Matrix*)malloc(row*sizeof(Matrix));
	matrix->row = row;
	matrix->col = col;
	matrix->val = (double**)malloc(row*sizeof(double*));
	for(int i=0;i<col;i++){
		matrix->val[i] = (double*)malloc(col*sizeof(double));
	}
	return matrix;
}

void initializeMatrix(Matrix *m){
	for(int i=0;i<m->row;i++){
		for(int j=0;j<m->col;j++){
			m->val[i][j] = (float)rand()/(float)RAND_MAX;
		}
	}
}

void fillMatrix(Matrix *m, int n){
	for(int i=0;i<m->row;i++){
		for(int j=0;j<m->col;j++){
			m->val[i][j]=n;
		}
	}
}

void freeMatrix(Matrix *m){
	for(int i=0;i<m->row;i++){
		free(m->val[i]);
	}
	free(m->val);
	free(m);
}

void printMatrix(Matrix* m) {
	printf("Rows: %d Columns: %d\n", m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%1.3f ", m->entries[i][j]);
		}
		printf("\n");
	}
}

Matrix* copyMatrix(Matrix* m) {
	Matrix* matrix = createMatrix(m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			matrix->val[i][j] = m->val[i][j];
		}
	}
	return matrix;
}
