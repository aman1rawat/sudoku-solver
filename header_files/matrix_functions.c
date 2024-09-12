#include "matrix_functions.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

Matrix * createMatrix(int row, int col){
	Matrix *matrix = (Matrix*)malloc(row*sizeof(Matrix));
	matrix->row = row;
	matrix->col = col;
	matrix->val = (double**)malloc(row*sizeof(double*));
	for(int i=0;i<row;i++){
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
	printf("Rows: %d Columns: %d\n", m->row, m->col);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			printf("%1.3f ", m->val[i][j]);
		}
		printf("\n");
	}
}

Matrix* copyMatrix(Matrix* m) {
	Matrix* matrix = createMatrix(m->row, m->col);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			matrix->val[i][j] = m->val[i][j];
		}
	}
	return matrix;
}

Matrix* loadMatrix(char* file_string) {
    FILE* file = fopen(file_string, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    char entry[1000];
    if (fgets(entry, sizeof(entry), file) == NULL) {
        perror("Error reading file");
        fclose(file);
        exit(1);
    }

    char* comma_pos = strchr(entry, ',');
    if (comma_pos == NULL) {
        fprintf(stderr, "Invalid file format\n");
        fclose(file);
        exit(1);
    }

    *comma_pos = '\0';
    char* matrix_data = entry;

    Matrix* m = createMatrix(9, 9);

    // Parse matrix values
    int index = 0;
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            if (matrix_data[index] >= '0' && matrix_data[index] <= '9') {
                m->val[i][j] = matrix_data[index] - '0';  
                index++;
            } 
            else {
                fprintf(stderr, "Invalid character in matrix data\n");
                freeMatrix(m);
                fclose(file);
                exit(1);
            }
        }
    }

    printf("Successfully loaded matrix from %s\n", file_string);
    fclose(file);
    return m;
}

Matrix* flattenMatrix(Matrix* m) {
	Matrix* matrix = createMatrix(1, m->row*m->col);
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			matrix->val[0][i * m->col + j] = m->val[i][j];
		}
	}
	return matrix;
}

Matrix* multiply(Matrix *m1, Matrix *m2){
	if((m1->row==m2->row) && (m1->col==m2->col)) {
		Matrix *m = createMatrix(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->val[i][j] = m1->val[i][j] * m2->val[i][j];
			}
		}
		return m;
	} 
	else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* add(Matrix *m1, Matrix *m2){
	if((m1->row==m2->row) && (m1->col==m2->col)){
		Matrix *m = createMatrix(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->val[i][j] = m1->val[i][j] + m2->val[i][j];
			}
		}
		return m;
	} 
	else{
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* subtract(Matrix *m1, Matrix *m2){
	if((m1->row==m2->row) && (m1->col==m2->col)){
		Matrix *m = createMatrix(m1->row, m1->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				m->val[i][j] = m1->val[i][j] - m2->val[i][j];
			}
		}
		return m;
	} 
	else{
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		fflush(stdout);  	
		exit(1);
	}
}


Matrix* dot(Matrix *m1, Matrix *m2){
	if (m1->col == m2->row) {
		Matrix *m = createMatrix(m1->row, m2->col);
		for (int i = 0; i < m1->row; i++) {
			for (int j = 0; j < m2->col; j++) {
				double sum = 0;
				for (int k = 0; k < m2->row; k++) {
					sum += m1->val[i][k] * m2->val[k][j];
				}
				m->val[i][j] = sum;
			}
		}
		return m;
	}
	else{
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->row, m1->col, m2->row, m2->col);
		exit(1);
	}
}

Matrix* addScalar(Matrix* m, double n) {
	Matrix* matrix = copyMatrix(m);
	for (int i = 0; i < matrix->row; i++) {
		for (int j = 0; j < matrix->col; j++) {
			matrix->val[i][j] += n;
		}
	}
	return matrix;
}

Matrix* transpose(Matrix* m) {
	Matrix* matrix = copyMatrix(m);
	for (int i = 0; i < matrix->row; i++) {
		for (int j = 0; j < matrix->col; j++) {
			matrix->val[j][i] = m->val[i][j];	
		}
	}
	return matrix;
}
