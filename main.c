#include<stdio.h>
#include<stdlib.h>

//gcc main.c header_files/matrix_functions.c -o main -Iheader_files -lm

#include "header_files/matrix_functions.h"


int main(){
	Matrix * m1 = createMatrix(3,2);
	fillMatrix(m1, 3);
	Matrix * m2 = createMatrix(3,2);
	fillMatrix(m2, 2);
	printf("\n");
	Matrix * mat = subtract(m1,m2);
	printMatrix(mat);
	freeMatrix(m1);
	freeMatrix(m2);
}