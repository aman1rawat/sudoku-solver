#include<stdio.h>
#include<stdlib.h>
#include<string.h>

//gcc main.c header_files/matrix_functions.c -o main -Iheader_files -lm

#include "header_files/matrix_functions.h"


int main(){
	Matrix * m = createMatrix(9,9);
	char *s = (char*)malloc(50*sizeof(char));
	strcpy(s,"D:\\Datasets\\sudoku scrambles\\sample.csv");
	m = loadMatrix(s);
	printf("\n");
	printMatrix(m);
	free(s);
	freeMatrix(m);
}