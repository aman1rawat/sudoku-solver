#include<math.h>

void initializeMatrix(float **m, int row, int col){
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			m[i][j] = (float)rand()/(float)RAND_MAX;
		}
	}
}

float * flattenMatrix(float **m, int row, int col){
	int *arr = (int*)malloc((row*col)*sizeof(float));
	int cur=0;
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			arr[cur++] = m[i][j];
		}
	}
	return arr;
}