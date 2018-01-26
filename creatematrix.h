#pragma once
#include <stdio.h>
void create_matrix(int M, int N, double *A, char name) {
	int i, j;
	double a;
	printf("(M,N)=(%d,%d)\n", M, N);
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			printf("%s_%d%d=",&name, i, j);
			scanf("%lf",&a);
			A[j + M*i] = a;
	}
	}
	printf("\n");
}