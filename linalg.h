#pragma once
#include <math.h>

// Matrix-vector
void matvec(int M, int N, double *A, double *x, double *Ax) {
	int i, j;
	for (i = 0; i < N; i++) {
		Ax[i] = 0.0;
		for (j = 0; j < M; j++) {
			Ax[i] += A[j + M*i] * x[j];
		}
	}
}

// Matrix-matrix
void matmat(int M, int N, int O, double *A, double *B, double *C) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			C[j+M*i] = 0.0;
			for (k = 0; k < O; k++) {
				C[j+M*i] += A[k + M*i]*B[j + O*k];
			}
		}
	}
}

// Dot product
double dot(int N, double *x, double *y) {
	int i;
	double sum = 0;

	for (i = 0; i < N; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}

// 2-norm
double eucl_norm(int N, double *x) {
	int i;
	double sum, norm;

	sum = 0;
	for (i = 0; i < N;i++) {
		sum += pow(x[i],2);
	}
	norm = sqrt(sum);
	return norm;
}