#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "linalg.h"
#include "creatematrix.h"

#define N 3
#define M 3

// Function declaration
double power_iteration(double *A, double tol);

//Main
void main() {
	double A[M*N], B[M*N], x[M], Ax[M], AB[M*N], xx, lambda;
	int i;

	// Generate matrix
	A[0] = -2;
	A[1] = -4;
	A[2] = 2;
	A[3] = -2;
	A[4] = 1;
	A[5] = 2;
	A[6] = 4;
	A[7] = 2;
	A[8] = 5;

	// Power iteration
	lambda = power_iteration(A, 1e-10);

}

double power_iteration(double *A, double tol) {

	double eps,lambda;
	double q[N], q_old[N], Aq[N], diff[N];
	int i,it;

	// Initial random guess
	for (i = 0; i < N; i++) {
		srand(time(NULL));
		q[i] = (double)rand()/RAND_MAX;
	}

	printf("Solves an eigenvalue problem using the power method\n");
	printf("-------------------------------------------------------\n\n");
	eps = 1;
	it = 0;
	while (eps > tol){
		matvec(N,N,A,q,Aq);
		for (i = 0; i < N; i++) {
			q_old[i] = q[i];
			q[i] = Aq[i] / eucl_norm(N,Aq);
			diff[i] = q[i] - q_old[i];
		}		
		lambda = dot(N, q, Aq);
		eps = eucl_norm(N, diff);
		it += 1;
		printf("it=%d, eps=%f\n", it,eps);
	}
	printf("\n\n------------ Result of power iteration -----------\n\n");
	printf("Maximum eigenvalue: %f\n", lambda);
	printf("Convergence reached after %d iterations, with residual %e.\n", it, eps);
	printf("Tolerance used: %e\n\n", tol);
	printf("-------------------------------------------------------\n\n");

	return lambda;
}