/**********************************************************

Parallel implementation of the Power iteration algorithm
using MPI (Message Passing Interface)

argv[1] -> number of iterations

Erik Boström, erikbos@kth.se

***********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "linalg.h"

#define N 500
#define PR 5
#define MASTER_TO_SLAVE_TAG 1
#define SLAVE_TO_MASTER_TAG 4

void genx(double x[N]){
  for(int i=0; i<N; i++) {
    x[i] = 1;
  }
}

void genA(double A[N][N]){
  /* Generates a matrix with user defined values */
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      printf("\nEnter value for a(%d,%d)",i,j);
      scanf("%lf",&A[i][j]);
    }
  }

  /* A[0][0] = -2; */
  /* A[0][1] = -4; */
  /* A[0][2] = 2; */
  /* A[1][0] = -2; */
  /* A[1][1] = 1; */
  /* A[1][2] = 2; */
  /* A[2][0] = 4; */
  /* A[2][1] = 2; */
  /* A[2][2] = 5; */      
}

void genStocA(double A[N][N]){
  
  /* Generates a stochastic matrix */
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      A[i][j]= rand()%100;
    }
  }
}


void printout(double A[N][N], double x[N], double y[N]){
  int i,j;
  for (i = 0; i < N; i++) {
    printf("\n");
    for (j = 0; j < N; j++)
      printf("%8.2f ", A[i][j]);
  }
  printf("\n\n\n");
  for (i = 0; i < N; i++) {
    printf("\n");
    printf("%8.2f ", x[i]);
  }
  printf("\n\n\n");
  for (i = 0; i < N; i++) {
    printf("\n");
    printf("%8.2f ", y[i]);
  }
  printf("\n\n");
}


/**********************************************************
 Main program
***********************************************************/
int main(int argc, char *argv[]){

  double A[N][N];
  double b[N];
  double b_new[N];
  double Ab[N];
  double diff[N];
  double err;
  double Ab_norm2;
  double Ab_norm2_p;
  double b_dot_Ab;
  double b_dot_Ab_p;
  int i,j;
  int part;
  int loc_lower;
  int loc_upper;
  int lower[PR];
  int upper[PR];
  int p,P;

  MPI_Status status; // store status of a MPI_Recv
  MPI_Request request; //capture request of a MPI_Isend

  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &p);

  if(p==0){

    printf("\n");
    printf("******************************************\n");
    printf("* Power iteration in parallel            *\n");
    printf("* ===========================            *\n");
    printf("* Number of processors used: P=%2d        *\n",P);
    printf("* Size of problem: N=%3d                 *\n",N);
    printf("******************************************\n\n");
    
    genx(b); // Generate initial x vector
    
    //genA(A);
    genStocA(A); // Generate random A matrix

    if(N<8){
      /* Print A matrix */
      printf("       ");
      for(i=0; i<N; i++) {
	printf("A(i,%d)    ",i);
      }
      for(i=0; i<N; i++) {
	printf("\nA(%d,j)",i);
	for(j=0; j<N; j++) {
	  printf(" %lf",A[i][j]);
	}
      }
      printf("\n");

      /* Display initial b_0 vector */
      printf("\nb0=[");
      for(i=0; i<N ; i++) {
	printf("%lf ",b[i]);
      }
      printf("]\n");
    }
    

    for (i = 1; i < P; i++) {
      part = (N / (P - 1)); // Width of the partition
      lower[i] = (i - 1) * part; // Lower bound of the partition for i:th processor
      if (((i + 1) == P) && ((N % (P - 1)) != 0)){ // If partition size does not divide N 
	upper[i] = N;
      } else {
	upper[i] = lower[i] + part;
      }
      /* Send out lower and upper bounds and the full  x vector to the slave processors */
      /* We use three tags, one for lower bound, one for upper bound and one for the partition of 
         the A matrix. */
      MPI_Isend(&lower[i], 1, MPI_INT, i, MASTER_TO_SLAVE_TAG,   MPI_COMM_WORLD, &request);
      MPI_Isend(&upper[i], 1, MPI_INT, i, MASTER_TO_SLAVE_TAG+1, MPI_COMM_WORLD, &request);
      MPI_Isend(&A[lower[i]][0], (upper[i]-lower[i])*N, MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG+2, MPI_COMM_WORLD, &request);
      MPI_Send(&b, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
  } else{

    /* Each slave recieves the partition bound information */
    MPI_Recv(&loc_lower, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG,   MPI_COMM_WORLD, &status);
    MPI_Recv(&loc_upper, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG+1, MPI_COMM_WORLD, &status);
    MPI_Recv(&A[loc_lower][0], (loc_upper-loc_lower)*N, MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG+2, MPI_COMM_WORLD, &status);
    //    printf("loclower=%d,locupper=%d,p=%d\n",loc_lower,loc_upper,p);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /***************************************************************************
   Main loop
  ****************************************************************************/  
  for (int it=0; it < atoi(argv[1]); it++){
    if (p>0){

      /* Each slave recieves the b vector */    
      MPI_Recv(&b, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            
      /* Compute in parallel, then send result to master */
      Ab_norm2_p = 0.0;
      b_dot_Ab_p = 0.0;
      for (i = loc_lower; i < loc_upper; i++) {
	Ab[i] = 0.0;	
	for (j = 0; j < N; j++) {
	  Ab[i]      += A[i][j] * b[j];
	}
	Ab_norm2_p += pow(Ab[i],2);
	b_dot_Ab_p   += b[i]*Ab[i];
      }

      MPI_Send(&Ab[loc_lower],(loc_upper-loc_lower), MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD);
      MPI_Send(&Ab_norm2_p, 1, MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG+1, MPI_COMM_WORLD);
      MPI_Send(&b_dot_Ab_p, 1, MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG+2, MPI_COMM_WORLD);

    }
    if(p==0){
      Ab_norm2 = 0.0;
      b_dot_Ab = 0.0;
      for (i = 1; i < P; i++) {
      
	/* Recieve computed data from the slaves */
	MPI_Recv(&Ab[lower[i]], upper[i]-lower[i], MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(&Ab_norm2_p, 1, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG+1, MPI_COMM_WORLD, &status);
	MPI_Recv(&b_dot_Ab_p, 1, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG+2, MPI_COMM_WORLD, &status);

	Ab_norm2 += Ab_norm2_p;
	b_dot_Ab += b_dot_Ab_p;
      }
      Ab_norm2 = sqrt(Ab_norm2);
      
      err = 0.0;
      for(i=0; i<N ; i++){
	b_new[i] = Ab[i]/Ab_norm2;
	diff[i]  = b[i] - b_new[i];
	err     += pow(diff[i],2);
	b[i]     = b_new[i];
      }
      err = sqrt(err);
      printf("\nIteration %4d:",it+1);
      printf("    ||abs-err|| = %e;    eig-max = %e",err,b_dot_Ab);

      for (i = 1; i < P; i++) {	
	/* Send partition of b to slaves */
	MPI_Send(&b, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);      
      }
    }
  }

  if (p==0) printf("\n-----------------------\nEnd of program\n\n");

  //  printout(A,x,y); // Print out results of the iter
  MPI_Finalize();
  return 0;
}
