/**********************************************************

Parallel implementation of the Power iteration algorithm
using MPI (Message Passing Interface)

argv[1] -> vector size, N
argv[2] -> number of iterations, N_iter

Erik Boström, erikbos@kth.se

***********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define MASTER_TO_SLAVE_TAG 1
#define SLAVE_TO_MASTER_TAG 4

void genx(double *x, int N){
  for(int i=0; i<N; i++) {
    x[i] = 1;
  }
}

void genA(double *A, int N){
  /* Generates a matrix with user defined values */
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      printf("\nEnter value for a(%d,%d)",i,j);
      scanf("%lf",&A[i*N+j]);
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

void genStocA(double *A, int N){
  
  /* Generates a stochastic matrix */
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      A[i*N+j]= rand()%100;
    }
  }
}


/**********************************************************
 Main program
***********************************************************/
int main(int argc, char *argv[]){

  /* Heap variables*/
  double *A;
  double *b;
  double *b_new;
  double *Ab;
  double *diff;
  int    *lower;
  int    *upper;
  
  /* Stack variables*/
  double err;
  double Ab_norm2;
  double Ab_norm2_p;
  double b_dot_Ab;
  double b_dot_Ab_p;
  int i,j;
  int part;
  int loc_lower;
  int loc_upper;
  int p,P;
  int N;
  int N_iter;

  MPI_Status status;   // store status of a MPI_Recv
  MPI_Request request; // capture request of a MPI_Isend
  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &p);
  
  N      = atoi(argv[1]);
  N_iter = atoi(argv[2]);
  
  A     = malloc(N*N*sizeof(double));
  b     = malloc(N*sizeof(double));
  b_new = malloc(N*sizeof(double));
  Ab    = malloc(N*sizeof(double));
  diff  = malloc(N*sizeof(double));
  lower = malloc(P*sizeof(int));
  upper = malloc(P*sizeof(int));

  if(p==0){

    printf("\n");
    printf("******************************************\n");
    printf("* Power iteration in parallel            *\n");
    printf("* ===========================            *\n");
    printf("* Number of processors used: P=%2d        *\n",P);
    printf("* Size of problem: N=%3d                 *\n",N);
    printf("* Number of iterations: N_iter=%3d       *\n",N_iter);
    printf("******************************************\n\n");

    genx(b,N); // Generate initial x vector
    
    //genA(A);
    genStocA(A,N); // Generate random A matrix
    
    if(N<8){
      /* Print A matrix */
      printf("       ");
      for(i=0; i<N; i++) {
	printf("A(i,%d)    ",i);
      }
      for(i=0; i<N; i++) {
	printf("\nA(%d,j)",i);
	for(j=0; j<N; j++) {
	  printf(" %lf",A[i*N+j]);
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
      /* We use three tags, one for lower bound, one for upper bound and one for the partition of the A matrix. */
      MPI_Isend(&lower[i], 1, MPI_INT, i, MASTER_TO_SLAVE_TAG,   MPI_COMM_WORLD, &request);
      MPI_Isend(&upper[i], 1, MPI_INT, i, MASTER_TO_SLAVE_TAG+1, MPI_COMM_WORLD, &request);
      MPI_Isend(&A[lower[i]*N], (upper[i]-lower[i])*N, MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG+2, MPI_COMM_WORLD, &request);
      MPI_Isend(b, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&request);
    }
  } else{

    /* Each slave recieves the partition bound information */
    MPI_Recv(&loc_lower, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG,   MPI_COMM_WORLD, &status);
    MPI_Recv(&loc_upper, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG+1, MPI_COMM_WORLD, &status);
    MPI_Recv(&A[loc_lower*N], (loc_upper-loc_lower)*N, MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG+2, MPI_COMM_WORLD, &status);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  
  /***************************************************************************
   Main loop
  ****************************************************************************/
  for (int it=0; it < N_iter; it++){
    if (p>0){
      
      /* Each slave recieves the b vector */
      MPI_Recv(b, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            
      /* Compute in parallel, then send result to master */
      Ab_norm2_p = 0.0;
      b_dot_Ab_p = 0.0;
      for (i = loc_lower; i < loc_upper; i++) {
      	Ab[i] = 0.0;
      	for (j = 0; j < N; j++) {
      	  Ab[i]      += A[i*N+j] * b[j];
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
  	MPI_Isend(b, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,&request);
      }
    }
  }
  if (p==0) printf("\n-----------------------\nEnd of program\n\n");
  
  //  printout(A,x,y); // Print out results of the iter
  MPI_Finalize();

  free(A); free(b); free(b_new); free(diff); free(Ab);
  free(lower); free(upper);
  return 0;
}
