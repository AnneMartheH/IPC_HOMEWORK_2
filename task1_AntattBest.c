#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#define MatrixSize 3000

//assumption that the matrixes always are square marixes of the same size, such that they are easy to multiply//
//write this comment in report 

float** matMul(float **A, float**B){
    //C allocating and initialization
    int i,j,k;
    float sum = 0;
    clock_t t1;
    double wt1,wt2;
    float **C = (float**)malloc(MatrixSize * sizeof(float *));
    for(i = 0; i< MatrixSize; i++){
        C[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            C[i][j]= 0;
        }
    } 

    //function work
    t1 = clock();
    for(i=0; i<MatrixSize; i++){
        for(j = 0; j<MatrixSize; j++){
            for (k = 0; k<MatrixSize; k++){
                //C[i][j]+=A[i][k]*B[k][j];
                sum += A[i][k]*B[k][j];
            }
            C[i][j]=sum;
        } 
    }
    t1 = clock()-t1;
    printf( " \n CPU time sequential (clock) = %12.4g sec\n", (t1)/1000000.0 );
    return C;
}

float** matMulPar(float **A, float**B){
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **D = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        D[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            D[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel for collapse(2) private(k)
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    float sum = 0;
                    for (k = 0; k<MatrixSize; k++){
                        //D[i][j]+=A[i][k]*B[k][j];
                        sum+=A[i][k]*B[k][j];
                        //private(k,sum)
                    }
                    D[i][j]=sum;  // kan denne egt paralelleiseres når det er en linje her
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
    printf( "wall clock time paralell (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return D;
}

float** matMulParV5(float **A, float**B){ //Best so far!!
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **H = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        H[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            H[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(k) ordered schedule(dynamic)
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    float sum = 0;
                    for (k = 0; k<MatrixSize; k++){
                        //D[i][j]+=A[i][k]*B[k][j];
                        sum+=A[i][k]*B[k][j];
                        //private(k,sum)
                    }
                    H[i][j]=sum;  // kan denne egt paralelleiseres når det er en
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell V5 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return H;
}

float** matMulParV6(float **A, float**B){ 
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **I = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        I[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            I[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel private(k) 
        #pragma omp for ordered schedule(dynamic)
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    float sum = 0;
                    for (k = 0; k<MatrixSize; k++){
                        //D[i][j]+=A[i][k]*B[k][j];
                        sum+=A[i][k]*B[k][j];
                        //private(k,sum)
                    }
                    I[i][j]=sum;  // kan denne egt paralelleiseres når det er en
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell V6 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return I;
}

int main(){
    srand(time(NULL));
    int i,j;

    //matrix allocation
    float **A = (float **)malloc(MatrixSize * sizeof(float *));
    float **B = (float **)malloc(MatrixSize * sizeof(float *));
    //float **C = (float **)malloc(MatrixSize * sizeof(float *));
    for (i = 0; i < MatrixSize; i++) {
        A[i] = (float *)malloc(MatrixSize * sizeof(float));
        B[i] = (float *)malloc(MatrixSize * sizeof(float));
        //C[i] = (float *)malloc(MatrixSize * sizeof(float));
    }

    //initialization
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            A[i][j]= (float)(rand()%20);
            B[i][j]=(float)(rand()%20);
        }
    } 

    //call upon function
    float **C = matMul(A,B);
    float **D = matMulPar(A,B);
    float **H= matMulParV5(A,B);
    //float **I= matMulParV6(A,B);
    printf("\n");
    //prints for testing 
    /*
     printf("\n matrix: A: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",A[i][j]);
        }
     }
    printf("\n matrix: B: ");
     for ( i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",B[i][j]);
        }
     }
    printf("\n matrix: C: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",C[i][j]);
        }
     }

    printf("\n matrix: D: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",D[i][j]);
        }
     } */

    //freeeing allocated memory
     for (i = 0; i< MatrixSize; i++){
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(D[i]);
        free(H[i]);
       // free(I[i]);
     }
     free(A);
     free(B);
     free(C);
     free(D);
     free(H);
    // free(I);
    } 

