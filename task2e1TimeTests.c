#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

////// V1 eller V3 er best her, sjekk opp i custer og sjekk at begge printer rett
#define MatrixSize 2048 //2048 1024 16384

//assumption that the matrixes always are square marixes of the same size
//write this comment in report 

float** matT(float **A){ //org er best
    //C allocating
    clock_t t1;
    int i,j;
    float **AT = (float**)malloc(MatrixSize * sizeof(float *));
    for(i = 0; i< MatrixSize; i++){
        AT[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            AT[i][j]= 0;
        }
    } 

    //function work
    t1 = clock();
    for( i=0; i<MatrixSize; i++){
        for( j = 0; j<MatrixSize; j++){
            AT[i][j]=A[j][i];  
        }
    }
    t1=clock()-t1;
    printf( "CPU time sequential transpose (clock) = %12.4g sec\n", (t1)/1000000.0 );
    return AT;
}

float** matTParV1(float **A){
    //C allocating
    double wt1,wt2;
    int i,j;
    float sum;
    float **ATP = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        ATP[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j = 0; j<MatrixSize; j++){
            ATP[i][j]= 0;
        }
    } 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for collapse(2)
        for( i=0; i<MatrixSize; i++){
            for(j = 0; j<MatrixSize; j++){
                ATP[i][j]=A[j][i];
            }
        }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell transpose V1 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return ATP;
}

float** matTParV2(float **A){
    //C allocating
    double wt1,wt2;
    int i,j;
    float sum;
    float **B = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        B[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j = 0; j<MatrixSize; j++){
            B[i][j]= 0;
        }
    } 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for collapse(2) ordered
        for( i=0; i<MatrixSize; i++){
            for(j = 0; j<MatrixSize; j++){
                B[i][j]=A[j][i];
            }
        }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell transpose V2 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return B;
}

//fiske denne 
float** matTParV3(float **A){
    //C allocating
    double wt1,wt2;
    int i,j;
    float sum;
    float **C = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        C[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j = 0; j<MatrixSize; j++){
            C[i][j]= 0;
        }
    } 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for collapse(2) ordered schedule(static)
        for( i=0; i<MatrixSize; i++){
            for(j = 0; j<MatrixSize; j++){
                C[i][j]=A[j][i];
            }
        }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell transpose V3 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return C;
}

float** matTParV4(float **A){
    //C allocating
    double wt1,wt2;
    int i,j;
    float sum;
    float **D = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        D[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j = 0; j<MatrixSize; j++){
            D[i][j]= 0;
        }
    } 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel 
    #pragma omp for collapse(2) ordered schedule(static) nowait
        for( i=0; i<MatrixSize; i++){
            for(j = 0; j<MatrixSize; j++){
                D[i][j]=A[j][i];
            }
        }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell transpose V4 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return D;
}

float** matTParV5(float **A){
    //C allocating
    double wt1,wt2;
    int i,j;
    float sum;
    float **E = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        E[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j = 0; j<MatrixSize; j++){
            E[i][j]= 0;
        }
    } 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp single
        for( i=0; i<MatrixSize; i++){
            for(j = 0; j<MatrixSize; j++){
                E[i][j]=A[j][i];
            }
        }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell transpose V5 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return E;
}

int main(){
    srand(time(NULL));
    clock_t t1, t2;
    int i,j;

    //matrix allocation
    float **A = (float **)malloc(MatrixSize * sizeof(float *));
    for (i = 0; i < MatrixSize; i++) {
        A[i] = (float *)malloc(MatrixSize * sizeof(float));
    }

    //matrix initialization
    for (i = 0; i<MatrixSize; i++){
        for ( j=0; j<MatrixSize; j++){
            A[i][j]= (float)(rand()%20);
        }
    } 

    //call upon function matT
    float **AT = matT(A);
     //call upon function matTPar
    float **ATP = matTParV1(A);
    //call upon function matBlockT
    float **B = matTParV2(A);
    float **C = matTParV3(A);
    float **D = matTParV4(A);
    float **E = matTParV5(A);

    printf("\n");


    //prints for testing 
    /*
     printf("\n matrix: A: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",A[i][j]);
        }
     } *//*
    printf("\n matrix: AT: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",AT[i][j]);
        }
     }*/
     /*
    printf("\n matrix: ATP: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",ATP[i][j]);
        }
     } */

    //freeeing allocated memory
     for (i = 0; i< MatrixSize; i++){
        free(A[i]);
        free(AT[i]);
        free(ATP[i]);
        free(B[i]);
        free(C[i]);
        free(D[i]);
        free(E[i]);


     }
     free(A);
     free(AT);
     free(ATP);
     free(B);
     free(C);
     free(D);
     free(E);

    } 

