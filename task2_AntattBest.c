#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#define MatrixSize 2048 //2048 1024 16384

//assumption that the matrixes always are square marixes of the same size
//write this comment in report 

float** matT(float **A){
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

float** matBlockT(float **A){
    //C allocating matrix that will be returned
    clock_t t1;
    double wt1,wt2;
    int bi,bj,i,j;
    float **ABT = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        ABT[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            ABT[i][j]= 0;
        }
    } 

    int nBlocks = 32;
    int blockSize = MatrixSize/nBlocks;

    //function work
    t1 = clock();
    wt1 = omp_get_wtick();
    for(bi = 0; bi < MatrixSize; bi += blockSize){
        for(bj = 0; bj < MatrixSize; bj += blockSize){
            for(i = bi; i < bi + blockSize; i++){
                for(j = bj; j < bj + blockSize; j++){
                    ABT[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtick();
    t1 = clock() -t1;
    printf( "CPU time sequential block transpose (clock) = %12.4g sec\n", (t1)/1000000.0 );
    return ABT;
}

float** matBlockTPar(float **A){
    //C allocating matrix that will be returned
    double wt1,wt2;
    int bi,bj,i,j;
    float **ABTP = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        ABTP[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            ABTP[i][j]= 0;
        }
    } 

    int nBlocks = 32;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(bi, bj, i, j) collapse(2)
    for(bi = 0; bi < MatrixSize; bi += blockSize){
        for(bj = 0; bj < MatrixSize; bj += blockSize){
            for(i = bi; i < bi + blockSize; i++){
                for(j = bj; j < bj + blockSize; j++){
                    ABTP[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell block transpose (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return ABTP;
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
    float **D = matTParV4(A);
    //call upon function matBlockT

    float **ABT = matBlockT(A);
    //call upon function matBlockTPar
    float **ABTP = matBlockTPar(A);

    printf("\n");
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
     } *//*
    printf("\n matrix: ABT: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",ABT[i][j]);
        }
     }*/ /*
    printf("\n matrix: ABTP: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",ABTP[i][j]);
        }
     } */

    //freeeing allocated memory
     for (i = 0; i< MatrixSize; i++){
        free(A[i]);
        free(AT[i]);
        free(D[i]);
        free(ABT[i]);
        free(ABTP[i]);
     }
     free(A);
     free(AT);
     free(D);
     free(ABT);
     free(ABTP);
    } 

