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


float** matBlockTPar4(float **A){
    //C allocating matrix that will be returned
    double wt1,wt2;
    int bi,bj,i,j;
    float **F = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        F[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            F[i][j]= 0;
        }
    } 

    int nBlocks = 4;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(bi, bj, i, j) collapse(2)
    for(bi = 0; bi < MatrixSize; bi += blockSize){
        for(bj = 0; bj < MatrixSize; bj += blockSize){
            for(i = bi; i < bi + blockSize; i++){
                for(j = bj; j < bj + blockSize; j++){
                    F[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell block transpose (omp_get_wtime) 4 = %12.4g sec\n", wt2-wt1 );
    return F;
}

float** matBlockTPar32(float **A){
    //C allocating matrix that will be returned
    double wt1,wt2;
    int bi,bj,i,j;
    float **B = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        B[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            B[i][j]= 0;
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
                    B[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell block transpose (omp_get_wtime) 32 = %12.4g sec\n", wt2-wt1 );
    return B;
}

float** matBlockTPar256(float **A){
    //C allocating matrix that will be returned
    double wt1,wt2;
    int bi,bj,i,j;
    float **C = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        C[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            C[i][j]= 0;
        }
    } 

    int nBlocks = 256;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(bi, bj, i, j) collapse(2)
    for(bi = 0; bi < MatrixSize; bi += blockSize){
        for(bj = 0; bj < MatrixSize; bj += blockSize){
            for(i = bi; i < bi + blockSize; i++){
                for(j = bj; j < bj + blockSize; j++){
                    C[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell block transpose (omp_get_wtime) 256 = %12.4g sec\n", wt2-wt1 );
    return C;
}

float** matBlockTPar512(float **A){
    //C allocating matrix that will be returned
    double wt1,wt2;
    int bi,bj,i,j;
    float **D = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        D[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            D[i][j]= 0;
        }
    } 

    int nBlocks = 512;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(bi, bj, i, j) collapse(2)
    for(bi = 0; bi < MatrixSize; bi += blockSize){
        for(bj = 0; bj < MatrixSize; bj += blockSize){
            for(i = bi; i < bi + blockSize; i++){
                for(j = bj; j < bj + blockSize; j++){
                    D[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell block transpose (omp_get_wtime) 512= %12.4g sec\n", wt2-wt1 );
    return D;
}

float** matBlockTPar1024(float **A){
    //C allocating matrix that will be returned
    double wt1,wt2;
    int bi,bj,i,j;
    float **E = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        E[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            E[i][j]= 0;
        }
    } 

    int nBlocks = 1024;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(bi, bj, i, j) collapse(2)
    for(bi = 0; bi < MatrixSize; bi += blockSize){
        for(bj = 0; bj < MatrixSize; bj += blockSize){
            for(i = bi; i < bi + blockSize; i++){
                for(j = bj; j < bj + blockSize; j++){
                    E[j][i]=A[i][j];
                }
            }
        }
    }
    wt2 = omp_get_wtime();
    printf( "wall clock time paralell block transpose (omp_get_wtime) 1024 = %12.4g sec\n", wt2-wt1 );
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
    float **F = matBlockTPar4(A);
    float **B = matBlockTPar32(A);
    float **C = matBlockTPar256(A);
    float **D = matBlockTPar512(A);
    float **E = matBlockTPar1024(A);

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
        free(B[i]);
        free(C[i]);
        free(D[i]);
        free(E[i]);
        free(F[i]);
     }
     free(A);
     free(B);
     free(C);
     free(D);
     free(E);
     free(F);
    } 

