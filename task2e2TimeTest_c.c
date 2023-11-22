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

    int nBlocks = 4;
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

float** matBlockTParV1(float **A){
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

    int nBlocks = 4;
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
    printf( "wall clock time paralell block transpose V1 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return ABTP;
}

float** matBlockTParV3(float **A){
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

    int nBlocks = 4;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel 
    #pragma omp for ordered schedule(dynamic) private(bi, bj, i, j) //collapse(3)
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
    printf( "wall clock time paralell block transpose V3 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
    return C;
}

float** matBlockTParV5(float **A){
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

    int nBlocks = 4;
    int blockSize = MatrixSize/nBlocks; 

    //function work
    wt1 = omp_get_wtime();
    #pragma omp parallel 
    #pragma omp single firstprivate(j) private(bi,i,bj) nowait
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
    printf( "wall clock time paralell block transpose V5 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
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


    //call upon function matBlockT
    float **ABT = matBlockT(A);
    //call upon function matBlockTPar
    float **ABTP = matBlockTParV1(A);
    //float **B = matBlockTParV2(A);
    float **C = matBlockTParV3(A);
    //float **D = matBlockTParV4(A);
    float **E = matBlockTParV5(A);




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
    printf("\n matrix: ABT: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",ABT[i][j]);
        }
     }
    printf("\n matrix: ABTP: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",ABTP[i][j]);
        }
     }
     printf("\n matrix: V2: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",B[i][j]);
        }
     }
     printf("\n matrix: v4: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",D[i][j]);
        }
     }*/

    //freeeing allocated memory
     for (i = 0; i< MatrixSize; i++){
        free(A[i]);
        free(ABT[i]);
        free(ABTP[i]);
        //free(B[i]);
        free(C[i]);
        //free(D[i]);
        free(E[i]);

     }
     free(A);
     free(ABT);
     free(ABTP);
     //free(B);
     free(C);
     //free(D);
     free(E);
    } 

