#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#define MatrixSize 1000

//assumption that the matrixes always are square marixes of the same size, such that they are easy to multiply//
//write this comment in report 

float** matMul(float **A, float**B){
    //C allocating and initialization
    int i,j,k;
    //float sum = 0;
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
            float sum = 0;
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

float** matMulParV1(float **A, float**B){
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
                    D[i][j]=sum;  // kan denne egt paralelleiseres når det er en
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell v1 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return D;
}

float** matMulParV2(float **A, float**B){ 
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **E = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        E[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            E[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(k)
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    float sum = 0;
                    for (k = 0; k<MatrixSize; k++){
                        //D[i][j]+=A[i][k]*B[k][j];
                        sum+=A[i][k]*B[k][j];
                        //private(k,sum)
                    }
                    E[i][j]=sum;  // kan denne egt paralelleiseres når det er en
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell v2 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return E;
}

float** matMulParV3(float **A, float**B){
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **F = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        F[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            F[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel for firstprivate(k)
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    float sum = 0;
                    for (k = 0; k<MatrixSize; k++){
                        //D[i][j]+=A[i][k]*B[k][j];
                        sum+=A[i][k]*B[k][j];
                        //private(k,sum)
                    }
                    F[i][j]=sum;  // kan denne egt paralelleiseres når det er en
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell V3 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return F;
}

float** matMulParV4(float **A, float**B){
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **G = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        G[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            G[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel for private(k) schedule(dynamic) 
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    float sum = 0;
                    for (k = 0; k<MatrixSize; k++){
                        //D[i][j]+=A[i][k]*B[k][j];
                        sum+=A[i][k]*B[k][j];
                        //private(k,sum)
                    }
                    G[i][j]=sum;  // kan denne egt paralelleiseres når det er en
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell V4 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return G;
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

float** matMulParV7(float **A, float**B){ 
    //C allocating
    int i,j,k;
    //float sum = 0;
    clock_t t2;
    double wt1,wt2;
    float **J = (float**)malloc(MatrixSize * sizeof(float *));
    for( i = 0; i< MatrixSize; i++){
        J[i] = (float *)malloc(MatrixSize * sizeof(float *));
    }
    for ( i = 0; i<MatrixSize; i++){
        for (j=0; j<MatrixSize; j++){
            J[i][j]= 0;
        }
    } 
    
    //function work
    t2 = clock();
    wt1 = omp_get_wtime();
    #pragma omp parallel private(k) 
        #pragma omp for ordered schedule(dynamic)
            for(i=0; i<MatrixSize; i++){
                for(j = 0; j<MatrixSize; j++){
                    for (k = 0; k<MatrixSize; k++){
                        #pragma omp atomic
                        //#pragma omp ordered
                        J[i][j]+=A[i][k]*B[k][j];
                    }
                }
            }
    
    wt2 = omp_get_wtime();
    t2 = clock()-t2;
	//printf( "CPU time paralell (clock) = %12.4g sec\n", (t2)/1000000.0 );
    printf( "wall clock time paralell V7 (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );

    return J;
}

// make a pbs file for the V5 function, so it can be started in the cluster 
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
    float **D = matMulParV1(A,B);
    float **E = matMulParV2(A,B);
    float **F = matMulParV3(A,B);
    float **G = matMulParV4(A,B);
    float **H = matMulParV5(A,B);
    float **I = matMulParV6(A,B);
    float **J = matMulParV7(A,B);

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

    printf("\n matrix: H: ");
     for (i = 0; i <MatrixSize; i++){
        printf("\n");
        for(j = 0; j < MatrixSize; j++){
            printf("%f ",H[i][j]);
        }
     } */

    //freeeing allocated memory
     for (i = 0; i< MatrixSize; i++){
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(E[i]);
        free(F[i]);
        free(G[i]);
        free(H[i]);
        free(I[i]);
        free(J[i]);
        //free(D[i]);
     }
     free(A);
     free(B);
     free(C);
     free(E);
     free(F);
     free(G);
     free(H);
     free(I);
     free(J);
     //free(D);
    } 

