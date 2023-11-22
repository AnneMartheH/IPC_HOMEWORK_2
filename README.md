# IPC_HOMEWORK_2

To run the files in this cluster you have to upload the file an the related pbs file to the unitn cluster. Then you hav eto run the related file by typing this instructions:

-qsub filename.pbs

After the file has run you could find the results in the .o file with the same name as the .pbs file.

The c files with the realted pbs files:
- the c file: nBlocksTest is related to the pbs file: nBlocksTest.pbs
- the c file: task1_AntattBest.c is related to the pbs file: task1_AntattBest_3000.pbs
- the c file: task2_AntattBest.c is related to the pbs file: task2_AntattBest_16384.pbs
- the c file: task1TimeTests.c is related to the pbs file: task1TimeTests_4.pbs
- the c file: task2e1TimeTests.c is related to the pbs file: task2e1TimeTest_4.pbs
- the c file: task2e2TimeTest_c.c is related to the pbs file: task2e2TimeTest_c_4.pbs

In order to run one of the above files with a different MatrixSize, you can change the variable MatrixSize once in the top section of each .c file. And then run the related pbs file again. 

The nBlocksTest.c is a file where different block sizes has been tested on the matBlockTPar function.
task1_AntattBest.c is where the code for matMul and matMulPar is
task2_AntattBest.c is where the code for the matT, matTPar, matBlockT and matBlockTPar is
task1TimeTests.c is where I tried different impementations of the matMulPar function
task2e1TimeTests.c is where different impementations of the matTPar function was tried
task2e2TimeTests_c.c is where different implementations of the matBlockTPar was tried

The computer i have worked on has this processor: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz   2.80 GHz
x64 based proccesor

File for plotting is scaling.py that is under lecture 11 at this subjects course site. In order to plot the output from a file you need to copy the output from the function from the .o file. And then copy it into the scaling.py file. Here the plottet data must be put into the array wih the name T.
