//
//  main.cpp
//  cuda-cpp-test
//
//  Created by Wagner Tsuchiya on 11/24/15.
//  Copyright © 2015 Wagner Tsuchiya. All rights reserved.
//
#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

#define N_THREADS 10;

__global__ void stddevPointer(double *sample, double *output, int *n) {
    int sampleIndex = threadIdx.x + blockIdx.x * N_THREADS;
    double out = 0;
    for (int j = 0; j < *n; j++) {
        out += sample[sampleIndex * *n + j];
    }
    output[sampleIndex] = out / (*n - 1);
}

double* generateRandomMatrix(int m, int n) {
    double *matrix;
    matrix = (double *)malloc(m * n * sizeof(double));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            matrix[i * n + j] = (double) rand() / RAND_MAX;
        }
    }
    return matrix;
}

void freeMatrix(double *matrix) {
    free(matrix);
}

double diffclock(clock_t clock1, clock_t clock2)
{
    double diffticks = clock1 - clock2;
    double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

int main(int argc, const char * argv[]) {
    int nBlocks = 100;
    int nThreads = N_THREADS;
    int m = nBlocks * nThreads;
    int n = 100000;

    int sizeOfSample = n * m * sizeof(double);
    int sizeOfOutput = m * sizeof(double);
    int sizeOfInt = sizeof(int);

    double *sample = generateRandomMatrix(m, n);

    double *deviceSample;
    double *deviceOutput;
    int *deviceN;
    cudaMalloc((void **) &deviceSample, sizeOfSample);
    cudaMalloc((void **) &deviceOutput, sizeOfOutput);
    cudaMalloc((void **) &deviceN, sizeOfInt);
    cudaMemcpy(deviceSample, sample, sizeOfSample, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, &n, sizeOfInt, cudaMemcpyHostToDevice);

    clock_t start = clock();
    // Launch stddevPointer() kernel on GPU
    stddevPointer<<<nBlocks,nThreads>>>(deviceSample, deviceOutput, deviceN);
    clock_t end = clock();

    double* output = (double*) malloc(sizeOfOutput);
    cudaMemcpy(output, deviceOutput, sizeOfOutput, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        cout << "stddev " << output[i] << endl;
    }

    cout << "Took " << diffclock(end, start) << "ms" << endl;

    freeMatrix(sample);
    free(output);

    cudaFree(deviceSample);
    cudaFree(deviceOutput);
    cudaFree(deviceN);
    return 0;
}
