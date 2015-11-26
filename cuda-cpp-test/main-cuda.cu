//
//  This CUDA test program calculates standard deviations of randomly generated samples of SAMPLE_SIZE.
//  The number of samples is defined by the variable nSamples in the main function.
//
//  Created by Wagner Tsuchiya on 11/24/15.
//  Copyright © 2015 Wagner Tsuchiya. All rights reserved.
//
#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

#define SAMPLE_SIZE 1000;
#define N_BLOCKS 100;

/*
 * Function that calculates the standard deviation of a sample.
 * The input is an array with sampleArraySize that contains 1-N samples of sampleSize.
 * E.g: {s(0, 0), s(0, 1), s(1, 0), s(1, 1), s(3, 0), s(3, 1)}, with sample size 2 and sampleArraySize 6.
 */
__global__ void stddevPointer(double *sample, double *output, int sampleSize, int sampleArraySize) {
    // Check the sizeof arrays
    int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int sampleIndex = outputIndex * sampleSize;
    output[outputIndex] = 0;
    for (int j = 0; j < sampleSize; j++) {
        if(sampleIndex + j >= sampleArraySize) {
            output[outputIndex] = 42;
            return;
        }
        output[outputIndex] += sample[sampleIndex + j];
    }
    output[outputIndex] /= (sampleSize - 1);
}

double* generateRandomArray(int size) {
    double *array = (double *)malloc(size * sizeof(double));
    for(int i = 0; i < size; i++) {
        array[i] = (double) rand() / RAND_MAX;
    }
    return array;
}

double diffclock(clock_t clock1, clock_t clock2)
{
    double diffticks = clock1 - clock2;
    double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

int main(int argc, const char * argv[]) {
    int nSamples = 100000;
    int nBlocks = N_BLOCKS;
    int nThreads = nSamples / nBlocks;
    int sampleSize = SAMPLE_SIZE;

    cout << "Threads: " << nThreads << endl;
    cout << "Blocks: " << nBlocks << endl;

    int sizeOfSampleArray = sampleSize * nSamples * sizeof(double);
    int sizeOfOutput = nSamples * sizeof(double);

    double *sample = generateRandomArray(nSamples * sampleSize);

    double *deviceSample;
    double *deviceOutput;
    cudaMalloc((void **) &deviceSample, sizeOfSampleArray);
    cudaMalloc((void **) &deviceOutput, sizeOfOutput);
    cudaMemcpy(deviceSample, sample, sizeOfSampleArray, cudaMemcpyHostToDevice);

    clock_t start = clock();
    // Launch stddevPointer() kernel on GPU
    stddevPointer<<<nBlocks,nThreads>>>(deviceSample, deviceOutput, sampleSize, sizeOfSampleArray);
    clock_t end = clock();

    double* output = (double*) malloc(sizeOfOutput);
    cudaMemcpy(output, deviceOutput, sizeOfOutput, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nSamples; i++) {
        cout << "Std.Dev. #" << i + 1 << ": " << output[i] << endl;
    }

    cout << "Took " << diffclock(end, start) << "ms" << endl;

    free(sample);
    free(output);

    cudaFree(deviceSample);
    cudaFree(deviceOutput);
    return 0;
}
