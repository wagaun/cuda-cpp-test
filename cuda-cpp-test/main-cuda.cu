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

void stddevPointer(double *sample, double *output, int *n) {
    double out = 0;
    for (int j = 0; j < *n; j++) {
        out += sample[j];
    }
    *output = out / (*n - 1);
}

double* stddev(double **samples, int m, int n) {
    double* out = (double*) malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        stddevPointer(samples[i], &out[i], &n);
        cout << "stddev " << out[i] << endl;
    }
    return out;
}

double** generateRandomMatrix(int m, int n) {
    double **matrix;
    matrix = (double **)malloc(m * sizeof(double*));
    for(int i = 0; i < m; i++) {
        matrix[i] = (double *)malloc(n * sizeof(double));
        for(int j = 0; j < n; j++) {
            matrix[i][j] = (double) rand() / RAND_MAX;
        }
    }
    return matrix;
}

void freeMatrix(double **matrix, int m) {
    for(int i = 0; i < m; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

double diffclock(clock_t clock1, clock_t clock2)
{
    double diffticks = clock1 - clock2;
    double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
    return diffms;
}

int main(int argc, const char * argv[]) {
    int m = 100;
    int n = 1000000;
    double **sample = generateRandomMatrix(m, n);

    clock_t start = clock();
    double* output = stddev(sample, m, n);
    clock_t end = clock();

    cout << "Took " << diffclock(end, start) << "ms" << endl;

    freeMatrix(sample, m);
    free(output);
    return 0;
}
