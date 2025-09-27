#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <iomanip>

#include "naive.cuh"
#include "optimized.cuh"

#define M 256  // Number of rows in A and C
#define N 512  // Number of columns in A and rows in B
using namespace std;

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// CPU matrix transpose
void mat_trans_cpu(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_input, *h_output_cpu, *h_output_gpu_naive, *h_output_gpu_optimized;
    float *d_input, *d_output_naive, *d_output_optimized;
    int size = M * N * sizeof(float);

    // Allocate host memory
    h_input = (float*)malloc(size);
    h_output_cpu = (float*)malloc(size);
    h_output_gpu_naive = (float*)malloc(size);
    h_output_gpu_optimized = (float*)malloc(size);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_input, M, N);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output_naive, size);
    cudaMalloc(&d_output_optimized, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        // vector_add_cpu(h_a, h_b, h_c_cpu, N);
        mat_trans_cpu(h_input, h_output_cpu, M, N);
    }

    // Benchmark CPU implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        mat_trans_cpu(h_input, h_output_cpu, M, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // Benchmark GPU naive implementation
    printf("Benchmarking GPU naive implementation...\n");
    double gpu_naive_total_time = 0.0;
    double gpu_naive_total_cuda_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_output_naive, 0, size);  // Clear previous results
        double start_time = get_time();
        naive(d_input, d_output_naive, M, N, &start, &stop);
        double end_time = get_time();
        gpu_naive_total_time += end_time - start_time;
        // cuda event time
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        gpu_naive_total_cuda_time += milliseconds;
    }
    double gpu_naive_avg_time = gpu_naive_total_time / 100.0;
    double gpu_naive_avg_cuda_time = gpu_naive_total_cuda_time / 100.0;

    // Verify naive results immediately
    cudaMemcpy(h_output_gpu_naive, d_output_naive, size, cudaMemcpyDeviceToHost);
    bool correct_naive = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu_naive[i]) > 1e-4) {
            correct_naive = false;
            cout << i << " cpu: " << h_output_cpu[i] << " != " << h_output_gpu_naive[i] << endl;
            break;
        }
    }
    printf("Naive Results are %s\n", correct_naive ? "correct" : "incorrect");

    // Benchmark GPU optimized implementation
    printf("Benchmarking GPU optimized implementation...\n");
    double gpu_optimized_total_time = 0.0;
    double gpu_optimized_total_cuda_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_output_optimized, 0, size);  // Clear previous results
        double start_time = get_time();
        optimized(d_input, d_output_optimized, M, N, &start, &stop);
        double end_time = get_time();
        gpu_optimized_total_time += end_time - start_time;
        // cuda event time
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        gpu_optimized_total_cuda_time += milliseconds;
    }
    double gpu_optimized_avg_time = gpu_optimized_total_time / 100.0;
    double gpu_optimized_avg_cuda_time = gpu_optimized_total_cuda_time / 100.0;

    // Verify optimized results immediately
    cudaMemcpy(h_output_gpu_optimized, d_output_optimized, size, cudaMemcpyDeviceToHost);
    bool correct_optimized = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu_optimized[i]) > 1e-4) {
            correct_optimized = false;
            cout << i << " cpu: " << h_output_cpu[i] << " != " << h_output_gpu_optimized[i] << endl;
            break;
        }
    }
    printf("Optimized Results are %s\n", correct_optimized ? "correct" : "incorrect");

    // Print results
    cout << endl;
    cout << left << setw(35) << "CPU average time:" << right << setw(12) << fixed << setprecision(3)
         << cpu_avg_time * 1000 << " ms" << endl;
    cout << left << setw(35) << "GPU naive average time:" << right << setw(12) << fixed << setprecision(3)
         << gpu_naive_avg_time * 1000 << " ms" << endl;
    cout << left << setw(35) << "GPU naive average cuda time:" << right << setw(12) << fixed << setprecision(3)
         << gpu_naive_avg_cuda_time << " ms" << endl;
    cout << left << setw(35) << "GPU optimized average time:" << right << setw(12) << fixed << setprecision(3)
         << gpu_optimized_avg_time * 1000 << " ms" << endl;
    cout << left << setw(35) << "GPU optimized average cuda time:" << right << setw(12) << fixed << setprecision(3)
         << gpu_optimized_avg_cuda_time << " ms" << endl;
    cout << left << setw(35) << "Speedup (CPU vs GPU naive):" << right << setw(12) << fixed << setprecision(3)
         << cpu_avg_time / gpu_naive_avg_time << "x" << endl;
    cout << left << setw(35) << "Speedup (CPU vs GPU optimized):" << right << setw(12) << fixed << setprecision(3)
         << cpu_avg_time / gpu_optimized_avg_time << "x" << endl;
    cout << left << setw(35) << "Speedup (GPU naive vs GPU optimized):" << right << setw(12) << fixed << setprecision(3)
         << gpu_naive_avg_time / gpu_optimized_avg_time << "x" << endl;

    // destroy event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu_naive);
    free(h_output_gpu_optimized);
    cudaFree(d_input);
    cudaFree(d_output_naive);
    cudaFree(d_output_optimized);

    return 0;
}