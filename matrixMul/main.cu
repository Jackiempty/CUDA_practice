#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

#include "naive.cuh"
#include "optimized.cuh"

#define M 256  // Number of rows in A and C
#define N 512   // Number of columns in A and rows in B
#define K 256  // Number of columns in B and C
#define BLOCK_SIZE 32

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2], 
//      [3, 4], 
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_naive, *h_c_gpu_optimized;
    float *d_a, *d_b, *d_c_naive, *d_c_optimized;
    int size_A = M * N * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * K * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu_naive = (float*)malloc(size_C);
    h_c_gpu_optimized = (float*)malloc(size_C);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_a, M, N);
    init_matrix(h_b, N, K);

    // Allocate device memory
    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c_naive, size_C);
    cudaMalloc(&d_c_optimized, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        // vector_add_cpu(h_a, h_b, h_c_cpu, N);
        matmul_cpu(h_a, h_b, h_c_cpu, M, N, K);
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        matmul_cpu(h_a, h_b, h_c_cpu, M, N, K);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // Benchmark GPU naive implementation
    printf("Benchmarking GPU naive implementation...\n");
    double gpu_naive_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_naive, 0, size_C);  // Clear previous results
        double start_time = get_time();
        naive(d_a, d_b, d_c_naive ,M , N, K);
        double end_time = get_time();
        gpu_naive_total_time += end_time - start_time;
    }
    double gpu_naive_avg_time = gpu_naive_total_time / 100.0;

    // Verify naive results immediately
    cudaMemcpy(h_c_gpu_naive, d_c_naive, size_C, cudaMemcpyDeviceToHost);
    bool correct_naive = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_naive[i]) > 1e-4) {
            correct_naive = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_naive[i] << std::endl;
            break;
        }
    }
    printf("Naive Results are %s\n", correct_naive ? "correct" : "incorrect");

    // Benchmark GPU optimized implementation
    printf("Benchmarking GPU optimized implementation...\n");
    double gpu_optimized_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_optimized, 0, size_C);  // Clear previous results
        double start_time = get_time();
        optimized(d_a, d_b, d_c_optimized ,M , N, K);
        double end_time = get_time();
        gpu_optimized_total_time += end_time - start_time;
    }
    double gpu_optimized_avg_time = gpu_optimized_total_time / 100.0;

    // Verify optimized results immediately
    cudaMemcpy(h_c_gpu_optimized, d_c_optimized, size_C, cudaMemcpyDeviceToHost);
    bool correct_optimized = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_optimized[i]) > 1e-4) {
            correct_optimized = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_optimized[i] << std::endl;
            break;
        }
    }
    printf("Optimized Results are %s\n", correct_optimized ? "correct" : "incorrect");

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU naive average time: %f milliseconds\n", gpu_naive_avg_time * 1000);
    printf("GPU optimized average time: %f milliseconds\n", gpu_optimized_avg_time * 1000);
    printf("Speedup (CPU vs GPU naive): %fx\n", cpu_avg_time / gpu_naive_avg_time);
    printf("Speedup (CPU vs GPU optimized): %fx\n", cpu_avg_time / gpu_optimized_avg_time);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_naive);
    free(h_c_gpu_optimized);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_naive);
    cudaFree(d_c_optimized);

    return 0;
}