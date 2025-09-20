#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

#include "naive.cuh"
#include "optimized.cuh"

#define N 10000000  // Vector size = 10 million

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// CPU vector addition
void vector_add_cpu(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Initialize vector with random values
void init_vector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_naive, *h_c_gpu_optimized;
    float *d_a, *d_b, *d_c_naive, *d_c_optimized;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_naive = (float*)malloc(size);
    h_c_gpu_optimized = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_naive, size);
    cudaMalloc(&d_c_optimized, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        // naive(d_a, d_b, d_c_naive, N);
        // cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // Benchmark GPU naive implementation
    printf("Benchmarking GPU naive implementation...\n");
    double gpu_naive_total_time = 0.0;
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_c_naive, 0, size);  // Clear previous results
        double start_time = get_time();
        naive(d_a, d_b, d_c_naive, N);
        double end_time = get_time();
        gpu_naive_total_time += end_time - start_time;
    }
    double gpu_naive_avg_time = gpu_naive_total_time / 100.0;

    // Verify naive results immediately
    cudaMemcpy(h_c_gpu_naive, d_c_naive, size, cudaMemcpyDeviceToHost);
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
        cudaMemset(d_c_optimized, 0, size);  // Clear previous results
        double start_time = get_time();
        optimized(d_a, d_b, d_c_optimized, N);
        double end_time = get_time();
        gpu_optimized_total_time += end_time - start_time;
    }
    double gpu_optimized_avg_time = gpu_optimized_total_time / 100.0;

    // Verify optimized results immediately
    cudaMemcpy(h_c_gpu_optimized, d_c_optimized, size, cudaMemcpyDeviceToHost);
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
    printf("Speedup (GPU naive vs GPU optimized): %fx\n", gpu_naive_avg_time / gpu_optimized_avg_time);

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