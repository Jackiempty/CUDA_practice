#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
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
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d;
  float *d_a, *d_b, *d_c_1d;
  size_t size = N * sizeof(float);

  // Allocate host memory
  h_a = (float*)malloc(size);
  h_b = (float*)malloc(size);
  h_c_cpu = (float*)malloc(size);
  h_c_gpu_1d = (float*)malloc(size);

  // Initialize vectors
  srand(time(NULL));
  init_vector(h_a, N);
  init_vector(h_b, N);

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c_1d, size);

  // Copy data to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Warm-up runs
  printf("Performing warm-up runs...\n");
  for (int i = 0; i < 3; i++) {
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    optimized(d_a, d_b, d_c_1d, N);
    cudaDeviceSynchronize();
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

  // Benchmark GPU 1D implementation
  printf("Benchmarking GPU 1D implementation...\n");
  double gpu_1d_total_time = 0.0;
  for (int i = 0; i < 100; i++) {
    cudaMemset(d_c_1d, 0, size);  // Clear previous results
    double start_time = get_time();
    optimized(d_a, d_b, d_c_1d, N);
    double end_time = get_time();
    gpu_1d_total_time += end_time - start_time;
  }
  double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

  // Verify 1D results immediately
  cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
  bool correct_1d = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4) {
      correct_1d = false;
      std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1d[i] << std::endl;
      break;
  }
  }
  printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

}