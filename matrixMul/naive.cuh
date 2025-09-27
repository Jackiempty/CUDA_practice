#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int l = 0; l < N; l++) {
            sum += A[row * N + l] * B[l * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void naive(const float* A, const float* B, float* C, int M, int N, int K, cudaEvent_t* start, cudaEvent_t* stop) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(*start);
    matrix_multiplication_kernel_naive<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaEventRecord(*stop);
    cudaDeviceSynchronize();
}
