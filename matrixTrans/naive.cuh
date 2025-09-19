#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel_naive(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void naive(const float* input, float* output, int rows, int cols) {
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel_naive<<<1, 1>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
