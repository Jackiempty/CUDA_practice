#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel_optimized(const float* input, float* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void optimized(const float* input, float* output, int rows, int cols, cudaEvent_t* start, cudaEvent_t* stop) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(*start);
    matrix_transpose_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaEventRecord(*stop);
    cudaDeviceSynchronize();
}
