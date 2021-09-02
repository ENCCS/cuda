#include <stdio.h>

__global__ void hello_kernel()
{
    printf("Hello from GPU thread (%d, %d) = (%d, %d) * (%d, %d) + (%d, %d)\n",
        threadIdx.x, threadIdx.y,
        blockIdx.x, blockIdx.y,
        blockDim.x, blockDim.y,
        blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

int main()
{
    dim3 numThreadsInBlock(8, 4);
    dim3 numBlocks(1, 2);

    hello_kernel<<<numBlocks, numThreadsInBlock>>>();

    cudaDeviceSynchronize();

    return 0;
}
