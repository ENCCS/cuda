#include <stdio.h>

__global__ void hello_kernel()
{
    int threadInBlock = threadIdx.x;
    int blockIndex = blockIdx.x;
    int blockSize = blockDim.x;
    int threadIndex = blockIndex * blockSize + threadInBlock;
    printf("Hello from GPU thread %d = %d * %d + %d\n", threadIndex, blockIndex, blockSize, threadInBlock);
}

int main()
{
    int numThreadsInBlock = 32;
    int numBlocks = 3;

    hello_kernel<<<numBlocks, numThreadsInBlock>>>();

    cudaDeviceSynchronize();

    return 0;
}
