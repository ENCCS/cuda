#include <stdio.h>

// 1. Add kernel function here
// 2. Get the number of threads in a block, block and thread indices and print them:
//    printf("Hello from GPU thread %d = %d * %d + %d\n", threadIndex, blockIndex, blockSize, threadInBlock);

int main()
{
    int numThreadsInBlock = 32;
    int numBlocks = 3;

    // 1. Call the kernel in the 1D grid of ``numBlocks`` of ``numThreadsInBlock`` threads.
    // 2. CUDA kernel calls are asynchronous, which means that one have to sync the device to make sure
    //    that GPU kernel completes before program exits.

    return 0;
}
