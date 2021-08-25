#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

#define BLOCK_SIZE 256

__global__ void reduce_kernel(const float* data, float* result, int numElements)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < numElements)
    {
        atomicAdd(result, data[i]);
    }
}

int main(int argc, char* argv[])
{

    int numElements = (argc > 1) ? atoi(argv[1]) : 100000000;

    printf("Reducing over %d values.\n", numElements);

    float* h_data   = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        h_data[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    float h_result = 0.0;

    float* d_data;
    cudaMalloc((void**)&d_data, numElements*sizeof(float));
    cudaMemcpy(d_data, h_data, numElements*sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = numElements/BLOCK_SIZE + 1;

    float* d_result;
    cudaMalloc((void**)&d_result, 1*sizeof(float));
    cudaMemset(d_result, 0.0, 1);

    // Timing
    clock_t start = clock();

    // Call the reduction kernel
    reduce_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result, numElements);

    cudaMemcpy(&h_result, d_result, 1*sizeof(float), cudaMemcpyDeviceToHost);

    // Timing
    clock_t finish = clock();

    printf("The result is: %f\n", h_result);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_result);
    
    return 0;
}