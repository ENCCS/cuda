#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ float getValue(const float* data, int index, int numElements)
{
    if(index < numElements)
    {
        return data[index];
    }
    else
    {
        return 0.0f;
    }
}

__global__ void reduce_kernel(const float* data, float* result, int numElements)
{
    int d_i = threadIdx.x + blockIdx.x*blockDim.x;
    
    result[d_i] = getValue(data, 2*d_i, numElements) + getValue(data, 2*d_i + 1, numElements);

    if (d_i == 0 && numElements % 2 != 0)
    {
        result[d_i] += data[numElements-1];
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
    int numBlocks = numElements/2/BLOCK_SIZE + 1;

    float* d_result1;
    float* d_result2;
    cudaMalloc((void**)&d_result1, numElements*sizeof(float));
    cudaMalloc((void**)&d_result2, numElements*sizeof(float));

    // Timing
    clock_t start = clock();

    // Main loop
    reduce_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result1, numElements);
    for (int numElementsCurrent = numElements/2; numElementsCurrent > 1; numElementsCurrent = numElementsCurrent/2)
    {
        int numBlocksCurrent = numElementsCurrent/2/BLOCK_SIZE + 1;
        reduce_kernel<<<numBlocksCurrent, threadsPerBlock>>>(d_result1, d_result2, numElementsCurrent);
        std::swap(d_result1, d_result2);
    }

    cudaMemcpy(&h_result, d_result1, 1*sizeof(float), cudaMemcpyDeviceToHost);

    // Timing
    clock_t finish = clock();

    printf("The result is: %f\n", h_result);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_result1);
    cudaFree(d_result2);
    
    return 0;
}