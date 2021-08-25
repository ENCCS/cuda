#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define BLOCK_SIZE 256

__global__ void vecAdd(int numElements, const float* a, const float* b, float* c)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < numElements)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int numElements = 10000;

    float* h_a = (float*)calloc(numElements, sizeof(float));
    float* h_b = (float*)calloc(numElements, sizeof(float));
    float* h_c = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        h_a[i] = float(rand())/float(RAND_MAX + 1.0);
        h_b[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    float* d_a;
    float* d_b;
    float* d_c;
    
    cudaMalloc((void**)&d_a, numElements*sizeof(float));
    cudaMalloc((void**)&d_b, numElements*sizeof(float));
    cudaMalloc((void**)&d_c, numElements*sizeof(float));

    cudaMemcpy(d_a, h_a, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numElements*sizeof(float), cudaMemcpyHostToDevice);

    vecAdd<<<numElements/BLOCK_SIZE + 1, BLOCK_SIZE>>>(numElements, d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, numElements*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(10, numElements); i++)
    {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }
    printf("...\n");

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}

