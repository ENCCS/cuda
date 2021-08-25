#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define BLOCK_SIZE 256

__global__ void dot(int numElements, const float3* a, const float3* b, float* c)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < numElements)
    {
        c[i] = a[i].x*b[i].x + a[i].y*b[i].y + a[i].z*b[i].z;
    }
}

int main()
{
    int numElements = 10000;

    float3* h_a = (float3*)calloc(numElements, sizeof(float3));
    float3* h_b = (float3*)calloc(numElements, sizeof(float3));
    float* h_c = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        h_a[i].x = float(rand())/float(RAND_MAX + 1.0);
        h_a[i].y = float(rand())/float(RAND_MAX + 1.0);
        h_a[i].z = float(rand())/float(RAND_MAX + 1.0);

        h_b[i].x = float(rand())/float(RAND_MAX + 1.0);
        h_b[i].y = float(rand())/float(RAND_MAX + 1.0);
        h_b[i].z = float(rand())/float(RAND_MAX + 1.0);
    }

    float3* d_a;
    float3* d_b;
    float* d_c;

    cudaMalloc((void**)&d_a, numElements*sizeof(float3));
    cudaMalloc((void**)&d_b, numElements*sizeof(float3));
    cudaMalloc((void**)&d_c, numElements*sizeof(float));

    cudaMemcpy(d_a, h_a, numElements*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numElements*sizeof(float3), cudaMemcpyHostToDevice);

    dot<<<numElements/BLOCK_SIZE + 1, BLOCK_SIZE>>>(numElements, d_a, d_b, d_c);
    
    cudaMemcpy(h_c, d_c, numElements*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(10, numElements); i++)
    {
        printf("%f*%f + %f*%f + %f*%f = %f\n", h_a[i].x, h_b[i].x, h_a[i].y, h_b[i].y, h_a[i].z, h_b[i].z, h_c[i]);
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
