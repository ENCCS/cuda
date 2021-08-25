#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void matAdd(int width, int height, const float* A, const float* B, float* C)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < height)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j < width)
        {
            int index = i*width + j;
            C[index] = A[index] + B[index];
        }
    }
}

int main()
{
    int width = 1000;
    int height = 100;

    int numElements = width*height;

    float* h_A = (float*)calloc(numElements, sizeof(float));
    float* h_B = (float*)calloc(numElements, sizeof(float));
    float* h_C = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        h_A[i] = float(rand())/float(RAND_MAX + 1.0);
        h_B[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    float* d_A;
    float* d_B;
    float* d_C;
    
    cudaMalloc((void**)&d_A, numElements*sizeof(float));
    cudaMalloc((void**)&d_B, numElements*sizeof(float));
    cudaMalloc((void**)&d_C, numElements*sizeof(float));

    cudaMemcpy(d_A, h_A, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements*sizeof(float), cudaMemcpyHostToDevice);

    dim3 numBlocks(height/BLOCK_SIZE_X + 1, width/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    matAdd<<<numBlocks, threadsPerBlock>>>(width, height, d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, numElements*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(5, height); i++)
    {
        for (int j = 0; j < std::min(5, width); j++)
        {
            int index = i*width + j;
            printf("%3.2f + %3.2f = %3.2f;\t", h_A[index], h_B[index], h_C[index]);
        }
        printf("...\n");
    }
    printf("...\n");

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
