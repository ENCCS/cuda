#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

#define BLOCK_SIZE 256

static constexpr int numIterations = 100;
static constexpr int numValuesToPrint = 10;

__global__ void func1_kernel(const float* in, float* out, int numElements)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < numElements)
    {
        float value = in[i];
        for (int iter = 0; iter < numIterations; iter++)
        {
            value = sinf(value);
        }
        out[i] = value;
    }
}

__global__ void func2_kernel(const float* in1, const float* in2, float* out, int numElements)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < numElements)
    {
        float value1 = in1[numElements - i - 1];
        float value2 = in2[i];
        for (int iter = 0; iter < numIterations; iter++)
        {
            value2 = -sinf(value2);
        }
        out[i] = value1 + value2;
    }
}

int main(int argc, char* argv[])
{

    int numElements = (argc > 1) ? atoi(argv[1]) : 1000000;

    printf("Transforming %d values.\n", numElements);

    float* h_data1   = (float*)calloc(numElements, sizeof(float));
    float* h_data2   = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        h_data1[i] = float(rand())/float(RAND_MAX + 1.0);
        h_data2[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = numElements/BLOCK_SIZE + 1;

    float* d_data1;
    float* d_data2;

    cudaMalloc((void**)&d_data1, numElements*sizeof(float));
    cudaMalloc((void**)&d_data2, numElements*sizeof(float));

    // Timing
    clock_t start = clock();

    cudaMemcpy(d_data1, h_data1, numElements*sizeof(float), cudaMemcpyHostToDevice);
    func1_kernel<<<numBlocks, threadsPerBlock>>>(d_data1, d_data1, numElements);
    cudaMemcpy(h_data1, d_data1, numElements*sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_data2, h_data2, numElements*sizeof(float), cudaMemcpyHostToDevice);
    func2_kernel<<<numBlocks, threadsPerBlock>>>(d_data1, d_data2, d_data2, numElements);
    cudaMemcpy(h_data2, d_data2, numElements*sizeof(float), cudaMemcpyDeviceToHost);

    // Timing
    clock_t finish = clock();

    printf("The results are:\n");
    for (int i = 0; i < numValuesToPrint; i++)
    {
        printf("%f, %f\n", h_data1[i], h_data2[i]);
    }
    printf("...\n");
    for (int i = numElements - numValuesToPrint; i < numElements; i++)
    {
        printf("%f, %f\n", h_data1[i], h_data2[i]);
    }
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < numElements; i++)
    {
        sum1 += h_data1[i];
        sum2 += h_data2[i];
    }
    printf("The summs are: %f and %f\n", sum1, sum2);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(h_data1);
    free(h_data2);
    
    cudaFree(d_data1);
    cudaFree(d_data2);

    return 0;
}