#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>

static constexpr int numIterations = 100;
static constexpr int numValuesToPrint = 10;

void func1(const float* in, float* out, int numElements)
{
    for (int i = 0; i < numElements; i++)
    {
        float value = in[i];
        for (int iter = 0; iter < numIterations; iter++)
        {
            value = std::sin(value);
        }
        out[i] = value;
    }
}

void func2(const float* in1, const float* in2, float* out, int numElements)
{
    for (int i = 0; i < numElements; i++)
    {
        float value1 = in1[numElements - i - 1];
        float value2 = in2[i];
        for (int iter = 0; iter < numIterations; iter++)
        {
            value2 = -std::sin(value2);
        }
        out[i] = value1 + value2;
    }
}

int main(int argc, char* argv[])
{

    int numElements = (argc > 1) ? atoi(argv[1]) : 1000000;

    printf("Transforming %d values.\n", numElements);

    float* data1   = (float*)calloc(numElements, sizeof(float));
    float* data2   = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        data1[i] = float(rand())/float(RAND_MAX + 1.0);
        data2[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    // Timing
    clock_t start = clock();

    func1(data1, data1, numElements);
    func2(data1, data2, data2, numElements);

    // Timing
    clock_t finish = clock();

    printf("The results are:\n");
    for (int i = 0; i < numValuesToPrint; i++)
    {
        printf("%f, %f\n", data1[i], data2[i]);
    }
    printf("...\n");
    for (int i = numElements - numValuesToPrint; i < numElements; i++)
    {
        printf("%f, %f\n", data1[i], data2[i]);
    }
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < numElements; i++)
    {
        sum1 += data1[i];
        sum2 += data2[i];
    }
    printf("The summs are: %f and %f\n", sum1, sum2);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(data1);
    free(data2);
    
    return 0;
}