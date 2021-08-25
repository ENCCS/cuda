#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

void reduce(const float* data, float* result, int numElements)
{
    for (int i = 0; i < numElements/2; i++)
    {
        result[i] = data[2*i] + data[2*i + 1];
    }
    if (numElements % 2 != 0)
    {
        result[0] += data[numElements-1];
    }
}


int main(int argc, char* argv[])
{

    int numElements = (argc > 1) ? atoi(argv[1]) : 100000000;

    printf("Reducing over %d values.\n", numElements);

    float* data   = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        data[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    float* result = (float*)calloc(numElements, sizeof(float));

    // Timing
    clock_t start = clock();

    reduce(data, result, numElements);
    // Main loop
    for (int numElementsCurrent = numElements/2; numElementsCurrent > 1; numElementsCurrent /= 2)
    {
        reduce(result, result, numElementsCurrent);
    }

    // Timing
    clock_t finish = clock();

    printf("The result is: %f\n", result[0]);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(data);
    free(result);
    
    return 0;
}