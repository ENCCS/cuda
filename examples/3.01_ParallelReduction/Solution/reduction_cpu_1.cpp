#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>

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

    float result = 0.0;

    // Timing
    clock_t start = clock();

    // Main loop
    for (int i = 0; i < numElements; i++)
    {
        result = result + data[i];
    }

    // Timing
    clock_t finish = clock();

    printf("The result is: %f\n", result);

    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    free(data);
    
    return 0;
}