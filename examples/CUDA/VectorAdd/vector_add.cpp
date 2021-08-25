#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

void vecAdd(int numElements, const float* a, const float* b, float* c)
{
    for (int i = 0; i < numElements; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int numElements = 10000;

    float* a = (float*)calloc(numElements, sizeof(float));
    float* b = (float*)calloc(numElements, sizeof(float));
    float* c = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        a[i] = float(rand())/float(RAND_MAX + 1.0);
        b[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    vecAdd(numElements, a, b, c);

    for (int i = 0; i < std::min(10, numElements); i++)
    {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }
    printf("...\n");

    free(a);
    free(b);
    free(c);
    
    return 0;
}

