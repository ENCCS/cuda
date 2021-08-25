#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <cuda_runtime.h>

void dot(int numElements, const float3* a, const float3* b, float* c)
{
    for (int i = 0; i < numElements; i++)
    {
        c[i] = a[i].x*b[i].x + a[i].y*b[i].y + a[i].z*b[i].z;
    }
}

int main()
{
    int numElements = 10000;

    float3* a = (float3*)calloc(numElements, sizeof(float3));
    float3* b = (float3*)calloc(numElements, sizeof(float3));
    float* c = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        a[i].x = float(rand())/float(RAND_MAX + 1.0);
        a[i].y = float(rand())/float(RAND_MAX + 1.0);
        a[i].z = float(rand())/float(RAND_MAX + 1.0);

        b[i].x = float(rand())/float(RAND_MAX + 1.0);
        b[i].y = float(rand())/float(RAND_MAX + 1.0);
        b[i].z = float(rand())/float(RAND_MAX + 1.0);
    }

    dot(numElements, a, b, c);

    for (int i = 0; i < std::min(10, numElements); i++)
    {
        printf("%f*%f + %f*%f + %f*%f = %f\n", a[i].x, b[i].x, a[i].y, b[i].y, a[i].z, b[i].z, c[i]);
    }
    printf("...\n");

    free(a);
    free(b);
    free(c);
    
    return 0;
}

