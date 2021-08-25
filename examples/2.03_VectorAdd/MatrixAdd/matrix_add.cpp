#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

void matAdd(int width, int height, const float* A, const float* B, float* C)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
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

    float* A = (float*)calloc(numElements, sizeof(float));
    float* B = (float*)calloc(numElements, sizeof(float));
    float* C = (float*)calloc(numElements, sizeof(float));

    srand(1214134);
    for (int i = 0; i < numElements; i++)
    {
        A[i] = float(rand())/float(RAND_MAX + 1.0);
        B[i] = float(rand())/float(RAND_MAX + 1.0);
    }

    matAdd(width, height, A, B, C);

    for (int i = 0; i < std::min(5, height); i++)
    {
        for (int j = 0; j < std::min(5, width); j++)
        {
            int index = i*width + j;
            printf("%3.2f + %3.2f = %3.2f;\t", A[index], B[index], C[index]);
        }
        printf("...\n");
    }
    printf("...\n");

    free(A);
    free(B);
    free(C);
    
    return 0;
}

