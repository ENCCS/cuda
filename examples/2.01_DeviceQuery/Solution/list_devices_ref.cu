#include <stdio.h>

int main()
{
    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA driver: %d\n", driverVersion);

    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA runtime: %d\n", runtimeVersion);

    int         numDevices;
    cudaError_t stat = cudaGetDeviceCount(&numDevices);

    for (int i = 0; i < numDevices; i++)
    {
        cudaDeviceProp prop;
        stat = cudaGetDeviceProperties(&prop, i);

        printf("%d: %s, CC %d.%d, %d SMs running at %dMHz, %luMB\n", i, prop.name,
            prop.major, prop.minor,
            prop.multiProcessorCount,
            prop.clockRate/1000,
            prop.totalGlobalMem/1024/1024);
    }

    return 0;
}

