#include <stdio.h>

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }
  return 0;
}

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

        printf("%d: %s, CC %d.%d, %dx%d=%d@%dMHz CUDA cores, %luMB\n", i, prop.name,
            prop.major, prop.minor,
            prop.multiProcessorCount, _ConvertSMVer2Cores(prop.major, prop.minor),
            prop.multiProcessorCount*_ConvertSMVer2Cores(prop.major, prop.minor), prop.clockRate/1000,
            prop.totalGlobalMem/1024/1024);

    }

    return 0;
}

