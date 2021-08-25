#!/bin/bash

module load gcc/system
module use /proj/snic2021-22-274/hpc_sdk/modulefiles
module load nvhpc

nvcc pngwriter.cpp heat_equation_gpu_1.cu -L/proj/snic2021-22-274/libpng/1.5.30/lib -lpng -o heat
#nvcc pngwriter.cpp heat_equation_gpu_2_remove_copy_calls.cu -L/proj/snic2021-22-274/libpng/1.5.30/lib -lpng -o heat
#nvcc pngwriter.cpp heat_equation_gpu_3_shared_memory.cu -L/proj/snic2021-22-274/libpng/1.5.30/lib -lpng -o heat
