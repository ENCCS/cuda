#!/bin/bash

nvcc pngwriter.cpp heat_equation_gpu_1.cu -lpng -o heat
#nvcc pngwriter.cpp heat_equation_gpu_2_remove_copy_calls.cu -lpng -o heat
#nvcc pngwriter.cpp heat_equation_gpu_3_shared_memory.cu -lpng -o heat