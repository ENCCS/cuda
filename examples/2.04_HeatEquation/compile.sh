#!/bin/bash

module load gcc/system
module use /proj/snic2021-22-274/hpc_sdk/modulefiles
module load nvhpc

gcc pngwriter.cpp heat_equation.cpp -L/proj/snic2021-22-274/libpng/1.5.30/lib -lpng -o heat