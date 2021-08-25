#!/bin/bash

module load gcc/system
module use /proj/snic2021-22-274/hpc_sdk/modulefiles
module load nvhpc

gcc vector_dot_product.cpp -o vector_dot_product