#!/bin/bash

#SBATCH -J myjob
#SBATCH -t 00:10:00
#SBATCH -o output.o
#SBATCH -e error.e
#SBATCH -n 1
###SBATCH --gpus-per-task=1
###SBATCH --mem=10

echo "CPU v1:"
./reduction_cpu_1 10000000
echo "CPU v2:"
./reduction_cpu_2 10000000

echo "GPU v1:"
./reduction_gpu_1 10000000
echo "GPU v2:"
./reduction_gpu_2 10000000
echo "GPU v3:"
./reduction_gpu_3 10000000
echo "GPU v4:"
./reduction_gpu_4 10000000
echo "GPU v5:"
./reduction_gpu_5 10000000
echo "GPU v6:"
./reduction_gpu_6 10000000
echo "GPU v7:"
./reduction_gpu_7 10000000

echo "============"
echo "With nvprof:"
echo "============"
echo "GPU v1:"
nvprof ./reduction_gpu_1 10000000
echo "GPU v2:"
nvprof ./reduction_gpu_2 10000000
echo "GPU v3:"
nvprof ./reduction_gpu_3 10000000
echo "GPU v4:"
nvprof ./reduction_gpu_4 10000000
echo "GPU v5:"
nvprof ./reduction_gpu_5 10000000
echo "GPU v6:"
nvprof ./reduction_gpu_6 10000000
echo "GPU v7:"
nvprof ./reduction_gpu_7 10000000