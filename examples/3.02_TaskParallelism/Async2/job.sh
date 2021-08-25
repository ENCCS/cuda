#!/bin/bash

#SBATCH -J reduction
#SBATCH -t 00:10:00
#SBATCH -o output.o
#SBATCH -e error.e
#SBATCH -n 1
###SBATCH --mem=10

./async_cpu 1000000
