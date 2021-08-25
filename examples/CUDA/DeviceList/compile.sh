#!/bin/bash

module load gcc/system
module use /proj/snic2021-22-274/hpc_sdk/modulefiles
module load nvhpc

gcc list_devices.cpp -o list_devices