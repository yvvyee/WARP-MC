#!/bin/bash
# yvvyee: build NNInterpCuda + gpu_test. CPU reference NNInterp.cpp is compiled read-only from the VTM tree.
set -e
cd "$(dirname "$0")"
VTM=/home/ywlee/SSD/dev/VVC/VTM-23.8_nn/source/Lib/CommonLib
NVCC=/usr/local/cuda/bin/nvcc
ARCH="${ARCH:--gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=sm_80}"
g++ -O3 -march=native -std=c++17 -I$VTM -c $VTM/NNInterp.cpp -o NNInterp.o
$NVCC -O3 -std=c++17 $ARCH -I$VTM -Xptxas -v -c nninterp_cuda.cu -o nninterp_cuda.o 2> ptxas.log || { cat ptxas.log; exit 1; }
grep -E "registers|spill|error" ptxas.log | head -20
g++ -O3 -march=native -std=c++17 -fopenmp -I$VTM -I/usr/local/cuda/include -c gpu_test.cpp -o gpu_test.o
$NVCC $ARCH -o gpu_test gpu_test.o nninterp_cuda.o NNInterp.o -Xcompiler -fopenmp -lgomp
