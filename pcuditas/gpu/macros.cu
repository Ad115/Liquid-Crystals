#pragma once

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x));}} while(0)
