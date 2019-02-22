#include <iostream>
#include <fstream>
#include <numeric>

#define CUDA_ENABLED 1

#include "src/Vector.cu"



int main( int argc, char **argv )
{

    // Test vector host
    Vector<10, double> vectorCuda{};
    vectorCuda[5] = 123.4;
    vectorCuda[2] = 0.1;

    printVector(&vectorCuda);

    Vector<3> vector_host{1.,2.,3.};
    printVector(&vector_host);


    // Test vector device
    Vector<100> *vector_ptr;
    cudaMallocManaged(&vector_ptr, sizeof(vector_ptr));
    initVector<<<1,1>>>(vector_ptr);
    printVectorDevice<<<1,1>>>(vector_ptr);
    cudaFree(vector_ptr);
    return 0;
}