#include <iostream>
#include <fstream>
#include <numeric>

#define CUDA_ENABLED 1

#include "src/Container.cu"



int main( int argc, char **argv )
{

    Container<3> *container_ptr;
    cudaMallocManaged( &container_ptr, sizeof(*container_ptr) );
    init_device_container<<<1,1>>>( container_ptr, 999 );
    print_device_container<<<1,1>>>( container_ptr );

    cudaFree(container_ptr);

    return 0;
}