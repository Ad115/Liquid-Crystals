#include <iostream>
#include <fstream>
#include <numeric>

#define CUDA_ENABLED 1

#include "src/Particle.cu"



int main( int argc, char **argv )
{

    // Test vector host
    Particle *particles;
    int n_particles=100;
    cudaMallocManaged(&particles, sizeof(Particle) * n_particles );

    int blocks=(n_particles/128)+1;
    initParticlesDevice<<<blocks,128>>>( particles, n_particles );
    printParticlesDevice<<<1,1>>>( particles, n_particles );

    cudaFree(particles);


    return 0;
}