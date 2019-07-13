#include "src/Particle.cu"

template<typename VectorT>
__global__ 
void print_particles_kernel(Particle<VectorT> *P, int n) {
    printf("Particles: \n");
    for (int i=0; i<n; i++) {
        printf("\t");
        print_particle(&P[i]);
        printf("\n");
    }
}

template<typename VectorT>
__global__ 
void init_particles_kernel(Particle<VectorT> *P, int size){

    int index=blockIdx.x * blockDim.x + threadIdx.x;
    if( index < size ){
        new (P+index) Particle<VectorT>{};
       
        (*(P+index)).position[0] = index;
        (*(P+index)).velocity[1] = index;
        (*(P+index)).force[2] = index;
    }
}

int main( int argc, char **argv ) {

    int n_particles=10;
    Particle<> *particles;
    cudaMallocManaged(&particles, sizeof(Particle<>) * n_particles );

    int blocks=(n_particles/128)+1;
    init_particles_kernel<<<blocks,128>>>( particles, n_particles );
    print_particles_kernel<<<1,1>>>( particles, n_particles );

    cudaFree(particles);
}