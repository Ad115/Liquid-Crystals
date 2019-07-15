/*
El siguiente código es para probar la funcionalidad de la clase 
`PeriodicBoundaryBox` en el device. Para ello, inicializamos dos partículas 
(reciclando código de `test_Particle.cu`) y las hacemos mover aleatoriamente 
para observar las condiciones de frontera y cómo evoluciona la distancia entre 
ellas.

Para compilar y ejecutar:
```
nvcc test_Container.cu -o test_Container
./test_Container
```

Salida esperada:
```
Container = {side_lengths:[3.00, 3.00, 3.00]}
Particles: 
	{"position": [0.50, 0.50, 0.50], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.50, 0.50], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Particles: 
	{"position": [0.88, 0.67, 1.24], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.29, 2.72, 0.52], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [-0.59, -0.95, -0.72]
Particles: 
	{"position": [1.60, 1.52, 1.41], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [2.30, 2.60, 1.30], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [0.70, 1.08, -0.11]
Particles: 
	{"position": [2.01, 1.20, 0.66], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.54, 1.62, 2.15], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [-0.47, 0.42, 1.49]
Particles: 
	{"position": [1.18, 0.53, 0.93], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [2.43, 2.11, 1.85], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [1.25, -1.43, 0.92]
Particles: 
	{"position": [1.90, 0.68, 0.99], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.58, 2.45, 2.54], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [-0.31, -1.23, -1.45]
Particles: 
	{"position": [2.40, 1.68, 0.22], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.65, 0.29, 2.56], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [-0.75, -1.39, -0.65]
Particles: 
	{"position": [1.77, 0.95, 0.89], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [2.28, 2.84, 2.79], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [0.51, -1.11, -1.10]
Particles: 
	{"position": [2.69, 0.98, 1.28], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [2.24, 0.74, 0.13], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Diff vector from particle 0 to particle 1: [-0.45, -0.24, -1.14]
...
``` 
*/


#include <thrust/random.h>

#include "src/Particle.cu"
#include "src/Container.cu"

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
    if( index < size )
        new (P+index) Particle<VectorT>;
        (P+index)->position = 0.5 + (P+index)->position;
}

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}


template<typename VectorT>
__global__ 
void move_particles_kernel(Particle<VectorT> *P, int size, int step){

    int index=blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index < size ){

        // seed a random number generator
        thrust::default_random_engine rng(hash(index*step + 10*step + 100*index));
        
        // create a mapping from random numbers to [0,1)
        thrust::uniform_real_distribution<float> dist;
        
        // Create a random motion vector                                                    
        VectorT delta;
        for (int i=0; i<delta.dimensions; i++) {
            delta[i] = 2*dist(rng) - 1;
        }
        
        
        (*(P+index)).position += delta;
    }
}
                                                    
template<typename VectorT, typename ContainerT>
__global__ 
void boundary_conditions_kernel(Particle<VectorT> *P, int size, ContainerT box){

    int index=blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index < size ){
        
        (P+index)->position = box->apply_boundary_conditions((P+index)->position);
    }
}
                                                    
template<typename VectorT, typename ContainerT>
__global__ 
void measure_distance_kernel(Particle<VectorT> *P, int size, ContainerT box){

    int index=blockIdx.x * blockDim.x + threadIdx.x;
    if( index == 0 ) {
    
        auto particle = P[0];
        for (int i=1; i<size; i++) {
            printf("Diff vector from particle 0 to particle %d: ", i);
            
            auto diff = box->distance_vector(particle.position, (P+i)->position);
            print_vector(&diff);
            printf("\n");
        }
    }
}

int main( int argc, char **argv ) {
    
    // Container creation
    PeriodicBoundaryBox<> *box;
    cudaMallocManaged(&box, sizeof(PeriodicBoundaryBox<>));
    init_container_kernel<<<1,1>>>(box, 3.);
    print_container_kernel<<<1,1>>>(box);

    // Create some test particles
    Particle<> *particles;
    cudaMallocManaged(&particles, sizeof(Particle<>) * 2 );
    init_particles_kernel<<<1,128>>>(particles, 2);
    print_particles_kernel<<<1,1>>>(particles, 2);
    
    for (int i=0; i<15; i++) {
      move_particles_kernel<<<1,128>>>(particles, 2, i);
        
      // Test boundary conditions
      boundary_conditions_kernel<<<1,128>>>(particles, 2, box);
      print_particles_kernel<<<1,1>>>(particles, 2);
        
      // Test distance function
      measure_distance_kernel<<<1,1>>>(particles, 2, box);
    }

    cudaFree(box);
    cudaFree(particles);
}