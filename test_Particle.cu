/* 
El siguiente código es para probar la funcionalidad de la clase `Particle`, en 
específico, se crea un arreglo de partículas en el GPU(Device).

Para compilar y ejecutar:
```
nvcc test_Particle.cu -o test_Particle
./test_Particle
```

Salida esperada:
```
Particles: 
	{"position": [0.00, 0.00, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.00, 0.00, 0.00], "velocity": [0.00, 1.00, 0.00], "force": [0.00, 0.00, 1.00]}
	{"position": [2.00, 0.00, 0.00], "velocity": [0.00, 2.00, 0.00], "force": [0.00, 0.00, 2.00]}
	{"position": [3.00, 0.00, 0.00], "velocity": [0.00, 3.00, 0.00], "force": [0.00, 0.00, 3.00]}
	{"position": [4.00, 0.00, 0.00], "velocity": [0.00, 4.00, 0.00], "force": [0.00, 0.00, 4.00]}
	{"position": [5.00, 0.00, 0.00], "velocity": [0.00, 5.00, 0.00], "force": [0.00, 0.00, 5.00]}
	{"position": [6.00, 0.00, 0.00], "velocity": [0.00, 6.00, 0.00], "force": [0.00, 0.00, 6.00]}
	{"position": [7.00, 0.00, 0.00], "velocity": [0.00, 7.00, 0.00], "force": [0.00, 0.00, 7.00]}
	{"position": [8.00, 0.00, 0.00], "velocity": [0.00, 8.00, 0.00], "force": [0.00, 0.00, 8.00]}
	{"position": [9.00, 0.00, 0.00], "velocity": [0.00, 9.00, 0.00], "force": [0.00, 0.00, 9.00]}

```
*/

#include "src_gpu/Particle.cu"

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