/* 
Para mostrar la funcionalidad del `device_obj`,  hacemos lo mismo que en las 
pruebas de `Vector` y `Container`. Nótese que el código para el 
`PeriodicBoundaryBox` se ha reducido considerablemente de esto:

```
PeriodicBoundaryBox<> *box; // Crea el puntero
cudaMallocManaged(&box, sizeof(PeriodicBoundaryBox<>)); // Haz espacio
init_container_kernel<<<1,1>>>(box, 3.); // Inicializa en GPU
...
...
...
cudaFree(box); // Libera el espacio
```

a esto:

```
device_obj< PeriodicBoundaryBox<> > box(3.); // Automatiza lo anterior
``` 

Para compilar y ejecutar:
```
nvcc test_device_obj.cu -o test_device_obj
./test_device_obj
```

Salida esperada:
```
[7.00, 7.00, 7.00]
Vector size: 3
Container = {side_lengths:[3.00, 3.00, 3.00]}
Particles: 
	{"position": [0.50, 0.50, 0.50], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.50, 0.50, 0.50], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Particles: 
	{"position": [2.50, 2.67, 0.70], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [2.50, 0.51, 2.77], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Particles: 
	{"position": [1.50, 0.37, 2.73], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.51, 0.38, 0.27], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Particles: 
	{"position": [0.50, 0.77, 1.78], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [0.51, 0.13, 0.00], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Particles: 
	{"position": [2.50, 0.87, 0.87], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [2.52, 2.75, 0.96], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
Particles: 
	{"position": [1.50, 0.68, 2.97], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
	{"position": [1.53, 2.24, 1.15], "velocity": [0.00, 0.00, 0.00], "force": [0.00, 0.00, 0.00]}
```
*/

#include "src_gpu/Vector.cu"
#include "src_gpu/Particle.cu"
#include "src_gpu/Container.cu"
#include "src_gpu/device_obj.cu"

#include <thrust/random.h>
#include <thrust/device_vector.h>


template <int Size, typename T>
__global__ 
void operate_on_vectors_device(Vector<Size,T> *v1, Vector<Size,T> *v2) {
    auto sum = 2*(3+*v1) + (1+(*v2));
    print_vector(&sum);
    printf("\nVector size: %d\n", Vector<Size,T>::dimensions);
}

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

template<typename VectorT>
__global__ 
void move_particles_kernel(Particle<VectorT> *P, int size, int step){

    int index=blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index < size ){

        // seed a random number generator
        thrust::default_random_engine rng(index*step + 10*step + 100*index);
        
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


int main( int argc, char **argv )
{
    // Test vector device
    device_obj< Vector<> > vector_1;
    device_obj< Vector<> > vector_2;
    
    operate_on_vectors_device<<<1,1>>>(
        vector_1.device_ptr(), 
        vector_2.device_ptr()
    );
    
    // Container creation
    device_obj< PeriodicBoundaryBox<> > box(3.);
    print_container_kernel<<<1,1>>>(box.device_ptr());

    // Create some test particles
    thrust::device_vector< Particle<> > particles(2);
    auto particles_dev_ptr = thrust::raw_pointer_cast(particles.data());
    
    init_particles_kernel<<<1,128>>>(particles_dev_ptr, 2);
    print_particles_kernel<<<1,1>>>(particles_dev_ptr, 2);
    
    for (int i=0; i<5; i++) {
      move_particles_kernel<<<1,128>>>(particles_dev_ptr, 2, i);
        
      // Test boundary conditions
      boundary_conditions_kernel<<<1,128>>>(particles_dev_ptr, 2, box.device_ptr());
      print_particles_kernel<<<1,1>>>(particles_dev_ptr, 2);
    }
    
}