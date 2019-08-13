/* 
## Clase `ParticleSystem`

Esta clase está diseñada para manejarse desde el Host(CPU), lo importante es que 
contiene un `thrust::device_vector` de partículas, por lo que estas viven 
completamente en el GPU y de ahí se operan. A su vez, el `Container` forma parte 
de un `device_obj`, por lo que reside también completamente en el device.  El 
`kernel` es un integrador muy simple donde cada partícula tiene su propio hilo. 
Falta algo para calcular la fuerza, esto probablemente se podrá hacer con otro 
kernel. 
*/

#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "Particle.cu"
#include "Vector.cu"
#include "Container.cu"
#include "device_obj.cu"



__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename ParticleT>
__global__                                                       
void init_kernel(ParticleT *particles, int n) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n){
        
          thrust::default_random_engine rng(index*1000 + index*index);
          rng.discard(index);

          // create a mapping from random numbers to [0,1)
          thrust::normal_distribution<double> dist(0, 1);

          // Create a random motion vector
          using vector_type = typename ParticleT::vector_type;
          vector_type delta;

          for (int i=0; i<delta.dimensions; i++) {
              float rnd_value = dist(rng);
              delta[i] = rnd_value;
          }

        particles[index].position = delta;
    }
}

template <typename ParticleT, typename ContainerT>
__global__ 
void force_kernel(ParticleT *particles, int n_particles, ContainerT *box) {
    unsigned int row = blockIdx.x;
    unsigned int column = blockIdx.y*blockDim.y + threadIdx.x;

    // Reset the forces
    if(column == 0) {
        particles[row].force = 0.;
    }

    __syncthreads();

    if( column > row && column < n_particles ){
        
        auto force = particles[row]
                        .force_law(&particles[column], box);
        
        for( int i=0; i<force.dimensions; ++i ){
            atomicAddDouble( &particles[row].force[i], force[i] );
            atomicAddDouble( &particles[column].force[i], -force[i] );
        }  
    }
}

template <typename ParticleT, typename ContainerT>
__global__                                                       
void first_half_kernel(ParticleT *particles, int n_particles, ContainerT *box, double dt) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_particles){
        
          // r(t + dt) = r(t) + v(t)*dt + 1/2*f(t)*dt^2
          auto particle = particles[index];
          auto dr = particle.velocity*dt + 0.5*particle.force*dt*dt;
          auto new_pos = particle.position + dr;

          particles[index].position = (*box).apply_boundary_conditions(new_pos); 

          // v(t + 1/2*dt) = v(t) + 1/2*f(t)*dt
          auto dv = 0.5*particle.force*dt;
          //print_vector( &dv );
          particles[index].velocity += 0.5*particle.force*dt;
    }
}            

template <typename ParticleT, typename ContainerT>
__global__                                                       
void second_half_kernel(ParticleT *particles, int n_particles, ContainerT *box, double dt)  {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n_particles){
        
          // v(t + dt) = v(t + 1/2*dt) + 1/2*f(t + dt)*dt
          auto particle = particles[index];
          particles[index].velocity += 0.5*particle.force*dt;
    }
}            

template< typename ParticleT=Particle<>, typename ContainerT=PeriodicBoundaryBox<> >
class ParticleSystem
{
    unsigned int n_particles;
    thrust::device_vector< ParticleT > particles;
    device_obj< ContainerT > box;

  public:
    
    using particle_type = ParticleT;
    using container_type = ContainerT;
    using vector_type = typename ParticleT::vector_type;
    static constexpr int dimensions = ParticleT::dimensions;
    
    ParticleSystem(unsigned int n, double numeric_density) 
        : n_particles{n},
          particles{thrust::device_vector<ParticleT>(n)},
          box{pow(n/numeric_density, 1./dimensions)} {};

    void integrator(double dt) { /*
        * Implementation of a velocity Vertlet integrator.
        * See: http://www.pages.drexel.edu/~cfa22/msim/node23.html#sec:nmni
        * 
        * This integrator gives a lower error O(dt^4) and more stability than
        * the standard forward integration (x(t+dt) += v*dt + 1/2 * f * dt^2)
        * by looking at more timesteps (t, t+dt) AND (t-dt), but in order to 
        * improve memory usage, the integration is done in two steps.
        */

        // r(t + dt) = r(t) + v(t)*dt + 1/2*f(t)*dt^2
        // v(t + 1/2*dt) = v(t) + 1/2*f(t)*dt
        first_half_step(dt);

        // r(t + dt)  -->  f(t + dt)
        update_forces();

        // v(t + dt) = v(t + 1/2*dt) + 1/2*f(t + dt)*dt
        second_half_step(dt);
    }

    void first_half_step(double dt) {
        
        ParticleT* particles_ptr = thrust::raw_pointer_cast(particles.data());
        ContainerT* box_ptr = box.device_ptr();

        unsigned int block_size = 1024;
        unsigned int grid_size = n_particles / block_size + 1;
        
        first_half_kernel<<<grid_size,block_size>>>(particles_ptr, n_particles, box_ptr, dt);
    }

    void second_half_step(double dt) {
        
        ParticleT* particles_ptr = thrust::raw_pointer_cast(particles.data());
        ContainerT* box_ptr = box.device_ptr();

        unsigned int block_size = 1024;
        unsigned int grid_size = n_particles / block_size + 1;
        
        second_half_kernel<<<grid_size,block_size>>>(particles_ptr, n_particles, box_ptr, dt);
    }

    void update_forces() {
        // As we cannot send device vectors to the kernel (as device_vector is at
        // the end of the day a GPU structure abstraction in CPU) we have to get the
        // pointer in GPU memory in order for the kernel to know where to start 
        // reading the particle array from.
        
        ParticleT* particles_ptr = thrust::raw_pointer_cast(particles.data());
        ContainerT* box_ptr = box.device_ptr();
      
        unsigned int block_size = 1024;
        
        // Esta wea si funciona 
        dim3 grid_size( 
                n_particles,
                n_particles / block_size + ( n_particles % block_size == 0 ? 0:1 ) 
             );  
        
        // Launch the kernel! As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel. No need to copy back, we can read from
        // the device vector with the ::operator[]() i.e. positions[2] and that would
        // do all the memory copying for us!

        // Update forces
        
        force_kernel<<<grid_size,block_size>>>(particles_ptr, n_particles, box_ptr);
    }
 
    void simulation_step(double dt) {
        integrator(dt);
    }
    
    void simulation_init() {
        
        ParticleT* particles_ptr = thrust::raw_pointer_cast(particles.data());

        unsigned int block_size = 1024;
        unsigned int grid_size = n_particles / block_size + 1;
        
        init_kernel<<<grid_size,block_size>>>(particles_ptr, n_particles);
    }

    void print() {
        printf("Container: \n\t");

        box.get();
        print_container(box.raw_ptr());

        printf("\n");
        
        thrust::host_vector<ParticleT> p(particles);

        printf("Particles: \n");
        for (int i=0; i<n_particles; i++) {
            printf("%d:\t", i);
            print_particle( &(p[i]) );
            printf("\n");
        }

    }
};