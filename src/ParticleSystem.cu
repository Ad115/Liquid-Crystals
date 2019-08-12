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


// seed a random number generator
/*
// This is the kernel that is launched from CPU and GPU runs it for each cell
template <typename VectorT>
__global__ 
void integrator_kernel(Particle<VectorT> *particles, int n, int step) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
   

    // --- First half step ---
    map_to_particles([dt, &box=container()](ParticleClass& p){

            // r(t+dt) = r(t) + v(t)*dt + 1/2*f*dt^2
            p.position = p.position + dt * p.velocity + 1/2.*dt*dt*p.force;
            p.position = box.apply_boundary_conditions(p.position);

            // v(t+dt/2) = v(t)+f(t)/2*dt
            p.velocity = p.velocity + dt/2.*p.force;
        }
    );
    
    // r(t + dt) --> f(t + dt)
    
    // First zero out the forces
    for(auto& p : particles){ p.force = 0 * p.force; }

    for(int i=0; i<particles.size()-1; i++) {
        for(int j=i+1; j<particles.size(); j++) {
            
            Vector force = particles[i].force_law(particles[j], container());

            particles[i].force += force;
            particles[j].force -= force;
        }
    }
    
    // --- Second half step ---
    map_to_particles([dt](ParticleClass& p){

            // v(t+dt) = v(t+dt/2)+f(t+dt)/2*dt
            p.velocity = p.velocity + dt/2.*p.force;
        }
    );


}
*/
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




// This is the kernel that is launched from CPU and GPU runs it for each cell
template <typename ParticleT, typename ContainerT>
__global__ 
void integrator_kernel(ParticleT *particles,  ContainerT *box, unsigned int n, int step) {
    unsigned int row = gridDim.x;
    unsigned int column = blockIdx.y * blockDim.y + threadIdx.x;

    if( column > row && column < n ){
        auto force = particles[row].force_law(&particles[column], box);
        print_vector( &force );
        for( int i=0; i<force.dimensions; ++i ){
            atomicAddDouble( &particles[row].force[i], force[i] );
            atomicAddDouble( &particles[column].force[i], - force[i] );
        }  
    }
}
                                                      
template <typename ParticleT>
__global__                                                       
void init_kernel(ParticleT *particles, int n) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n){
        
          thrust::default_random_engine rng(index);
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

    void simulation_step(int step) {
        // As we cannot send device vectors to the kernel (as device_vector is at
        // the end of the day a GPU structure abstraction in CPU) we have to get the
        // pointer in GPU memory in order for the kernel to know where to start 
        // reading the particle array from.
        
        ParticleT* particles_ptr = thrust::raw_pointer_cast(particles.data());
        ContainerT* box_ptr = box.device_ptr();
      
        /* This is the way I structured my blocks and threads. I fixed the amount of
         * threads per block to 1024. So to get the amount of blocks we just get the
         * total number of elements in positions and divide it by 1024. We add one in
         * case the division leaves remainder.
         *
         * ┌──────────────────────grid─┬of─blocks─────────────────┬──────────
         * │     block_of_threads      │     block_of_threads     │  
         * │ ┌───┬───┬───────┬──────┐  │ ┌───┬───┬───────┬──────┐ │
         * │ │ 0 │ 1 │ [...] │ 1023 │  │ │ 0 │ 1 │ [...] │ 1023 │ │   ...
         * │ └───┴───┴───────┴──────┘  │ └───┴───┴───────┴──────┘ │
         * └───────────────────────────┴──────────────────────────┴──────────
         */
        
        unsigned int block_size = 1024;
        dim3 grid_size( n_particles, n_particles / block_size + ( n_particles % block_size == 0 ? 0:1 ) );  
        
        // Launch the kernel! As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel. No need to copy back, we can read from
        // the device vector with the ::operator[]() i.e. positions[2] and that would
        // do all the memory copying for us!
        
        integrator_kernel<<<grid_size,block_size>>>(particles_ptr, box_ptr, n_particles, step);
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
        for (int i=0; i<(n_particles-1); i++) {
            printf("\t");
            print_particle( &(p[i]) );
            printf("\n");
        }

    }
};
