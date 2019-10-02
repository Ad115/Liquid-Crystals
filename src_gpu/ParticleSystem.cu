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
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <iostream>
#include "Particle.cu"
#include "Vector.cu"
#include "Container.cu"
#include "device_obj.cu"
#include "Transformations.cu"


template <typename VectorT, typename RandomEngine>
__host__ __device__ 
VectorT random_position(unsigned int particle_idx, double side_length, RandomEngine rng) {
    // Create random numbers in the range [0,L)
    thrust::uniform_real_distribution<double> unif(0, side_length);

    VectorT position;
    for (int i=0; i<position.dimensions; i++) {
        float random_value = unif(rng);
        position[i] = random_value;
    }

    return position;
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
        
        double cutoff_radius = 3.5;
        auto dr = box->distance_vector(
            particles[row].position, 
            particles[column].position
        );

        if ((dr * dr) < (cutoff_radius * cutoff_radius)) {
            auto force = particles[row]
                .interaction_force_with(particles[column], *box);

            for( int i=0; i<force.dimensions; ++i ){
                atomicAdd( &particles[row].force[i], force[i] );
                atomicAdd( &particles[column].force[i], -force[i] );
            }  
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

 public:
    thrust::device_vector< ParticleT > particles;
    device_obj< ContainerT > box;

    unsigned int n_particles;
    
    using particle_type = ParticleT;
    using container_type = ContainerT;
    using vector_type = typename ParticleT::vector_type;
    static constexpr int dimensions = ParticleT::dimensions;

    ParticleSystem(unsigned int n, double numeric_density) 
        : n_particles{n},
          particles{thrust::device_vector<ParticleT>(n)},
          box{pow(n/numeric_density, 1./dimensions)} {};

    template <typename MeasurementUnit, typename MeasureFn>
    MeasurementUnit measure_particles(MeasureFn measure_fn) {
        return 
            thrust::transform_reduce(
                    particles.begin(), particles.end(), 
                    measure_fn,
                    0.,
                    thrust::plus<MeasurementUnit>{}
            );
    }

    template <typename ParticleFn>
    ParticleFn map_to_particles(ParticleFn particle_fn) {
        
        thrust::for_each(
            particles.begin(), particles.end(),
            particle_fn
        );

        return particle_fn;
    }

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
        initial_conditions::apply(*this);
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

    void write_xyz(std::ostream& stream) { /*
        * Output the positions of the particles in the XYZ format.
        * The format consists in a line with the number of particles,
        * then a comment line followed by the space-separated coordinates 
        * of each particle in different lines.
        * 
        * Example (for 3 particles in the xyz diagonal):
        * 
        *   10
        *   
        *   1.0 1.0 1.0
        *   1.5 1.5 1.5
        *   2.0 2.0 2.0
        */
        
        thrust::host_vector<ParticleT> host_particles = particles;
        stream << n_particles << "\n";
        for (ParticleT p: host_particles) {
            stream << "\n";
            for (int D = 0; D < dimensions; D++)
                stream << p.position[D] << " ";
        }
        stream << std::endl;
        return;
    };
};

template<typename ParticleT, typename ContainerT>
constexpr int ParticleSystem<ParticleT, ContainerT>::dimensions;