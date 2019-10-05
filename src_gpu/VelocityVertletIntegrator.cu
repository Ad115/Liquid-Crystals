#ifndef VELOCITY_VERTLET_INTEGRATOR_HEADER
#define VELOCITY_VERTLET_INTEGRATOR_HEADER

#include "core/interfaces/Integrator.h"

template <typename ParticleT, typename ContainerT>
__global__ 
void force_kernel(ParticleT *particles, int n_particles, ContainerT *box);

template <typename ParticleT, typename ContainerT>
__global__                                                       
void first_half_kernel(ParticleT *particles, int n_particles, ContainerT *box, double dt);

template <typename ParticleT, typename ContainerT>
__global__                                                       
void second_half_kernel(ParticleT *particles, int n_particles, ContainerT *box, double dt);


class VelocityVertlet: Integrator {

public:
    double time_step = 0.001;

    template<typename SystemT>
    void integration_step(SystemT& system) { /*
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
        first_half_step(system);

        // r(t + dt)  -->  f(t + dt)
        update_forces(system);

        // v(t + dt) = v(t + 1/2*dt) + 1/2*f(t + dt)*dt
        second_half_step(system);
    }

    template<typename SystemT>
    void first_half_step(SystemT& s) {

        auto particles_ptr = thrust::raw_pointer_cast(s.particles.data());
        auto box_ptr = s.box.device_ptr();
        auto n_particles = s.n_particles;

        unsigned int block_size = 1024;
        unsigned int grid_size = n_particles / block_size + 1;
        
        first_half_kernel<<<grid_size,block_size>>>(
            particles_ptr, n_particles, 
            box_ptr, 
            time_step
        );
    }

    template<typename SystemT>
    void second_half_step(SystemT& s) {
        
        auto particles_ptr = thrust::raw_pointer_cast(s.particles.data());
        auto box_ptr = s.box.device_ptr();
        auto n_particles = s.n_particles;

        unsigned int block_size = 1024;
        unsigned int grid_size = n_particles / block_size + 1;
        
        second_half_kernel<<<grid_size,block_size>>>(
            particles_ptr, n_particles, 
            box_ptr, 
            time_step
        );
    }

    
    template<typename SystemT>
    void update_forces(SystemT& s) {
        // As we cannot send device vectors to the kernel (as device_vector is at
        // the end of the day a GPU structure abstraction in CPU) we have to get the
        // pointer to GPU memory in order for the kernel to know where to start 
        // reading the particle array from.
        
        auto particles_ptr = thrust::raw_pointer_cast(s.particles.data());
        auto box_ptr = s.box.device_ptr();
        auto n_particles = s.n_particles;
      
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
        // the device vector with the ::operator[]() i.e. poarticles[2] and that would
        // do all the memory copying for us!

        // Update forces
        force_kernel<<<grid_size,block_size>>>(
            particles_ptr, n_particles, 
            box_ptr
        );
    }

    template<typename SystemT>
    void operator()(SystemT& system) {
        integration_step(system);
    }

};

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

#endif