#ifndef VELOCITY_VERTLET_INTEGRATOR_HEADER
#define VELOCITY_VERTLET_INTEGRATOR_HEADER

#include "core/interfaces/Integrator.h"


template <typename ParticleT, typename ContainerT>
__global__ 
void update_forces_kernel(ParticleT *particles, int n_particles, ContainerT *box);


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

        // v(t + 1/2*dt) = v(t) + 1/2*f(t)*dt
        update_velocities(system);

        // r(t + dt) = r(t) + v(t+1/2dt)*dt
        update_positions(system);

        // r(t + dt)  -->  f(t + dt)
        update_forces(system);

        // v(t + dt) = v(t + 1/2*dt) + 1/2*f(t + dt)*dt
        update_velocities(system);
    }

    template<typename SystemT>
    void update_positions(SystemT& system) {

        auto box_ptr = system.box.device_ptr();
        
        using particle_t = typename SystemT::particle_type;

        system.map_to_particles(
            [box_ptr, dt=time_step] __device__ (particle_t& p){

                // r(t + dt) = r(t) + v(t + 1/2dt)*dt
                auto new_pos = p.position + p.velocity*dt;

                p.position = (*box_ptr).apply_boundary_conditions(new_pos); 
            }
        );
    }

    template<typename SystemT>
    void update_velocities(SystemT& system) {

        auto box_ptr = system.box.device_ptr();
        
        using particle_t = typename SystemT::particle_type;

        system.map_to_particles(
            [box_ptr, dt=time_step] __device__ (particle_t& p){

                // v(t + dt) = v(t) + 1/2*f(t + dt)*dt
                p.velocity = p.velocity + 0.5*p.force*dt;
            }
        );
    }
    
    template<typename SystemT>
    void update_forces(SystemT& s) {

        auto box_ptr = s.box.device_ptr();
        using particle_t = typename SystemT::particle_type;
        using vector_t = typename particle_t::vector_type;

        // First, reset forces
        s.map_to_particles(
            [box_ptr] __device__ (particle_t& p){

                p.force = vector_t::null();
            }
        );

        // As we cannot send device vectors to the kernel (as device_vector is at
        // the end of the day a GPU structure abstraction in CPU) we have to get the
        // pointer to GPU memory in order for the kernel to know where to start 
        // reading the particle array from.
        
        auto particles_ptr = thrust::raw_pointer_cast(s.particles.data());
        auto n_particles = s.n_particles;
        
        // Launch the kernel! As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel. No need to copy back, we can read from
        // the device vector with the ::operator[]() i.e. poarticles[2] and that would
        // do all the memory copying for us!

        unsigned int block_size = 1024;
        unsigned int threads_per_block = 1024;
        update_forces_kernel<<<block_size,threads_per_block>>>(
            particles_ptr, n_particles, 
            box_ptr
        );
    }

    template<typename SystemT>
    void operator()(SystemT& system) {
        integration_step(system);
    }

};


template<class vector_t>
__device__ double magnitude(vector_t v) {
    return sqrt(v*v);
}

template <typename ParticleT, typename ContainerT>
__global__ 
void update_forces_kernel(ParticleT *particles, int n_particles, ContainerT *box) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n_particles*n_particles; 
        i += blockDim.x * gridDim.x) {

        int row = i/n_particles;
        int column = i % n_particles;

        if (column > row) {
        
            double cutoff_radius = 3.5;
            auto dr = (*box).distance_vector(
                particles[row].position, 
                particles[column].position
            );

            if (magnitude(dr) < cutoff_radius) {
                auto force = particles[row]
                    .interaction_force_with(particles[column], *box);

                for( int i=0; i<force.dimensions; ++i ){
                    atomicAdd( &(particles[row].force[i]), force[i] );
                    atomicAdd( &(particles[column].force[i]), -force[i] );
                }  
            }
        }
    }
}


#endif