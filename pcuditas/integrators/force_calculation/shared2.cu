#pragma once

#include "pcuditas/gpu/gpu_array.cu"

template <class ContainerT>
struct force_calculation {
    ContainerT *box;

    __host__ __device__
    force_calculation(ContainerT *box_) 
        : box(box_) {}

    template <typename ParticleT>
    using Vector_t = typename ParticleT::vector_type;
    
    template <typename ParticleT>
    __host__ __device__
    Vector_t<ParticleT> operator()(ParticleT &p1, ParticleT &p2) {
        // Calculate force
        return ParticleT::interaction::interaction_force(p1, p2, *box);
    }
};



template<class ParticleT, class ForceFn, class ForceT, class ForceSaveFn>
__global__ 
void update_forces_shared2_kernel(
        ParticleT *particles, size_t n_particles, 
        ForceFn force_fn, ForceT zero_force,
        ForceSaveFn force_save_fn) {

    extern __shared__ ForceT forces_sh[];

    for (int i = blockIdx.x;
         i < n_particles; 
         i += gridDim.x) {

        auto self = particles[i];

        // Initialize force to zero
        forces_sh[threadIdx.x] = zero_force;
        __syncthreads();
        __threadfence();


        // Move the block through the columns
        for (int j=0; j<n_particles; j += blockDim.x) {

            // Each thread in the block calculates the 
            // interaction between particle i and (j + tid) 
            int other_idx = j + threadIdx.x;
            auto other = particles[other_idx];

            auto force = zero_force;
            if (other_idx < n_particles && i != other_idx) {
                force = force_fn(self, other);
            }

            if (other_idx < n_particles)

            // Save in shared memory
            forces_sh[threadIdx.x] += force;
            __syncthreads();
        }

        // Reduce the partial results in shared memory
        for (int working_threads = blockDim.x/2; 
             working_threads > 0; 
             working_threads /= 2) {

            if (threadIdx.x < working_threads) {
                forces_sh[threadIdx.x] += forces_sh[threadIdx.x + working_threads];
            }
            __syncthreads();
        }

        // Save the final result
        if (threadIdx.x == 0) {
            force_save_fn(particles[i], forces_sh[0], i);
        }
    }
}


template<class ParticleT, class ForceFn, class ForceT, class ForceSaveFn>
void update_forces_shared2(
            gpu_array<ParticleT> &particles,
            ForceFn force_fn,
            ForceT zero_force,
            ForceSaveFn force_save_fn,
            unsigned int block_size = 512,
            unsigned int threads_per_block = 64) {

        unsigned int shared_memory_size = (
            threads_per_block * sizeof(ForceT)
        );
        update_forces_shared2_kernel<<<block_size, threads_per_block, shared_memory_size>>>(
            particles.gpu_pointer(), particles.size, 
            force_fn, zero_force, force_save_fn
        );
}


/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "tests/doctest.h"
#include <typeinfo>   // operator typeid


TEST_SUITE("Force calculation specification") {

    SCENARIO("Description") {
        GIVEN("A GPU array of particles") {

            using vector_t = double;
            using particle_t = double;
            int n = 100000;

            // Inicializa en 0, 1, 2, ... n
            auto particles = gpu_array<particle_t>(n,
                [] __device__ (particle_t &p, int idx) {
                    p = idx;
            });

            WHEN("The forces are calculated") {
                auto forces = gpu_array<vector_t>(n);

                // Fuerza p[i] y p[j] = i*j
                auto force_fn 
                    = [] __device__ 
                      (particle_t &p1, particle_t &p2) {
                        return p1 * p2;
                    };

                // Guarda en forces
                auto force_save_fn
                    = [forces_gpu=forces.gpu_pointer()] 
                      __device__
                      (particle_t &p, vector_t &force, int idx) {
                          forces_gpu[idx] = force;
                      };

                update_forces_shared2(particles, force_fn, 0., force_save_fn);

                THEN("The force is calculated correctly") {
                    forces.to_cpu();

                    for(int i=0; i<n; i++) {
                        vector_t f = forces[i];
                        CAPTURE(i);
                        CHECK(f == doctest::Approx(i*( n*(n-1.)/2. - i ) ));
                    }
                }
            }
        }
    }
}

#endif