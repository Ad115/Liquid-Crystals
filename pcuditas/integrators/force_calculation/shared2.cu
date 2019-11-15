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
    Vector_t<ParticleT> operator()(ParticleT p1, ParticleT p2) {
        // Calculate force
        return p1.interaction_force_with(p2, *box);
    }


};

template<class ParticleT, class ForceFn, class ForceT>
__global__ 
void update_forces_shared2_kernel(
        ParticleT *particles, int n_particles, 
        ForceFn force_fn, ForceT zero_force) {

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
            auto other = particles[j];

            auto force = zero_force;
            if (other_idx < n_particles && i != other_idx) {
                force = force_fn(self, other);
            }

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
        if (threadIdx.x == 0)
            particles[i].force = forces_sh[0];
    }
}


template<class ParticleT, class ForceFn, class ForceT>
void update_forces_shared2(
            gpu_array<ParticleT> &particles,
            ForceFn force_fn,
            ForceT zero_force,
            unsigned int block_size = 512,
            unsigned int threads_per_block = 64) {

        unsigned int shared_memory_size = (
            threads_per_block * sizeof(ForceT)
        );
        update_forces_shared2_kernel<<<block_size, threads_per_block, shared_memory_size>>>(
            particles.gpu_pointer(), particles.size, 
            force_fn, zero_force
        );
}
