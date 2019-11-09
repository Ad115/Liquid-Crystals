#pragma once

#include "pcuditas/gpu/gpu_array.cu"

template<class ParticleT, class EnvironmentT>
__global__ 
void update_forces_shared2_kernel(
        ParticleT *particles, int n_particles, 
        EnvironmentT *env_ptr) {

    using vector_t = typename ParticleT::vector_type;
    extern __shared__ vector_t forces_sh[];

    EnvironmentT env = (*env_ptr);

    for (int i = blockIdx.x;
         i < n_particles; 
         i += gridDim.x) {

        auto self_pos = particles[i].position;

        // Initialize force to zero
        forces_sh[threadIdx.x] = vector_t::zero();
        __syncthreads();
        __threadfence();


        // Move the block through the columns
        for (int j=0; j<n_particles; j += blockDim.x) {

            // Each thread in the block calculates the 
            // interaction between particle i and (j + tid) 
            int other_idx = j + threadIdx.x;

            auto force = vector_t::zero();
            if (other_idx < n_particles) {

                // Calculate force
                auto other_pos = particles[other_idx].position;
                auto dr = env.distance_vector(other_pos, self_pos);

                force = (i != other_idx) 
                    ? ParticleT::force_law(dr) 
                    : vector_t::zero();
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


template<class ParticleT, class EnvironmentT>
void update_forces_shared2(
            gpu_array<ParticleT> &particles,
            gpu_object<EnvironmentT> &env,
            unsigned int block_size = 512,
            unsigned int threads_per_block = 64) {
        
        // Launch the kernel. As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel.

        unsigned int shared_memory_size = (
            threads_per_block * sizeof(typename ParticleT::vector_type)
        );
        update_forces_shared2_kernel<<<block_size, threads_per_block, shared_memory_size>>>(
            particles.gpu_pointer(), particles.size, 
            env.gpu_pointer()
        );
}
