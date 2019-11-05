#pragma once

#include "pcuditas/gpu/gpu_array.cu"


template<class ParticleT, class EnvironmentT>
__global__ 
void update_forces_shared_kernel(
        ParticleT *particles, int n_particles, 
        EnvironmentT *env_ptr) {

    extern __shared__ ParticleT particles_sh[];
    using vector_t = typename ParticleT::vector_type;
    EnvironmentT env = (*env_ptr);

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; 
         i < n_particles; 
         i += blockDim.x*gridDim.x) {
        
        auto force = vector_t::zero();
        auto self_pos = particles[i].position;

        // For every other particle
        for (int j=0; j<n_particles; j += blockDim.x) {
            // Copy to shared memory
            particles_sh[threadIdx.x] = particles[j + threadIdx.x];
            __syncthreads();

            // Reduce on block
            for(size_t k=0; k<blockDim.x; k++) {
                auto other_pos = particles[k].position;
                auto dr = env.distance_vector(other_pos, self_pos);

                auto f_ij = (i != k) ? ParticleT::force_law(dr) : vector_t::zero();
                force += f_ij;   
            }
            __syncthreads();
        }

        // Save the results
        particles[i].force = force;
    }
}

template<class ParticleT, class EnvironmentT>
void update_forces_shared(
            gpu_array<ParticleT> &particles,
            gpu_object<EnvironmentT> &env,
            unsigned int block_size = 1024,
            unsigned int threads_per_block = 32) {
        
        // Launch the kernel. As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel.

        unsigned int shared_memory_size = threads_per_block * sizeof(ParticleT);
        update_forces_shared_kernel<<<block_size, threads_per_block, shared_memory_size>>>(
            particles.gpu_pointer(), particles.size, 
            env.gpu_pointer()
        );
}


// ===   ===   ===

template<class ParticleT, class EnvironmentT>
void update_forces_naive(
        gpu_array<ParticleT> &particles,
        gpu_object<EnvironmentT> &environment) {

    using vector_t = typename ParticleT::vector_type;

    // Naïve paralellization.
    particles.for_each(
        [others=particles.gpu_pointer(), n=particles.size, 
         env_ptr=environment.gpu_pointer()] 
        __device__ (ParticleT& self, int i) {
            auto force = vector_t::zero();
            auto self_pos = self.position;
            for(int j=0; j<n; j++) {
                auto other_pos = others[j].position;
                auto dr = env_ptr->distance_vector(other_pos, self_pos);
                
                auto f_ij = (i != j) ? ParticleT::force_law(dr) : vector_t::zero();
                force += f_ij;    
            }
        
            self.force = force;
        }
    );
}


// ===  ===  === 


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template<class ParticleT, class EnvironmentT>
__global__ 
void update_forces_atomic_kernel(
        ParticleT *particles, int n_particles,
        EnvironmentT *env_ptr) {

    EnvironmentT env = (*env_ptr);

    for (int k = blockIdx.x*blockDim.x + threadIdx.x; 
         k < n_particles*n_particles; 
         k += blockDim.x*gridDim.x) {

        int i = k % n_particles;
        int j = k/n_particles;
        
        double cutoff_radius = 3.5;
        auto dr = env.distance_vector(
            particles[j].position,
            particles[i].position 
        );

        if (dr.magnitude() < cutoff_radius) {
            auto force = ParticleT::force_law(dr);

            for( int d=0; d<force.dimensions; ++d ){
                atomicAddDouble( &(particles[i].force[d]), force[d] );
            }  
        }
    }
}

template<class ParticleT, class EnvironmentT>
void update_forces_atomic(
        gpu_array<ParticleT> &particles,
        gpu_object<EnvironmentT> &environment) {

        using vector_t = typename ParticleT::vector_type;

        // First, reset forces
        particles.for_each([] __device__ (ParticleT& self, int i){
                self.force = vector_t::zero();
        });
        
        // Launch the kernel! As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel.

        unsigned int block_size = 1024;
        unsigned int threads_per_block = 32;
        update_forces_atomic_kernel<<<block_size,threads_per_block>>>(
            particles.gpu_pointer(), particles.size,
            environment.gpu_pointer()
        );
}

