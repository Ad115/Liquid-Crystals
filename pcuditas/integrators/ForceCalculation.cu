#pragma once

#include "pcuditas/gpu/gpu_array.cu"


template<class ParticleT>
void update_forces_naive(gpu_array<ParticleT> &particles) {

        using vector_t = typename ParticleT::vector_type;

        // Naïve paralellization.
        particles.for_each(
            [others=particles.gpu_pointer(), n=particles.size] 
            __device__ (ParticleT& self, int i) {
                auto force = vector_t::zero();
                auto self_pos = self.position;

                for(int j=0; j<n; j++) {
                    auto other_pos = others[j].position;
                    auto dr = self_pos - other_pos;

                    auto f_ij = (i != j) ? ParticleT::force_law(dr) : vector_t::zero();
                    force += f_ij;    
                }

                self.force = force;
        });
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

template <typename ParticleT>
__global__ 
void update_forces_atomic_kernel(ParticleT *particles, int n_particles) {

    for (int k = blockIdx.x*blockDim.x + threadIdx.x; 
         k < n_particles*n_particles; 
         k += blockDim.x*gridDim.x) {

        int i = k % n_particles;
        int j = k/n_particles;
        
        double cutoff_radius = 3.5;
        auto dr = (
            particles[i].position - particles[j].position
        );

        if (dr.magnitude() < cutoff_radius) {
            auto force = ParticleT::force_law(dr);

            for( int d=0; d<force.dimensions; ++d ){
                atomicAddDouble( &(particles[i].force[d]), force[d] );
            }  
        }
    }
}

template<class ParticleT>
void update_forces_atomic(gpu_array<ParticleT> &particles) {

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
            particles.gpu_pointer(), particles.size
        );
}

