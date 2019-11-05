#pragma once

#include "pcuditas/gpu/gpu_array.cu"

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
void update_forces_kernel(ParticleT *particles, int n_particles);

template<class ParticleT>
void update_forces(gpu_array<ParticleT> &particles) {

        using vector_t = typename ParticleT::vector_type;

        // Naïve paralellization.
        particles.for_each(
            [others=particles.gpu_pointer(), n=particles.size] 
            __device__ (ParticleT& self, int i) {
                auto force = vector_t::zero();
                auto self_pos = self.position;

                for(int j=0; j<n; j++) {
                    auto other_pos = others[j].position;
                    auto dr = other_pos - self_pos;
                    auto f_ij = ParticleT::force_law(dr);

                    force += (i != j) ? f_ij : vector_t::zero();    
                }

                self.force = force;
        });
}

template <typename ParticleT>
__global__ 
void update_forces_kernel(ParticleT *particles, int n_particles) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n_particles*n_particles; 
         i += blockDim.x * gridDim.x) {

        int row = i/n_particles;
        int column = i % n_particles;

        if (column > row) {
        
            double cutoff_radius = 3.5;
            auto dr = (
                particles[row].position - particles[column].position
            );

            if (dr.magnitude() < cutoff_radius) {
                auto force = ParticleT::force_law(dr);

                for( int i=0; i<force.dimensions; ++i ){
                    atomicAddDouble( &(particles[row].force[i]), force[i] );
                    atomicAddDouble( &(particles[column].force[i]), -force[i] );
                }  
            }
        }
    }
}