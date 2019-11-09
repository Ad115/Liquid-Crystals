#pragma once

#include "pcuditas/gpu/gpu_array.cu"

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