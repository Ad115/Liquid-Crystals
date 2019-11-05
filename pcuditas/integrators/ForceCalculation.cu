#pragma once

#include "pcuditas/gpu/gpu_array.cu"


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
                    auto dr = self_pos - other_pos;

                    auto f_ij = (i != j) ? ParticleT::force_law(dr) : vector_t::zero();
                    force += f_ij;    
                }

                self.force = force;
        });
}