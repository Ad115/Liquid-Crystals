#pragma once

#include <assert.h>

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

                    auto r = dr.magnitude();
                    auto expected = (i != j) ? abs(48*(pow(r, -12) - 0.5*pow(r, -6))/r) : 0.;
                    auto actual = f_ij.magnitude();
                    if (abs(expected - actual) > 0.001*expected)
                    printf("%d, %d -- Expected: %f, Actual: %f, Diff: %f\n", i,j,
                           expected,
                           actual,
                           expected - actual);

                    force += f_ij;    
                }

                self.force = force;
        });
}