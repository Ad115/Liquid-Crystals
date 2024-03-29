#pragma once

#include "pcuditas/gpu/gpu_array.cu"

template<class ParticleT>
void move_to_origin(gpu_array<ParticleT> &particles) { /*
    * Set the positions of all the particles to the origin.
    */

    using vector_t = typename ParticleT::vector_type;

    particles.for_each([] __device__ (ParticleT &p, size_t idx) {
        p.position = vector_t::zero();
    });
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

TEST_SUITE("move_to_origin transformation specification") {

    SCENARIO("Usage") {

        GIVEN("A GPU array of particles at arbitrary positions") {

            using particle_t = SimpleParticle;
            using vector_t = particle_t::vector_type;

            auto particles 
                = gpu_array<particle_t>(100,
                    [] __device__ (particle_t &p, size_t idx) {
                        // Initial positions at distinct arbitrary unit vectors
                        p.position = {
                            sin(idx)*cos(idx*idx), 
                            sin(idx)*sin(idx*idx),
                            cos(idx)};
                    });

            // Verify no vector is zero
            // --> on device
            particles.for_each([] __device__ (particle_t &p, size_t idx) {
                assert(p.position.magnitude() > 0);
            });

            // --> on host
            particles.to_cpu();
            for(auto p : particles) {
                REQUIRE(p.position.magnitude() > 0);
            }

            WHEN("The transformation is applied to the GPU particles") {

                move_to_origin(particles);
                
                THEN("The transformation sets all positions to the zero vector") {

                    // Verify all vectors are zero
                    // --> on device
                    particles.for_each([] __device__ (particle_t &p, size_t idx) {
                        assert(p.position.magnitude() == 0);
                    });


                    // --> on host
                    particles.to_cpu();
                    for(auto p : particles) {
                        CHECK(p.position == vector_t::zero());
                    }
                }
            }
        }
    }
}
#endif