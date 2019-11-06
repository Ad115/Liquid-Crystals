#pragma once

#include "pcuditas/gpu/gpu_array.cu"

#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <memory>


template <class VectorT>
__device__ 
VectorT random_vector(curandState_t *state) {
    int dimensions = VectorT::dimensions;
    auto result = VectorT{};

    for (int i=0; i<dimensions; i++) {
        result[i] = 2*curand_uniform(state) - 1;
    }

    return result;
}

/* Random numbers in CUDA: 
    http://ianfinlayson.net/class/cpsc425/notes/cuda-random 
*/

__global__ 
void init_random_kernel(
        curandState_t* states,
        size_t n,
        unsigned int seed) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
    {
        curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                    i, /* the sequence number should be different for each core (unless you want all
                                cores to get the same sequence of numbers for some reason - use thread id! */
                    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &states[i]);
    }
}

#define DEFAULT_RANDOM_NUMBERS 2014

class RandomWalk { /*
    * An integrator that adds random positions each step.
    */
    std::unique_ptr<
        gpu_array<curandState_t>> random_state;

public:

    RandomWalk(
            unsigned int seed = 0,
            int random_array_size = DEFAULT_RANDOM_NUMBERS) {
        
        if (seed == 0) 
            seed = time(0);
        
        initialize_random_state(random_array_size);
    }

    template <class ParticleT>
    void operator()(gpu_array<ParticleT> &particles) {

        using vector_t = typename ParticleT::vector_type;

        particles.for_each(
            [rand_state=random_state->gpu_pointer(),
             rand_size=random_state->size] 
            __device__ 
            (ParticleT &p, size_t i) {
                auto delta = random_vector<vector_t>(&rand_state[i % rand_size]);
                p.position += delta.unit_vector();
            }
        );
    }

    template <class ParticleT, class EnvironmentT>
    void operator()(
            gpu_array<ParticleT> &particles, 
            gpu_object<EnvironmentT> &env) {
        
        // Apply usual integration with no environment
        (*this)(particles);

        // Apply boundary conditions
        particles.for_each(
            [environment=env.gpu_pointer()] 
            __device__ 
            (ParticleT &p, size_t idx) {
                p.position = environment->apply_boundary_conditions(p.position);
            }
        );
    }

    void initialize_random_state(size_t n) {
        random_state.reset(new gpu_array<curandState_t>(n));

        auto seed = time(0);
        init_random_kernel<<<128,32>>>(
            random_state->gpu_pointer(), n,
            seed
        );
    }
};

/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "tests/doctest.h"
#include <typeinfo>   // operator typeid

#include "pcuditas/initial_conditions/move_to_origin.cu"
#include "pcuditas/particles/SimpleParticle.cu"
#include "pcuditas/vectors/EuclideanVector.cu"

TEST_SUITE("Random Walk specification") {

    SCENARIO("Description") {
        GIVEN("A GPU array of 1D-particles at the origin") {

            using vector_t = EuclideanVector<1>;
            using particle_t = Particle<vector_t>;

            auto particles = gpu_array<particle_t>(100);
            move_to_origin(particles);

            WHEN("A RandomWalk object is used to move them") {
                auto move = RandomWalk{};
                int steps = 1000;
                for (int i=0; i<steps; i++) {
                    move(particles);
                }

                THEN("The radius of the particle cloud increases as sqrt(t)") {
                    particles.to_cpu();
                    double r_mag_sum = 0;
                    auto r_sum = vector_t::zero();
                    for(auto p : particles) {
                        r_mag_sum += p.position.magnitude();
                        r_sum += p.position;
                    }

                    auto r_average = r_mag_sum / particles.size;
                    CHECK(r_average > 0);
                    CHECK(r_average < steps); // Bounded by 0 and t

                    CHECK(r_sum.magnitude() < steps); // not larger than t
                }
            }
        }
    }
}

#endif