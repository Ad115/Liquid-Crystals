#pragma once

#include <pcuditas/gpu/gpu_array.cu>
#include <thrust/random.h>

template <typename VectorT, typename RandomEngine>
__host__ __device__ 
VectorT random_position(unsigned int particle_idx, double side_length, RandomEngine rng) {
    // Create random numbers in the range [0,L)
    thrust::uniform_real_distribution<double> unif(0, side_length);

    VectorT position;
    for (int i=0; i<position.dimensions; i++) {
        float random_value = unif(rng);
        position[i] = random_value;
    }

    return position;
}

template <class ParticleT>
void set_random_positions(gpu_array<ParticleT> &particles, double side_length) {
    particles.for_each(
        [side_length] 
        __device__ (ParticleT &p, size_t idx) {
            // Instantiate random number engine
            thrust::default_random_engine rng(idx*1000 + idx*idx);
            rng.discard(idx);

            // Set particle position        
            using vector_t = typename ParticleT::vector_type;
            p.position = random_position<vector_t>(idx, side_length, rng);
    });
}

class random_positions {
public:
    double side_length;

    random_positions(double L): side_length(L) {};

    template<class ParticleT>
    random_positions(gpu_array<ParticleT> &particles, double L)
        : side_length(L) {
        
        set_random_positions(particles, side_length);
    }

    template<class ParticleT>
    void operator() (gpu_array<ParticleT> &particles) {
        set_random_positions(particles, side_length);
    }
};