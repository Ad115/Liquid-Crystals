#pragma once

#include <pcuditas/gpu/gpu_array.cu>
#include <thrust/random.h>

template <typename VectorT, typename RandomEngine>
__host__ __device__ 
VectorT random_vector(
        unsigned int particle_idx, 
        RandomEngine rng, 
        double min = 0., double max = 1.) {
    // Create random numbers in the range [0,L)
    thrust::uniform_real_distribution<double> unif(min, max);

    VectorT v;
    for (int i=0; i<v.dimensions; i++) {
        v[i] = unif(rng);
    }

    return v;
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
            p.position = random_vector<vector_t>(idx, rng, 0, side_length);
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

// ---

template <class ParticleT>
void set_random_velocities(gpu_array<ParticleT> &particles, double max_speed) {
    particles.for_each(
        [max_speed] 
        __device__ (ParticleT &p, size_t idx) {
            // Instantiate random number engine
            thrust::default_random_engine rng(idx*1000 + idx*idx);
            rng.discard(idx);

            // Set particle position        
            using vector_t = typename ParticleT::vector_type;
            p.velocity = random_vector<vector_t>(idx, rng, -max_speed, max_speed);
    });
}

class random_velocities {
public:
    double max_speed;

    random_velocities(double s): max_speed(s) {};

    template<class ParticleT>
    random_velocities(gpu_array<ParticleT> &particles, double s)
        : max_speed(s) {
        
        set_random_velocities(particles, s);
    }

    template<class ParticleT>
    void operator() (gpu_array<ParticleT> &particles) {
        set_random_velocities(particles, max_speed);
    }
};