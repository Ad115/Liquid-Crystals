#pragma once

#include <stdlib.h>
#include <pcuditas/gpu/gpu_array.cu>
#include <pcuditas/cpu/cpu_array.cu>
#include <thrust/random.h>


template <typename VectorT>
__device__ 
VectorT random_vector_gpu(
        unsigned int particle_idx, 
        double min = 0., double max = 1.) {

    // Instantiate random number engine
    thrust::default_random_engine rng(particle_idx*particle_idx);
    rng.discard(particle_idx);
    
    // Create random numbers in the range [0,L)
    thrust::uniform_real_distribution<double> unif(min, max);

    VectorT v;
    for (int i=0; i<v.dimensions; i++) {
        v[i] = unif(rng);
    }

    return v;
}

template <typename VectorT>
__host__ 
VectorT random_vector_cpu(
        unsigned int particle_idx, 
        double min = 0., double max = 1.) {

    VectorT v;
    for (int i=0; i<v.dimensions; i++) {
        v[i] = (max-min)*drand48() + min;
    }

    return v;
}

template <class ParticleT>
void set_random_positions(gpu_array<ParticleT> &particles, double side_length) {

    using vector_t = typename ParticleT::vector_type;

    particles.for_each(
        [side_length] 
        __device__ (ParticleT &p, size_t idx) {

            // Set particle position        
            p.position = random_vector_gpu<vector_t>(idx, 0, side_length);
    });
}

template <class ParticleT>
void set_random_positions(cpu_array<ParticleT> &particles, double side_length) {

    srand48(time(0));
    using vector_t = typename ParticleT::vector_type;

    particles.for_each(
        [side_length] 
         (ParticleT &p, size_t idx) {

            // Set particle position        
            p.position = random_vector_cpu<vector_t>(idx, 0, side_length);
    });
}


// ---

template <class ParticleT>
void set_random_velocities(gpu_array<ParticleT> &particles, double max_speed=1) {

    using vector_t = typename ParticleT::vector_type;

    particles.for_each(
        [max_speed] 
        __device__ (ParticleT &p, size_t idx) {

            // Set particle velocity        
            p.velocity = random_vector_gpu<vector_t>(idx, -max_speed, max_speed);
    });
}

template <class ParticleT>
void set_random_velocities(cpu_array<ParticleT> &particles, double max_speed=1) {

    srand48(time(0));
    using vector_t = typename ParticleT::vector_type;

    particles.for_each(
        [max_speed] 
        (ParticleT &p, size_t idx) {

            // Set particle position        
            p.velocity = random_vector_cpu<vector_t>(idx, -max_speed, max_speed);
    });
}