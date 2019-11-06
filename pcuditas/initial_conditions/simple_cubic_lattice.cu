#include "pcuditas/gpu/gpu_array.cu"

template <typename VectorT>
__host__ __device__ 
VectorT cubic_lattice_position(
  size_t particle_idx, 
  double side_length, 
  size_t n_particles) {

    // No. of particles along every side of the cube
    int cube_length = ceil(pow(n_particles, 1./VectorT::dimensions));
    // The lowest integer such that cube_length^DIMENSIONS >= n. 
    // Think of a cube with side cube_length where all particles 
    // are evenly spaced on a simple grid.

    VectorT position;
    for (int D=0; D<position.dimensions; D++) {
         // Get position in a hypercube with volume = cube_length^DIMENSIONS.
        position[D] = ((int)( (particle_idx / pow(cube_length, D)) )%cube_length);
        // Rescale to a box of volume = L^DIMENSIONS
        position[D] *= (side_length/cube_length);  // Make the cube be as big as the box.
    }
    return position;
}

template<class ParticleT>
void arrange_on_cubic_lattice(gpu_array<ParticleT> &particles, double side_length) {
    particles.for_each(
        [side_length, n=particles.size] 
        __device__ (ParticleT &p, size_t idx) {
            using vector_t = typename ParticleT::vector_type;
            p.position = cubic_lattice_position<vector_t>(idx, side_length, n);
    });
}

