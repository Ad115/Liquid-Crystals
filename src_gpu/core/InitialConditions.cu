#include "Transformations.cu"

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

template <typename VectorT>
__host__ __device__ 
VectorT cubic_lattice_position(
  unsigned int particle_idx, 
  double side_length, 
  int n_particles) {

    // No. of particles along every side of the cube
    long cube_length = ceil(pow(n_particles, 1./VectorT::dimensions));
    // The lowest integer such that cube_length^DIMENSIONS >= n. 
    // Think of a cube with side cube_length where all particles 
    // are evenly spaced on a simple grid.

    VectorT position;
    for (int D=0; D<position.dimensions; D++) {
         // Get position in a hypercube with volume = cube_length^DIMENSIONS.
        position[D] = ( (int)(particle_idx / pow(cube_length, D)) % cube_length );
        // Rescale to a box of volume = L^DIMENSIONS
        position[D] *= (side_length/cube_length) * 0.9;  // Make the cube be as big as the box.
                                                // ^^ This last factor rescales the cube.
    }
    return position;
}

template <typename VectorT, typename RandomEngine>
__host__ __device__ 
VectorT random_velocity(RandomEngine rng) {

    thrust::normal_distribution<double> norm(0., 1.);

    VectorT velocity;
        for (int i=0; i<velocity.dimensions; i++) {
            float random_value = norm(rng);
            velocity[i] = random_value;
        }
    return velocity;
}

template <typename ParticleT, typename ContainerT>
__global__                                                       
void init_kernel(ParticleT *particles, int n, ContainerT *box) {

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n){

        double L = (*box).side_length;
        using vector_type = typename ParticleT::vector_type;

        // Set particle position        
        //auto position = random_position<vector_type>(index, L, rng);
        auto position = cubic_lattice_position<vector_type>(index, L, n);
        particles[index].position = (*box).apply_boundary_conditions(position);
        

        // Instantiate random number engine
        thrust::default_random_engine rng(index*1000 + index*index);
        rng.discard(index);

        // Set particle velocity
        particles[index].velocity = random_velocity<vector_type>(rng);
    }
}

class initial_conditions: public Transformation {

public:

    template<typename ParticleSystemT>
    void operator()(ParticleSystemT& s) {

        using particle_t = typename ParticleSystemT::particle_type;
        using container_t = typename ParticleSystemT::container_type;
        
        particle_t* particles_ptr = thrust::raw_pointer_cast(s.particles.data());
        container_t* box_ptr = s.box.device_ptr();

        unsigned int block_size = 1024;
        unsigned int grid_size = s.n_particles / block_size + 1;
        
        init_kernel<<<grid_size,block_size>>>(particles_ptr, s.n_particles, box_ptr);
    }
};