#include <thrust/device_vector.h>

#include "Particle.cu"
#include "Vector.cu"


// This is the kernel that is launched from CPU and GPU runs it for each cell
template <typename ParticleT>
__global__ 
void kernel(ParticleT *particles, int n) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n)
        particles[index].position += 0.5;
}

template< typename ParticleT=Particle<> >
class ParticleSystem
{
    unsigned int n_particles;
    thrust::device_vector<ParticleT> particles;

  public:
    ParticleSystem(unsigned int n) 
        : n_particles{n},
          particles{thrust::device_vector<ParticleT>(n)} {};

    void simulation_step() {
        // As we cannot send device vectors to the kernel (as device_vector is at
        // the end of the day a GPU structure abstraction in CPU) we have to get the
        // pointer in GPU memory in order for the kernel to know where to start 
        // reading the particle array from.
        ParticleT* particles_ptr = thrust::raw_pointer_cast(particles.data());
      
        /* This is the way I structured my blocks and threads. I fixed the amount of
         * threads per block to 1024. So to get the amount of blocks we just get the
         * total number of elements in positions and divide it by 1024. We add one in
         * case the division leaves remainder.
         *
         * ┌──────────────────────grid─┬of─blocks─────────────────┬──────────
         * │     block_of_threads      │     block_of_threads     │  
         * │ ┌───┬───┬───────┬──────┐  │ ┌───┬───┬───────┬──────┐ │
         * │ │ 0 │ 1 │ [...] │ 1023 │  │ │ 0 │ 1 │ [...] │ 1023 │ │   ...
         * │ └───┴───┴───────┴──────┘  │ └───┴───┴───────┴──────┘ │
         * └───────────────────────────┴──────────────────────────┴──────────
         */
        unsigned int block_size = 1024;
        unsigned int grid_size = n_particles / block_size + 1;
      
        // Launch the kernel! As you can see we are not copying memory from CPU to GPU
        // as you would normally do with cudaMemcpy(), as we don't need to! The
        // vectors live in GPU already so we just need to know where they start (GPU
        // pointer) and pass it to the kernel. No need to copy back, we can read from
        // the device vector with the ::operator[]() i.e. positions[2] and that would
        // do all the memory copying for us!
        kernel<<<grid_size,block_size>>>(particles_ptr, n_particles);
    }

    void print() {
        thrust::host_vector<ParticleT> p(particles);

        printf("Particles: \n");
        for (int i=0; i<(n_particles-1); i++) {
            printf("\t");
            print_particle(&(p[i]));
            printf("\n");
        }

    }
};