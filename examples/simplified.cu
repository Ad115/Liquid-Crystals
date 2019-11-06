#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"

#include "pcuditas/particles/LennardJonesParticle.cu"
#include "pcuditas/integrators/VelocityVertlet.cu"
#include "pcuditas/environments/PeriodicBoundaryBox.cu"
#include "pcuditas/input_output/XYZformat.cu"

#include "pcuditas/initial_conditions/simple_cubic_lattice.cu"
#include "pcuditas/initial_conditions/random.cu"

int n_particles = 40;
double time_step = 0.003;

int n_blocks = 1024;
int threads_per_block = 32;

int main(int argc, char *argv[]) {

    if (argc >= 2) n_blocks = atoi(argv[1]);
    if (argc >= 3) threads_per_block = atoi(argv[2]);

    auto particles = gpu_array<LennardJonesParticle>(n_particles);
    arrange_on_cubic_lattice(particles, 6.);
    set_random_velocities(particles);

    auto move = VelocityVertlet{};
    auto environment = in_gpu(PeriodicBoundaryBox{30.});

    for (int i=0; i < 50000; ++i) {
        move(particles, environment, time_step, 
            n_blocks, threads_per_block);
    }
}
