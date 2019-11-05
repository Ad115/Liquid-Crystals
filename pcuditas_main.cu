#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/particles/SimpleParticle.cu"
#include "pcuditas/particles/LennardJonesParticle.cu"
#include "pcuditas/initial_conditions/simple_cubic_lattice.cu"
#include "pcuditas/initial_conditions/random.cu"
#include "pcuditas/input_output/XYZformat.cu"
#include "pcuditas/integrators/RandomWalk.cu"
#include "pcuditas/integrators/SimpleIntegrator.cu"
#include "pcuditas/environments/EmptySpace.cu"



int main() {
    auto particles = gpu_array<LennardJonesParticle>{100};
    arrange_on_cubic_lattice(particles, 9.);
    set_random_velocities(particles, 0.1);


    auto move = SimpleIntegrator{};
    auto environment = gpu_object_from(EmptySpace{});
    std::ofstream output("output.xyz");

    for (int i=0; i < 1000; ++i) {
        move(particles, environment);
        if (i%5 == 0) {
            XYZ::write(output, particles);
        }
        
    }
}
