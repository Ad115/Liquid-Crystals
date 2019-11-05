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
    arrange_on_cubic_lattice(particles, 15.);
    set_random_velocities(particles, 5.);


    auto move = SimpleIntegrator{};
    auto environment = gpu_object_from(EmptySpace{});
    std::ofstream output("output.xyz");

    for (int i=0; i < 100; ++i) {
        move(particles, environment);
        XYZ::write(output, particles);
    }
}
