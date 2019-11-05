#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/particles/LennardJonesParticle.cu"
#include "pcuditas/initial_conditions/simple_cubic_lattice.cu"
#include "pcuditas/initial_conditions/random.cu"
#include "pcuditas/input_output/XYZformat.cu"
#include "pcuditas/integrators/SimpleIntegrator.cu"
#include "pcuditas/environments/PeriodicBoundaryBox.cu"



int main() {
    auto particles = gpu_array<LennardJonesParticle>{40};
    arrange_on_cubic_lattice(particles, 6.);
    set_random_velocities(particles, .05);


    auto move = SimpleIntegrator{};
    auto environment = gpu_object_from(PeriodicBoundaryBox{10.});
    std::ofstream output("output.xyz");
    double dt = 0.003;

    for (int i=0; i < 50000; ++i) {
        if (i%100 == 0) {
            XYZ::write(output, particles);
        }

        move(particles, environment, dt);
    }
}
