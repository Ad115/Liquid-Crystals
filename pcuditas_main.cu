#include <pcuditas/gpu/gpu_array.cu>
#include <pcuditas/gpu/gpu_object.cu>
#include <pcuditas/particles/SimpleParticle.cu>
#include <pcuditas/transform_measure/simple_cubic_lattice.cu>
#include <pcuditas/transform_measure/random.cu>
#include <pcuditas/input_output/XYZformat.cu>
#include <pcuditas/integrators/RandomWalk.cu>
#include <pcuditas/environments/EmptySpace.cu>



int main() {
    auto particles = gpu_array<SimpleParticle>{100};
    set_random_positions(particles, 100.);
    set_random_velocities(particles, 100.);

    auto move = RandomWalk{};
    auto environment = gpu_object_from(EmptySpace{});
    std::ofstream output("output.xyz");

    for (int i=0; i < 100; ++i) {
        move(particles, environment);
        XYZ::write(output, particles);
    }
}
