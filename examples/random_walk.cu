#include <pcuditas/gpu/gpu_array.cu>
#include <pcuditas/particles/SimpleParticle.cu>
#include <pcuditas/initial_conditions/move_to_origin.cu>
#include <pcuditas/input_output/XYZformat.cu>
#include <pcuditas/integrators/RandomWalk.cu>
#include <pcuditas/environments/EmptySpace.cu>



int main() {
    auto particles = gpu_array<SimpleParticle>(100);
    move_to_origin(particles);

    auto move = RandomWalk{};
    XYZ output{"output.xyz"};

    for (int i=0; i < 100; ++i) {
        move(particles);
        output(particles.to_cpu());
    }
}
