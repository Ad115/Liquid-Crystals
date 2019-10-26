#include <pcuditas/gpu/gpu_array.cu>
#include <pcuditas/particles/SimpleParticle.cu>
#include <pcuditas/transform_measure/Spatial.cu>
#include <pcuditas/transform_measure/XYZformat.cu>
#include <pcuditas/integrators/RandomWalk.cu>
#include <pcuditas/environments/EmptySpace.cu>



int main() {
    auto particles = gpu_array<SimpleParticle>(100);
    move_to_origin(particles);

    auto move = RandomWalk{};
    //auto environment = EmptySpace{};
    XYZ output{"output.xyz"};

    for (int i=0; i < 100; ++i) {
        //move(particles, environment);
        move(particles);
        output(particles);
    }
}
