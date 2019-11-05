#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/particles/LennardJonesParticle.cu"
#include "pcuditas/initial_conditions/simple_cubic_lattice.cu"
#include "pcuditas/initial_conditions/random.cu"
#include "pcuditas/input_output/XYZformat.cu"
#include "pcuditas/integrators/VelocityVertlet.cu"
#include "pcuditas/environments/PeriodicBoundaryBox.cu"
#include "pcuditas/tools/Temperature.cu"


int main() {
    auto particles = gpu_array<LennardJonesParticle>{40};
    arrange_on_cubic_lattice(particles, 6.);
    set_random_velocities(particles);

    auto thermostat = Temperature{0.1};
    auto move = VelocityVertlet{};
    auto environment = gpu_object_from(PeriodicBoundaryBox{30.});
    std::ofstream output("output.xyz");
    double dt = 0.003;

    std::cout << 0 << " temperature: " << Temperature::measure(particles)
    << std::endl;

    for (int i=0; i < 50000; ++i) {
        if (i%10 == 0) {
            XYZ::write(output, particles);
        }

        thermostat.setpoint += 5e-4;
        thermostat(particles);

        std::cout << i << " temperature: " << Temperature::measure(particles)
        << std::endl;

        move(particles, environment, dt);
    }
}
