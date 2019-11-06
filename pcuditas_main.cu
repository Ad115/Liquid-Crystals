#include "pcuditas/gpu/gpu_array.cu"
#include "pcuditas/gpu/gpu_object.cu"
#include "pcuditas/particles/LennardJonesParticle.cu"
#include "pcuditas/initial_conditions/simple_cubic_lattice.cu"
#include "pcuditas/initial_conditions/random.cu"
#include "pcuditas/input_output/XYZformat.cu"
#include "pcuditas/integrators/VelocityVertlet.cu"
#include "pcuditas/environments/PeriodicBoundaryBox.cu"
#include "pcuditas/tools/Temperature.cu"

int n_particles = 40;
double time_step = 0.003;

int main() {
    // Create the particles on the GPU
    auto particles = gpu_array<LennardJonesParticle>(n_particles);

    // Apply initial conditions
    arrange_on_cubic_lattice(particles, 6.);
    set_random_velocities(particles);

    // Measure initial temperature
    std::cout << 0 
        << " temperature: " << Temperature::measure(particles)
        << std::endl;

    auto environment = in_gpu(PeriodicBoundaryBox{30.});
    auto thermostat  = Temperature{0.1};
    auto move        = VelocityVertlet{}; // integrator
    std::ofstream output{"output.xyz"};

    for (int i=0; i < 50000; ++i) {

        if (i%10 == 0) { // Write to output file
            XYZ::write(output, particles);
        }

        // Increase temperature
        thermostat.setpoint += 5e-4;
        thermostat(particles);

        // Measure temperature
        std::cout << i 
            << " temperature: " << Temperature::measure(particles)
            << std::endl;

        // Move one step in time
        move(particles, environment, time_step);
    }
}
