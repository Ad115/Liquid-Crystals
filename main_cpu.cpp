#include <iostream>
#include <fstream>
#include <numeric>
#include "src_cpu/Particle.hpp"
#include "src_cpu/Vector.hpp"
#include "src_cpu/ParticleSystem.hpp"
#include "src_cpu/InitialConditions.hpp"

//Compilation: g++ main.cpp -std=c++11 -Wc++11-extensions -o PartiCuditas.out

int main( int argc, char **argv ) {
    int n_particles = 100;
    int dimensions = 3; 
    double numeric_density = .05;
    temperature thermostat{5.};

    auto system = ParticleSystem<LennardJones, PeriodicBoundaryBox>(
                    n_particles, 
                    dimensions, 
                    numeric_density,
                    initial_conditions( // Initialized in the given order!
                        simple_cubic_lattice{},
                        random_velocities{},
                        temperature{thermostat.setpoint}
                        // Insert more conditions here as you please ...
                    )
    );


    std::ofstream outputf("output.xyz");
    int simulation_steps = 2000;
    int sampling_period = 0.1;
    double time_step = 0.01;

    // Print the system's initial state
    std::cout << system << std::endl;
    std::cout 
        << 0 << " temperature: "
        << temperature::measure(system)
        << std::endl;


    // Simulation loop
    system.simulation(simulation_steps, time_step)

            .at_each_step([&](int step) { // <-- Execute this code after each step of the simulation
                    thermostat.setpoint += 5e-4;
                    system.apply(thermostat);

                    std::cout 
                        << step
                        << " temperature: "
                        << temperature::measure(system)
                        << std::endl;
            })

            .take_samples(sampling_period, [&](int _) { // <-- Execute this code in intervals of
                    system.write_xyz(outputf);          //  "sampling_period" 
            })
    .run();


    // Print the system's final state
    //std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature::measure(system)
              << std::endl;
}
