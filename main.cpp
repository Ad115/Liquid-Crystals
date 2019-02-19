#include <iostream>
#include <fstream>
#include <numeric>
#include "src/Particle.hpp"
#include "src/Vector.hpp"
#include "src/ParticleSystem.hpp"
#include "src/InitialConditions.hpp"

//Compilation: g++ main.cpp -std=c++11 -Wc++11-extensions -o PartiCuditas.bin

int main( int argc, char **argv ) {
    int n_particles = 100;
    int dimensions = 3; 
    double numeric_density = .01;
    temperature thermostat{5.};

    auto system = ParticleSystem<LennardJones, PeriodicBoundaryBox>(
                    n_particles, 
                    dimensions, 
                    numeric_density,
                    initial_conditions( // Initialized in the given order!
                        simple_cubic_lattice{},
                        random_velocities{},
                        temperature{thermostat.setpoint}
                    )
    );


    std::ofstream outputf("output.xyz");
    int simulation_steps = 2000;
    int sampling_period = 0.05;
    double time_step = 0.01;

    // Simulation loop
    system.simulation(simulation_steps, time_step)

            .at_each_step([&]() { // <-- Execute this code after each step of the simulation
                    thermostat.setpoint += 5e-4;
                    system.apply(thermostat);
            })

            .take_samples(sampling_period, [&]() { // <-- Execute this code in intervals of
                    system.write_xyz(outputf);     //  "sampling_period" (first sample at t=0) 
            })
    .run();

    // Print the system's final state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature::measure(system)
              << std::endl;
}
