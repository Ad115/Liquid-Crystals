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
    int sampling_frecuency = 5;
    double time_step = 0.01;

    // Simulation loop
    for (int i=0; i<simulation_steps; i++) {

        if(i%sampling_frecuency==0) system.write_xyz( outputf );

        system.simulation_step(time_step);

        thermostat.setpoint += 5e-4;
    	system.apply(thermostat);
    }

    // Print the system's final state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature::measure(system)
              << std::endl;
}
