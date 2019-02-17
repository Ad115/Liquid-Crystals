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

    auto system = ParticleSystem<Particle, PeriodicBoundaryBox>(
                    n_particles, 
                    dimensions, 
                    numeric_density
    );

    // Apply initial conditions
    system.initialize_with(simple_cubic_lattice{});
    system.initialize_with(random_velocities{});
    system.initialize_with(set_temperature(1E3));


    std::ofstream outputf("output.xyz");
    set_temperature thermostat(0);
    int simulation_steps = 100000;
    int sampling_frecuency = 10;
    double time_step = 0.0005;

    // Simulation loop
    for (int i=0; i<simulation_steps; i++) {

        thermostat.setpoint += 5e-4;
    	thermostat(system);

        if(i%sampling_frecuency==0) system.write_xyz( outputf );
        system.simulation_step(time_step);
        
    }

    // Print the system's final state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature(system)
              << std::endl;
}
