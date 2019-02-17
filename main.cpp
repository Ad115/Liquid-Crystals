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
    system.apply(simple_cubic_lattice{});
    system.apply(random_velocities{});
    system.apply(set_temperature{0});


    std::ofstream outputf("output.xyz");
    set_temperature thermostat(1);
    int simulation_steps = 100000;
    int sampling_frecuency = 10;
    double time_step = 0.0005;

    // Simulation loop
    for (int i=0; i<simulation_steps; i++) {

        system.simulation_step(time_step);

        if(i%sampling_frecuency==0) system.write_xyz( outputf );

        thermostat.setpoint += 5e-4;
    	system.apply(thermostat);
    }

    // Print the system's final state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature(system)
              << std::endl;
}
