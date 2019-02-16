#include <iostream>
#include <fstream>
#include "src/ParticleSystem.hpp"
#include "src/Particle.hpp"
#include "src/Simulation.hpp"


//Compilation: g++ main.cpp -std=c++11 -Wc++11-extensions -o PartiCuditas.bin


double measure_temperature(Particle& p) {
    return p.kinetic_energy();
}

int main( int argc, char **argv )
{
    int n_particles = 100000;
    int dimensions = 3; 
    double numeric_density = 1;

    ParticleSystem<Particle> system(
        n_particles, 
        dimensions, 
        numeric_density,
        InitialConditions<SimpleCubicLattice, RandomVelocities>()
    );


    // Output positions to an XYZ file.
    system.write_xyz( std::ofstream("output.xyz") );

    // Print the system's initial state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << system.measure(measure_temperature)
              << std::endl;
}



    
