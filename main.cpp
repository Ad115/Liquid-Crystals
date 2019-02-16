#include <iostream>
#include <fstream>
#include <numeric>
#include "src/Particle.hpp"
#include "src/ParticleSystem.hpp"
#include "src/InitialConditions.hpp"

//Compilation: g++ main.cpp -std=c++11 -Wc++11-extensions -o PartiCuditas.bin

template< typename ParticleSystem >
double temperature(ParticleSystem& system) { /*
    * Measure the temperature of the system (the sum of the particle's kinetic energies).
    */
    auto measurements = system.measure_particles( 
                            [](Particle& p) { return p.kinetic_energy(); } 
                        );
    return std::accumulate(std::begin(measurements), std::end(measurements), 0.);
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


    // Output initial positions to an XYZ file.
    system.write_xyz( std::ofstream("output.xyz") );

    // Print the system's initial state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature(system)
              << std::endl;
}