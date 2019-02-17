#include <iostream>
#include <fstream>
#include <numeric>
#include "src/Particle.hpp"
#include "src/Vector.hpp"
#include "src/ParticleSystem.hpp"
#include "src/InitialConditions.hpp"

//Compilation: g++ main.cpp -std=c++11 -Wc++11-extensions -o PartiCuditas.bin

template< typename ParticleSystem >
double temperature(ParticleSystem& system) { /*
    * Measure the temperature of the system (the sum of the particle's kinetic energies).
    */
    auto measurements = system.measure_particles( [n=system.n_particles()](Particle& p) { 
					return 2./(3*n)*p.kinetic_energy(); 
				} 
                        );
    return std::accumulate(std::begin(measurements), std::end(measurements), 0.);
}



int main( int argc, char **argv )
{
    int n_particles = 50;
    int dimensions = 3; 
    double numeric_density = .01;

    auto system = ParticleSystem<Particle, PeriodicBoundaryBox>(
                    n_particles, 
                    dimensions, 
                    numeric_density,
                    InitialConditions<SimpleCubicLattice, RandomVelocities>()
    );

    // Output initial positions to an XYZ file.
    std::ofstream outputf("output.xyz");
    for (int i=0; i<10000; i++) {
	
       if(i%10==0) system.write_xyz( outputf );
	system.integrator(.005);
        
    }

    // Print the system's initial state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature(system)
              << std::endl;
}
