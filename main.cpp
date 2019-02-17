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
    double numeric_density = .001;

    auto system = ParticleSystem<Particle, PeriodicBoundaryBox>(
                    n_particles, 
                    dimensions, 
                    numeric_density
    );

    // Apply initial conditions
    system.initialize_with(simple_cubic_lattice{});
    system.initialize_with(random_velocities{});
    system.initialize_with(set_temperature(1E3));

    

    // Output initial positions to an XYZ file.
    std::ofstream outputf("output.xyz");
    for (int i=0; i<10000; i++) {
    	system.initialize_with(set_temperature(1E2-i*1E-2));
        if(i%10==0) system.write_xyz( outputf );
        system.integrator(.0005);
        
    }

    // Print the system's initial state
    std::cout << system << std::endl;

    std::cout << "temperature: "
              << temperature(system)
              << std::endl;
}
