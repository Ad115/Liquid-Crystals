#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "src/ParticleSystem.hpp"
#include "src/Particle.hpp"
#include "src/random.h"
//Compilation: g++ main.cpp -std=c++11 -Wc++11-extensions -o PartiCuditas.bin
double measure_temperature(Particle& p) {
    return p.kinetic_energy();
}

int main( int argc, char **argv )
{
    init_random();

    int nParticles=100000;
    int dimensions=3; 
    double numeric_density=1;
    ParticleSystem<Particle> miSistemaParticula( nParticles, dimensions, numeric_density );
    miSistemaParticula.write_xyz();
    return 0;
}



    //std::cout << "{\"system\": " << miSistemaParticula;
    //std::cout << ", \"temperature\": "
    //          << miSistemaParticula.measure<double>(measure_temperature)
    //          << "}" << std::endl;