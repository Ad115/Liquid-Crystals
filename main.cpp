#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "src/ParticleSystem.hpp"
#include "src/Particle.hpp"
#include "src/random.h"


double measure_temperature(Particle& p) {
    return p.kinetic_energy();
}

class init_random_velocities { /*
    * Functor to initialize random velocities.
    */
    unsigned int dimensions;

    public:
        init_random_velocities(unsigned int dimensions) 
            : dimensions(dimensions) {}

        void operator()(Particle& p) {
            p.velocity =  Vector::random_unit(dimensions);
        }
};

class init_simple_positions { /*
    * Functor to initialize random velocities.
    */
    unsigned int dimensions;
    double L;
    int cube_length;
    int n;
    int particle_idx;

    public:
        init_simple_positions(unsigned int dimensions, double L, int n) 
            : dimensions(dimensions), 
              L(L), n(n),

              // The lowest integer such that cube_length^DIMENSIONS >= n. Think of a 
              // cube in DIMENSIONS with side cube_length where all particles are evenly 
              // spaced on a simple grid.
              cube_length(ceil(pow(n, 1./dimensions))) // No. of particles along every side
                                                       // of the cube
              {}

        void operator()(Particle& p) {
            Vector position(dimensions);
            for (int D=0; D<dimensions; D++) {
                // Get position in a hypercube with volume = cube_length^DIMENSIONS.
                position[D] = ((int)( (particle_idx / pow(cube_length, D)) )%cube_length);
                // Rescale to a box of volume = L^DIMENSIONS
                position[D] *= (L/cube_length)*0.75;
            }
            p.position =  position;
            particle_idx++;
        }
};


int main()
{
    /*std::vector<double> obj{1.,2.,3.,4.,5.,6.,7.,8.,9.,10.};
    std::vector<double> resObj(10);


    std::transform( obj.begin(), obj.end(), resObj.begin(), EjemploFormulaCerda(10) );
    for( double dato: resObj )
    {
        std::cout << dato << std::endl;
    }*/
    init_random();

    int n= 20;

    ParticleSystem<Particle> miSistemaParticula( n, 3, 1 );
    miSistemaParticula.map_to_particles(init_random_velocities(3));
    miSistemaParticula.map_to_particles(init_simple_positions(3, 10, n));

    //std::cout << "{\"system\": " << miSistemaParticula;
    //std::cout << ", \"temperature\": "
    //          << miSistemaParticula.measure<double>(measure_temperature)
    //          << "}" << std::endl;
    miSistemaParticula.write_xyz();

}