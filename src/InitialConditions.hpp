#ifndef INITIAL_CONDITIONS_HEADER
#define INITIAL_CONDITIONS_HEADER

#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include "Vector.hpp"


class simple_cubic_lattice {
    
    unsigned dimensions;
    double L;
    int cube_length;
    int particle_idx=0;

    public:

        template< typename ParticleSystem >
        void fetch_parameters_from(const ParticleSystem& system) {
            dimensions = system.dimensions();
            L = system.container().side_length();

             // No. of particles along every side of the cube
            cube_length = ceil(pow(system.n_particles(), 1./system.dimensions()));
                  // The lowest integer such that cube_length^DIMENSIONS >= n. 
                  // Think of a cube with side cube_length where all particles 
                  // are evenly spaced on a simfunction ple grid.
        } 

        template< typename ParticleSystem>
        simple_cubic_lattice& operator()(ParticleSystem& system) { 
            fetch_parameters_from(system);

            using Particle = typename ParticleSystem::Particle_t;
            system.map_to_particles([this](Particle& p) { (*this).particle_fn(p); });

            return *this;
        }

        template< typename ParticleClass>
        void particle_fn(ParticleClass& p) {

            Vector position(dimensions);
            for (int D=0; D<dimensions; D++) {
                // Get position in a hypercube with volume = cube_length^DIMENSIONS.
                position[D] = ((int)( (particle_idx / pow(cube_length, D)) )%cube_length);
                // Rescale to a box of volume = L^DIMENSIONS
                position[D] *= (L/cube_length)*0.75;
            }
            p.set_position(position);
            particle_idx++;
        }
};

class random_velocities {

    public:
        template< typename ParticleSystem>
        random_velocities& operator()(ParticleSystem& system) { 
            using Particle = typename ParticleSystem::Particle_t;
            system.map_to_particles([this](Particle& p) { (*this).particle_fn(p); });

            return *this;
        }

        template< typename ParticleClass >
        void particle_fn(ParticleClass& p) {
            p.set_velocity( Vector::random_unit(p.dimensions()) );
        }
};



#endif