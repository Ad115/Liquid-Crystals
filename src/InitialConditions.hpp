#ifndef INITIAL_CONDITIONS_HEADER
#define INITIAL_CONDITIONS_HEADER

#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include "Vector.hpp"


template< typename ...IList > // <-- Base case, stops recursion with no parameters
class InitialConditions {
    public:

        template< typename ParticleSystem >
        void operator()(ParticleSystem& system) {}
};

template< typename Initializer, typename ...IList > // <-- Recursive case
class InitialConditions<Initializer, IList...> {

    Initializer first;
    InitialConditions<IList...> remaining;

    public:

        InitialConditions(Initializer initializer, IList ...p)
         : first(initializer),
           remaining(p...)
         {}

        

        template< typename ParticleSystem >
        void operator()(ParticleSystem& system) {

            // Initialize with "first"
            first(system);

            // Initialize with the remaining conditions
            remaining(system);
        }
    

};

template< typename Initializer, typename ...IList >
InitialConditions<Initializer, IList...> 
        initial_conditions(Initializer first, IList ...rest) {
    return InitialConditions<Initializer, IList...>{first, rest...};
}


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
        void operator()(ParticleSystem& system) { 

            fetch_parameters_from(system);

            using Particle = typename ParticleSystem::Particle_t;
            system.map_to_particles([this](Particle& p) { (*this).particle_fn(p); });
        }

        template< typename ParticleClass>
        void particle_fn(ParticleClass& p) {

            Vector position(dimensions);
            for (int D=0; D<dimensions; D++) {
                // Get position in a hypercube with volume = cube_length^DIMENSIONS.
                position[D] = ((int)( (particle_idx / pow(cube_length, D)) )%cube_length);
                // Rescale to a box of volume = L^DIMENSIONS
                position[D] *= (L/cube_length)*0.9; // The 0.9 factor is for safety,
                                                    // particles on the edges aren't
                                                    // too close.
            }
            p.set_position(position);
            particle_idx++;
        }
};

class random_velocities {

    public:
        template< typename ParticleSystem>
        void operator()(ParticleSystem& system) { 

            using Particle = typename ParticleSystem::Particle_t;

            system.map_to_particles([this](Particle& p) { 
                    p.set_velocity( Vector::random_unit(p.dimensions()) );
                }
            );
        }
};


class temperature {
    public:

        double setpoint;

        template< typename ParticleSystem >
        static double measure(ParticleSystem& system) { /*
            * Measure the temperature of the system (the sum of the particle's 
            * kinetic energies).
            */
            auto measurements = system.measure_particles( 
                                    [n=system.n_particles()](Particle& p) { 
                                        return 2./(3*n)*p.kinetic_energy(); 
                                    } 
                                );
            return std::accumulate(std::begin(measurements), 
                                   std::end(measurements), 
                                   0.
                    );
        }

        temperature(double setpoint)
         : setpoint(setpoint)
         {}

        template< typename ParticleSystem>
        void operator()(ParticleSystem& system) { 
            
            double current_temperature = measure(system);
            double correction_factor = sqrt(setpoint / current_temperature);

            using Particle = typename ParticleSystem::Particle_t;

            system.map_to_particles([correction_factor](Particle& p){
                    p.velocity = correction_factor * p.velocity;
                }
            );
        }
};


#endif
