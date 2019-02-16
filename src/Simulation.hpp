#include "Container.hpp"
#include "ParticleSystem.hpp"


template< typename ...IList > // <-- Base case, stops recursion with no parameters
class InitialConditions {
    public:
        template< typename ParticleClass >
        void operator()(ParticleSystem<ParticleClass>& system) {}
};

template< typename Initializer, typename ...IList > // <-- Recursive case
class InitialConditions<Initializer, IList...> {
    public:

        template< typename ParticleClass >
        void operator()(ParticleSystem<ParticleClass>& system) {

            // Initialize with "Initializer" conditions
            system.map_to_particles( Initializer::with_parameters_from(system) );

            // Initialize with the remaining conditions
            auto remaining_conditions = InitialConditions<IList...>(); // Create object
            remaining_conditions(system); // Apply
        }
};


class SimpleCubicLattice {
    
    unsigned dimensions;
    double L;
    int cube_length;
    int particle_idx=0;

    public:
        SimpleCubicLattice(unsigned d, double L, int l) 
            : dimensions(d), L(L), cube_length(l) 
            {}

        template< typename ParticleClass >
        static SimpleCubicLattice with_parameters_from(
                const ParticleSystem<ParticleClass>& system) {

            unsigned dimensions = system.dimensions();
            double L = system.container().side_length();

            // The lowest integer such that cube_length^DIMENSIONS >= n. Think of a 
            // cube in DIMENSIONS with side cube_length where all particles are evenly 
            // spaced on a simple grid.
            int cube_length = ceil(pow(system.n_particles(), 1./system.dimensions()));
                                            // No. of particles along every side
                                            // of the cube

            return SimpleCubicLattice(dimensions, L, cube_length);
        }

        template< typename ParticleClass >
        void operator()(ParticleClass& p) {

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

class RandomVelocities {/*
    * Functor to initialize random velocities.
    * 
    * Instantiate with the parameters of a ParticleSystem<Particle>: 
    * 
    *   instance = RandomVelocities::with_parameters_from(system)
    * 
    * Initialize the system:
    * 
    *   system.map_to_particles(instance)
    */

    public:
        template< typename ParticleClass >
        static RandomVelocities with_parameters_from(
                const ParticleSystem<ParticleClass>& system) { 
            return RandomVelocities();  
        }

        template< typename ParticleClass >
        void operator()(ParticleClass& p) {
            p.set_velocity( Vector::random_unit(p.dimensions()) );
        }
};
