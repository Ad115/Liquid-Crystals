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
            Initializer::initialize(system);

            // Initialize with the remaining conditions
            auto remaining_conditions = InitialConditions<IList...>(); // Create object
            remaining_conditions(system); // Apply
        }
};


class Initializer {
    public:

        template< typename ParticleClass >
        static Initializer initialize(ParticleSystem<ParticleClass>& system) { 
            return Initializer(); 
        }
};


class SimpleCubicLattice : Initializer {
    
    unsigned dimensions;
    double L;
    int cube_length;
    int particle_idx=0;

    public:

        template< typename ParticleClass >
        SimpleCubicLattice(const ParticleSystem<ParticleClass>& system) 
            : dimensions(system.dimensions()),
              L(system.container().side_length()),
              cube_length( // No. of particles along every side of the cube
                  ceil(pow(system.n_particles(), 1./system.dimensions())) 
                  // The lowest integer such that cube_length^DIMENSIONS >= n. 
                  // Think of a cube with side cube_length where all particles 
                  // are evenly spaced on a simple grid.
              )
            {}

        template< typename ParticleClass >
        static Initializer initialize(ParticleSystem<ParticleClass>& system) {

            auto initializer = SimpleCubicLattice(system);
            system.map_to_particles(initializer);

            return initializer;
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

class RandomVelocities : Initializer {

    public:
        template< typename ParticleClass >
        static Initializer initialize(ParticleSystem<ParticleClass>& system) { 

            auto initializer = RandomVelocities();
            system.map_to_particles(initializer);

            return initializer;  
        }

        template< typename ParticleClass >
        void operator()(ParticleClass& p) {
            p.set_velocity( Vector::random_unit(p.dimensions()) );
        }
};
