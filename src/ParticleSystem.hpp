#ifndef PARTICLE_SYSTEM_HEADER
#define PARTICLE_SYSTEM_HEADER

#include <vector>
#include <iostream>
#include <iterator>
#include <numeric>
#include "Vector.hpp"
#include "Container.hpp"

/*  
    Part I: DECLARATIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


/*Particles system class, it is the space
  where the particles will be simulated
Local joke: Camel Case xD */
template< typename Particle_class >
class ParticleSystem {
    private:
        unsigned int n;
        int dimensions;

        Container system_container;
        std::vector<Particle_class> particles;        

    public:
        ParticleSystem( 
                int nParticles, //1: Number of particles to create
                int dimensions, //2: Number Dimensions to work [MAX=3]   
                double numeric_density //3: Initial numeric density (particles/volume)
                //positions_init_fn init_positions,
                //velocities_init_fn init_velocities
        );
        ~ParticleSystem() = default;
        
        template< typename ParticleFunction >
        void map_to_particles(ParticleFunction particle_fn);

        template< typename Value, typename ParticleFunction >
        Value measure(ParticleFunction measure_fn);

        template< typename T >
        friend std::ostream& operator<<(
                    std::ostream& stream, 
                    const ParticleSystem<T>& sys
        );

        void write_xyz(void);
};

/*  
    Part II: IMPLEMENTATION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

template <typename Particle_class>
void ParticleSystem<Particle_class>::write_xyz() {
    std::cout << this->n << "\n";
    for (auto p: particles) {
        std::cout << "\n";
        for (int D=0; D<dimensions; D++)
            std::cout << p.position[D] << " ";
    }
    return;
}

template< typename Particle_class >
std::ostream& operator<<(std::ostream& stream, 
                         const ParticleSystem<Particle_class>& sys) {

    stream << "{";

    stream << "\"container\": " << sys.system_container << ", ";

    stream << "\"particles\": [";
    std::copy(std::begin(sys.particles), std::end(sys.particles)-1, 
              std::ostream_iterator<Particle_class>(stream, ", "));

    stream << sys.particles.back() 
           << "]";

    stream << "}";

    return stream;
}

template <typename Particle_class>
ParticleSystem<Particle_class>::ParticleSystem( 
                int nParticles, //1: Number of particles to create
                int dimensions, //2: Number Dimensions to work [MAX=3]   
                double numeric_density //3: Initial numeric density (particles/volume)
                //positions_init_fn init_positions,
                //velocities_init_fn init_velocities
    )
    : n(nParticles) , 
      dimensions(dimensions), 
      particles(n, Particle_class(dimensions)),
      system_container( dimensions, pow(n/numeric_density, 1/3.) ) {
}


template< typename Particle_class >
template< typename ParticleFunction >
void ParticleSystem<Particle_class>::map_to_particles(
        ParticleFunction particle_fn) { /*
        * Map the given function to the particles. 
        * Useful for initialization.
        */

    for (auto& p : particles) {
        particle_fn(p);
    }

    return;
};

template< typename Particle_class >
template< typename Value, typename ParticleFunction >
Value ParticleSystem<Particle_class>::measure(
        ParticleFunction measure_fn) { /*
    * Measure some property of the particles. 
    * Accumulates the results of the measure function.
    */
    // Measure the property for each particle
    std::vector<Value> measurements(this->n);
    std::transform( this->particles.begin(), this->particles.end(),
                    measurements.begin(),
                    measure_fn );

    Value sum(0);
    return std::accumulate(std::begin(measurements), std::end(measurements),
                           Value(0));
}

#endif

