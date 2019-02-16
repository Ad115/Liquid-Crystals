#ifndef PARTICLE_SYSTEM_HEADER
#define PARTICLE_SYSTEM_HEADER

#include <vector>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <algorithm>
#include "Container.hpp"


/*  
    Part I: DECLARATIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/* Local joke: Camel Case xD */
template< typename ParticleClass >
class ParticleSystem { /*
    * The main class. Handles the particles vector and the container in which 
    * the simulation develops.
    */
    private:
        Container _container;
        std::vector<ParticleClass> particles;        

    public:
        template< typename Initializer >
        ParticleSystem( 
                int n_particles, // Number of particles to create
                int dimensions, // Dimensionality of the system [2 or 3]
                double numeric_density, // no. of particles / unit volume
                Initializer initial_conditions // Sets initial conditions
        );
        ~ParticleSystem() = default;

        unsigned dimensions() const; /*
        * Getter for the dimensionality of the system.
        */
        unsigned n_particles() const; /*
        * The number of particles in the system.
        */
        const Container& container() const; /*
        * The space in which the particles interact.
        */

        template< typename ParticleFunction >
        ParticleFunction map_to_particles(ParticleFunction particle_fn); /*
        * Map the given function to the particles. 
        * Useful for initialization.
        */    

        template< typename Value=double, typename ParticleFunction >
        Value measure(ParticleFunction measure_fn); /*
        * Measure some property of the particles. 
        * Accumulates the results of the measure function.
        */

        template< typename T >
        friend std::ostream& operator<<(std::ostream&, const ParticleSystem<T>&); /*
        * To print the state of the system with:
        *   
        *   std::cout << system;
        * 
        * Output: 
        * 
        *   {"container": {...container info...}, 
        *    "particles": [{...particle info...}, {...}, ...] }
        */

        void write_xyz(std::ostream&&); /*
        * Output the positions of the particles in the XYZ format.
        * The format consists in a line with the number of particles,
        * then a comment line followed by the space-separated coordinates 
        * of each particle in different lines.
        * 
        * Example (for 3 particles in the xyz diagonal):
        * 
        *   10
        *   
        *   1.0 1.0 1.0
        *   1.5 1.5 1.5
        *   2.0 2.0 2.0
        */

        double minimum_image( double r1, double r2, double box_length );

        double distance( ParticleClass other_particle );
};



/*  
    Part II: IMPLEMENTATION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

template< typename ParticleClass >
template< typename Initializer >
ParticleSystem<ParticleClass>::ParticleSystem( 
                int n_particles, // Number of particles to create
                int dimensions, // Dimensionality of the system [MAX=3]   
                double numeric_density, // Initial numeric density (particles/volume)
                Initializer set_initial_conditions
    )
    : particles(n_particles, ParticleClass(dimensions)),
      _container( dimensions, pow(n_particles/numeric_density, 1/3.) ) {

    set_initial_conditions(*this);
};


template< typename ParticleClass >
template< typename ParticleFunction >
ParticleFunction ParticleSystem<ParticleClass>::map_to_particles(
            ParticleFunction particle_fn) { /*
        * Map the given function to the particles. 
        * Useful for initialization.
        */
    return std::for_each(particles.begin(), particles.end(), particle_fn);
};


template< typename ParticleClass >
template< typename Value, typename ParticleFunction >
Value ParticleSystem<ParticleClass>::measure(ParticleFunction measure_fn) { /*
        * Measure some property of the particles. 
        * Accumulates the results of the measure function.
        */

    // Measure the property for each particle
    std::vector<Value> measurements( n_particles() );
    std::transform( particles.begin(), particles.end(),
                    measurements.begin(),
                    measure_fn );

    return std::accumulate(std::begin(measurements),
                           std::end(measurements),
                           Value(0));
};

template <typename ParticleClass>
unsigned ParticleSystem<ParticleClass>::dimensions() const { /*
        * Getter for the dimensionality of the system.
        */
        return container().dimensions();
}

template <typename ParticleClass>
unsigned ParticleSystem<ParticleClass>::n_particles() const { /*
        * The number of particles in the system.
        */
        return particles.size();
}

template <typename ParticleClass>
const Container& ParticleSystem<ParticleClass>::container() const { /*
        * The space in which the particles interact.
        */
        return _container;
}

template <typename ParticleClass>
void ParticleSystem<ParticleClass>::write_xyz(std::ostream&& stream) { /*
        * Output the positions of the particles in the XYZ format.
        * The format consists in a line with the number of particles,
        * then a comment line followed by the space-separated coordinates 
        * of each particle in different lines.
        * 
        * Example (for 3 particles in the xyz diagonal):
        * 
        *   10
        *   
        *   1.0 1.0 1.0
        *   1.5 1.5 1.5
        *   2.0 2.0 2.0
        */
    stream << n_particles() << "\n";
    for (auto p: particles) {
        stream << "\n";
        for (int D = 0; D < dimensions(); D++)
            stream << p.position[D] << " ";
    }
    stream << std::endl;

    return;
};

template< typename ParticleClass >
std::ostream& operator<<(std::ostream& stream, 
                         const ParticleSystem<ParticleClass>& sys) { /*
    * To print the state of the system with:
    *   
    *   std::cout << system;
    * 
    * Output: 
    * 
    *   {"container": {...container info...}, 
    *    "particles": [{...particle info...}, {...}, ...] }
    */

    stream << "{";

    stream << "\"container\": " << sys.container() << ", ";

    stream << "\"particles\": [";
    std::copy(std::begin(sys.particles), std::end(sys.particles)-1, 
              std::ostream_iterator<ParticleClass>(stream, ", "));

    stream << sys.particles.back() 
           << "]";

    stream << "}";

    return stream;
};

double minimum_image(double r1, double r2, double box_length) { /*
    * Get the distance to the minimum image.
    */
    double half_length = box_length/2;
    double dr = r1 - r2;

    if (dr <= -half_length) {
        return (r1 + box_length-r2);

    } else if (dr <= half_length) {
        return dr;

    } else
        return -(r2 + box_length-r1);
}

/*
template < class ParticleClass >
std::vector<double> distance( ParticleClass this_particle, ParticleClass other_particle ) { 
    * Calculate the distance using periodic boundaries.
    
    //std::vector<double> dim_distance(this_particle->dimensions);
    Distance distance = {.distance_squared=0};
    for (int D=0; D<this.dimensions; D++) {
        double dx = minimum_image( this_particle->position[D], other_particle->position[D], this_particle->getBoxLength() );
        //dim_distance[D] = dx;
        //distance.distance_squared += dx*dx;
    }

    return distance;
}
*/

#endif

