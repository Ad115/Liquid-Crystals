#ifndef PARTICLE_SYSTEM_HEADER
#define PARTICLE_SYSTEM_HEADER

#include <vector>
#include <iostream>
#include <iterator>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "Vector.hpp"
#include "Container.hpp"


/*  
    Part I: DECLARATIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


/*Particles system class, it is the space
  where the particles will be simulated
Local joke: Camel Case xD */
template< typename ParticleClass >
class ParticleSystem {
    private:
        Container container;
        std::vector<ParticleClass> particles;        

    public:
        ParticleSystem( 
                int n_particles, // Number of particles to create
                int dimensions, // Dimensionality of the system [2 or 3]
                double numeric_density // no. of particles / unit volume
        );
        ~ParticleSystem() = default;

        unsigned dimensions(); /*
        * Getter for the dimensionality of the system.
        */
        unsigned n_particles(); /*
        * The number of particles in the system.
        */
        double container_side_length(); /*
        * The length of each side of the container.
        * (warning, assumes a square container)
        */

        template< typename ParticleFunction >
        ParticleFunction map_to_particles(ParticleFunction particle_fn);
    

        template< typename Value=double, typename ParticleFunction >
        Value measure(ParticleFunction measure_fn);

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



class init_random_velocities { /*
    * Functor to initialize random velocities.
    */
    unsigned int dimensions;

    public:
        init_random_velocities(unsigned int dimensions) 
            : dimensions(dimensions) {}
        template < class ParticleClass >
        void operator()(ParticleClass& p) {
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
        template < class ParticleClass >
        void operator()(ParticleClass& p) {
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


/*  
    Part II: IMPLEMENTATION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

template < typename ParticleClass >
ParticleSystem<ParticleClass>::ParticleSystem( 
                int n_particles, // Number of particles to create
                int dimensions, // Dimensionality of the system [MAX=3]   
                double numeric_density // Initial numeric density (particles/volume)
                //positions_init_fn init_positions,
                //velocities_init_fn init_velocities
    )
    : particles(n_particles, ParticleClass(dimensions)),
      container( dimensions, pow(n_particles/numeric_density, 1/3.) ) {

    //Start the particle setup
    map_to_particles( init_random_velocities( dimensions ) );

    map_to_particles( init_simple_positions( dimensions, 
                                             container_side_length(), 
                                             n_particles ));
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
unsigned ParticleSystem<ParticleClass>::dimensions() { /*
        * Getter for the dimensionality of the system.
        */
        return container.dimensions();
}

template <typename ParticleClass>
unsigned ParticleSystem<ParticleClass>::n_particles() { /*
        * The number of particles in the system.
        */
        return particles.size();
}

template <typename ParticleClass>
double ParticleSystem<ParticleClass>::container_side_length() { /*
        * The length of each side of the container.
        * (warning, assumes a square container)
        */
        return container.side_length();
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

    stream << "\"container\": " << sys.container << ", ";

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

