#ifndef PARTICLE_SYSTEM_HEADER
#define PARTICLE_SYSTEM_HEADER

#include <vector>
#include <cmath>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "Container.hpp"


/*  
    Part I: DECLARATIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/* Local joke: Camel Case xD */
template< typename ParticleClass, typename ContainerClass >
class ParticleSystem { /*
    * The main class. Handles the particles vector and the container in which 
    * the simulation develops.
    */
    private:
        ContainerClass _container;
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
        const ContainerClass& container() const; /*
        * The space in which the particles interact.
        */

	ContainerClass& container();

        template< typename ParticleFunction >
        ParticleFunction map_to_particles(ParticleFunction particle_fn); /*
        * Map the given function to the particles. 
        * Useful for initialization.
        */    

        template< typename Value=double, typename ParticleFunction >
        std::vector<Value> measure_particles(ParticleFunction measure_fn); /*
        * Measure some property of the particles. 
        * Returns a vector of the measurements for each particle.
        */

	void integrator(double dt);

        template< typename T, typename O >
        friend std::ostream& operator<<(std::ostream&, const ParticleSystem<T,O>&); /*
        * To print the state of the system with:
        *   
        *   std::cout << system;
        * 
        * Output: 
        * 
        *   {"container": {...container info...}, 
        *    "particles": [{...particle info...}, {...}, ...] }
        */

        void write_xyz(std::ostream&); /*
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
	
	void update_forces();
};



/*  
    Part II: IMPLEMENTATION
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

template< typename ParticleClass, typename ContainerClass >
template< typename Initializer >
ParticleSystem<ParticleClass, ContainerClass>::ParticleSystem( 
                int n_particles, // Number of particles to create
                int dimensions, // Dimensionality of the system [MAX=3]   
                double numeric_density, // Initial numeric density (particles/volume)
                Initializer set_initial_conditions
    )
    : particles(n_particles, ParticleClass(dimensions)),
      _container( dimensions, pow(n_particles/numeric_density, 1/3.) ) {

    set_initial_conditions(*this);
};


template< typename ParticleClass, typename ContainerClass >
template< typename ParticleFunction >
ParticleFunction ParticleSystem<ParticleClass, ContainerClass>::map_to_particles(
            ParticleFunction particle_fn) { /*
        * Map the given function to the particles. 
        * Useful for initialization.
        */
    return std::for_each(particles.begin(), particles.end(), particle_fn);
};


template< typename ParticleClass, typename ContainerClass >
template< typename Value, typename ParticleFunction >
std::vector<Value> ParticleSystem<ParticleClass, ContainerClass>::measure_particles(
        ParticleFunction measure_fn) { /*
    * Measure some property of the particles. 
    * Returns a vector of the measurements for each particle.
    */
    std::vector<Value> measurements( n_particles() );

    // Measure the property for each particle
    std::transform( particles.begin(), particles.end(),
                    measurements.begin(),
                    measure_fn );

    return measurements;
};

template < typename ParticleClass, typename ContainerClass >
unsigned ParticleSystem<ParticleClass, ContainerClass>::dimensions() const { /*
        * Getter for the dimensionality of the system.
        */
        return container().dimensions();
}

template < typename ParticleClass, typename ContainerClass >
unsigned ParticleSystem<ParticleClass, ContainerClass>::n_particles() const { /*
        * The number of particles in the system.
        */
        return particles.size();
}

template < typename ParticleClass, typename ContainerClass >
const ContainerClass& ParticleSystem<ParticleClass, ContainerClass>::container() const { /*
        * The space in which the particles interact.
        */
        return _container;
}

template < typename ParticleClass, typename ContainerClass >
ContainerClass& ParticleSystem<ParticleClass, ContainerClass>::container() { /*
        * The space in which the particles interact.
        */
        return _container;
}


template < typename ParticleClass, typename ContainerClass >
void ParticleSystem<ParticleClass, ContainerClass>::write_xyz(std::ostream& stream) { /*
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

template< typename ParticleClass, typename ContainerClass >
void ParticleSystem<ParticleClass, ContainerClass>::integrator(double dt){
        map_to_particles([dt, &box=container()](ParticleClass& p){
			// r(t+dt) = r(t) + v(t)*dt + [1/2*f*dt^2]
    			p.position = p.position + dt * p.velocity + 1/2.*dt*dt*p.force;
                	p.position = box.apply_boundary_conditions(p.position);
			
			// v(t+dt/2) = v(t)+f(t)/2*dt
			p.velocity = p.velocity + dt/2.*p.force;

		}
	);
	
	// r(t + dt) --> f(t + dt)
	update_forces();
	
	map_to_particles([dt](ParticleClass& p){
			// v(t+dt) = v(t+dt/2)+f(t+dt)/2*dt
			p.velocity = p.velocity + dt/2.*p.force;
		}
	);
}

template< typename ParticleClass, typename ContainerClass >
void ParticleSystem<ParticleClass, ContainerClass>::update_forces(){
	for(auto& p : particles){p.force=0*p.force;}
	for(int i=0; i<particles.size()-1; i++) {
		for(int j=i+1; j<particles.size(); j++) {
			Vector force = particles[i].interaction(particles[j], container());
			particles[i].force += force;
			particles[j].force -= force;
		}
	}
}

template< typename ParticleClass, typename ContainerClass >
std::ostream& operator<<(std::ostream& stream, 
                         const ParticleSystem<ParticleClass, ContainerClass>& sys) { /*
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

