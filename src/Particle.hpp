#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER
#include <iostream>
#include <vector>
#include "Vector.hpp"


class Particle {
    public:
        Vector position;
        Vector velocity;
        Vector force;
	
        //Constructor
        Particle( unsigned int dimensions )
            : position(dimensions),
              velocity(dimensions),  
              force(dimensions) {}

        //Destructor
        ~Particle() = default;

        unsigned dimensions() const { return position.dimensions(); }

        void set_position(const Vector& new_position) { position = new_position; }
        void set_velocity(const Vector& new_velocity) { velocity = new_velocity; }

        double kinetic_energy() {
            Vector& v = this->velocity;
            return 1/2. * (v*v);
        };

	template<typename Container>
	Vector interaction(const Particle& other, const Container& box ){
		Vector dr=box.minimum_image((*this).position, other.position);
		double r2=1./(dr*dr);
		double rcut=3.5;
		double f;
		if (r2<rcut*rcut){
			double r6=r2*r2*r2;
			f=48*(r6*r6-0.5*r6);
		}else { f = 0; }
		return f/r2*dr;
	}


        
        friend std::ostream& operator<<(std::ostream& s, const Particle& p);

};


std::ostream& operator<<(std::ostream& stream, 
                         const Particle& p) {
    stream << "{";
    stream << "\"position\":" << p.position << ", ";
    stream << "\"velocity\":" << p.velocity << ", ";
    stream << "\"force\":" << p.force;
    stream << "}";
    return stream;
}




#endif
