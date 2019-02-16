#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER
#include <iostream>
#include <vector>
#include "ParticleSystem.hpp"
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
