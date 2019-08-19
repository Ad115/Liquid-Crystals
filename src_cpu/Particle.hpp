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

        unsigned dimensions() const { return position.dimensions(); }

        void set_position(const Vector& new_position) { position = new_position; }
        void set_velocity(const Vector& new_velocity) { velocity = new_velocity; }

        double kinetic_energy() {
            Vector& v = this->velocity;
            return 1/2. * (v*v);
        };

        template<typename Container>
        Vector force_law(const Particle& other, const Container& box);

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


class LennardJones : public Particle {
    public:

        //Constructor
        LennardJones( unsigned int dimensions )
            : Particle(dimensions) 
            {}

        template<typename Container>
        Vector force_law(const Particle& other, const Container& box ){ /*
            * The force law: Lennard Jones
            * ============================
            * 
            *  f_ij = 48*e*[(s / r_ij)^12 - 1/2(s / r_ij)^6] * dr/r^2
            * 
            * See: http://www.pages.drexel.edu/~cfa22/msim/node26.html
            */
            Vector dr = box.distance_vector((*this).position, other.position);
            double r2=1./(dr*dr);
            double rcut=3.5;

            double f;
            if (r2<rcut*rcut){
                double r6=r2*r2*r2;
                f=48*(r6*r6-0.5*r6);

            } else { f = 0; }

            return f/r2 * dr;
        }
};



#endif
