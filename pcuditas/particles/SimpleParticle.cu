/* 
## Clase `Particle`

Al igual que la clase `Vector`, la clase `Particle` es una clase que puede ser 
instanciada tanto en el Host como en el Device. Contiene vectores de posición, 
velocidad y fuerza. 
*/ 

#ifndef PARTICLE_HEADER
#define PARTICLE_HEADER
#include "pcuditas/vectors/Euclidean.cu"

template< typename VectorT=EuclideanVector<> >
struct Particle {

    using vector_type = VectorT;
    static constexpr int dimensions = VectorT::dimensions;
        
    VectorT position;
    VectorT velocity;
    VectorT force;

    friend std::ostream& operator<<(
        std::ostream& stream, 
        const Particle<VectorT>& p) {
    
        stream 
            << "Particle{"
            << p.position << ", "
            << p.velocity << ", "
            << p.force
            << "}";
            
        return stream;
    }
};

template<class VectorT>
constexpr int Particle<VectorT>::dimensions;


/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "doctest.h"
#include <typeinfo>   // operator typeid

SCENARIO("Particle specification") {

    GIVEN("A particle") {

        using vector_t = Vector<3, double>;
        Particle<vector_t> p;

        THEN("It's state is given by a position, a velocity and a force vector") {
            // Nothing to test here...
        }

        AND_THEN("It state is given in terms of a specific frame of reference") {

            CHECK(p.dimensions == vector_t::dimensions);

            using p_vector_t = typename decltype(p)::vector_type;
            CHECK(typeid(p_vector_t) == typeid(vector_t));
        }
    }
}

#endif
#endif