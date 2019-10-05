/* 
## Clase `Container`

El espacio donde se mueven las partículas. Contiene la información sobre las 
condiciones de frontera y la distancia entre partículas. Está diseñada para 
vivir en el Device. 
*/

#ifndef EMPTY_SPACE_HEADER
#define EMPTY_SPACE_HEADER

#include "core/interfaces/Container.h"
#include "core/Vector.cu"


template< typename VectorT=Vector<> >
class EmptySpace : public Container<VectorT> {
private:

public:

    __host__ __device__ 
    VectorT apply_boundary_conditions(const VectorT& position) const {
        return position;
    }

    __host__ __device__
    VectorT distance_vector(const VectorT& p1, const VectorT& p2) const {
        return (p2 - p1);
    }
};



/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/
#ifdef __TESTING__

#include "doctest.h"
#include <typeinfo>   // operator typeid
#include "core/Vector.cu"


SCENARIO("Empty space specification") {

    GIVEN("Empty, infinite space as a container") {

        using vector_t = Vector<3, double>;
        EmptySpace<vector_t> empty_space;

        REQUIRE(empty_space.dimensions == 3);

        using empty_space_vector_t = typename decltype(empty_space)::vector_type;
        REQUIRE(typeid(empty_space_vector_t) == typeid(vector_t));


        WHEN("Boundary conditions are applied") {

            THEN("All vectors are left unchanged") { 

                Vector<> random1 = {102., 0.54, -23.4};
                Vector<> random2 = {0., 0., 0.};
                Vector<> random3 = {5e-12, 0.54, -.4};
    
                Vector<> wrapped;
                wrapped = empty_space.apply_boundary_conditions(random1);
                CHECK(wrapped == random1);
    
                wrapped = empty_space.apply_boundary_conditions(random2);
                CHECK(wrapped == random2);
    
                wrapped = empty_space.apply_boundary_conditions(random3);
                CHECK(wrapped == random3);
            }
        }

        WHEN("It is used to measure distance btw vectors") {

            THEN("The distance vector is vector difference") {

                Vector<> u = {0., 0., 0.};
                Vector<> v = {102., 0.54, -23.4};

                CHECK(empty_space.distance_vector(u, v) == (v - u));
                CHECK(empty_space.distance_vector(u, v) == v);
                CHECK(empty_space.distance_vector(v, u) == (u - v));
                CHECK(empty_space.distance_vector(v, u) == -v);
            }
        }
    }
}

#endif
#endif