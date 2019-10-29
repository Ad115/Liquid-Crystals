#pragma once

class EmptySpace {
public:

    template <class VectorT>
    __host__ __device__
    VectorT apply_boundary_conditions(const VectorT& position) const {
        return position;
    }

    template <class VectorT>
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
#include "pcuditas/vectors/EuclideanVector.cu"


SCENARIO("Empty space specification") {

    GIVEN("Empty, infinite space as a container") {

        EmptySpace empty_space;
        using vector_t = EuclideanVector<3>;

        WHEN("Boundary conditions are applied") {

            vector_t random1 = {102., 0.54, -23.2e10};
            vector_t wrapped1 = empty_space.apply_boundary_conditions(random1);

            vector_t random2 = {0., 0., 0.};
            vector_t wrapped2 = empty_space.apply_boundary_conditions(random2);

            vector_t random3 = {5e-12, 0.54, -.4};
            vector_t wrapped3 = empty_space.apply_boundary_conditions(random3);


            THEN("All vectors are left unchanged (empty space has no boundary)") { 
    
                CHECK(wrapped1 == random1);
                CHECK(wrapped2 == random2);
                CHECK(wrapped3 == random3);
            }
        }

        WHEN("It is used to measure distance btw vectors") {

            THEN("The distance vector is vector difference") {

                vector_t u = {0., 0., 0.};
                vector_t v = {102., 0.54, -23.4};

                CHECK(empty_space.distance_vector(u, v) == (v - u));
                CHECK(empty_space.distance_vector(u, v) == v);
                CHECK(empty_space.distance_vector(v, u) == (u - v));
                CHECK(empty_space.distance_vector(v, u) == -v);
            }
        }
    }
}

#endif
