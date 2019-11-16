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
        return (p1 - p2);
    }
};


/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/
#ifdef __TESTING__

#include "tests/doctest.h"
#include <typeinfo>   // operator typeid
#include "pcuditas/vectors/EuclideanVector.cu"


SCENARIO("Empty space specification") {

    GIVEN("Empty, infinite space as a container") {

        auto empty_space = EmptySpace{};
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

#include "pcuditas/gpu/gpu_object.cu"

SCENARIO("Empty space on GPU") {

    GIVEN("Empty, infinite space as an object in GPU") {

        auto empty_space = in_gpu(EmptySpace{});
        using vector_t = EuclideanVector<3>;

        WHEN("Boundary conditions are applied on vectors in CPU") {

            auto cpu_space = empty_space.to_cpu();

            auto random1 = vector_t{102., 0.54, -23.2e10};
            auto wrapped1 = cpu_space.apply_boundary_conditions(random1);

            auto random2 = vector_t{0., 0., 0.};
            auto wrapped2 = cpu_space.apply_boundary_conditions(random2);

            auto random3 = vector_t{5e-12, 0.54, -.4};
            auto wrapped3 = cpu_space.apply_boundary_conditions(random3);


            THEN("The boundary conditions applied on GPU yield the same results") { 
    
                empty_space.call_on_gpu(
                    [random1, random2, random3,
                     wrapped1, wrapped2, wrapped3] 
                     __device__ 
                     (EmptySpace &s) {
                        assert(s.apply_boundary_conditions(random1) == wrapped1);
                        assert(s.apply_boundary_conditions(random2) == wrapped2);
                        assert(s.apply_boundary_conditions(random3) == wrapped3);
                    }
                );
            }
        }

        WHEN("It is used to measure distance btw vectors in GPU") {

            THEN("The distance vector is vector difference (checked on GPU)") {

                vector_t u = {0., 0., 0.};
                vector_t v = {102., 0.54, -23.4};

                empty_space.call_on_gpu(
                    [u,v] 
                     __device__ 
                     (EmptySpace &s) {
                        assert(s.distance_vector(u, v) == (v - u));
                        assert(s.distance_vector(v, u) == (u - v));
                        assert(s.distance_vector(u, u) == vector_t::zero());
                        assert(s.distance_vector(v, v) == vector_t::zero());
                    }
                );
            }
        }
    }
}

#endif
