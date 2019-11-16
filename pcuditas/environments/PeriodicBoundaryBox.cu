#pragma once

class PeriodicBoundaryBox {
public:

    double side_length;

    __host__ __device__ PeriodicBoundaryBox( double L )
            : side_length(L) {}

    template <class VectorT>
    __host__ __device__ 
    VectorT apply_boundary_conditions(const VectorT& position) {
        VectorT new_pos(position);
        for (int d=0; d<position.dimensions;d++){

            if (position[d] > side_length) {
                new_pos[d] -= side_length * (int)(position[d]/side_length);
            }
            
            if (position[d] < 0) {
                new_pos[d] += side_length * ((int)(position[d]/side_length) + 1);
            }
            
        }
        return new_pos;
    }

    template <class VectorT>
    __host__ __device__ 
    VectorT distance_vector(const VectorT& r1, const VectorT& r2) const { /*
        * Get the distance to the minimum image.
        */
        double half_length = side_length/2;
        VectorT dr = r1 - r2;

        for(int D=0; D<dr.dimensions; D++) {

            if (dr[D] <= -half_length) {
                dr[D] += side_length;

            } else if (dr[D] > +half_length) {
                dr[D] -= side_length;
            }
        }
        return dr;
    }

    friend std::ostream& operator<<(
        std::ostream& stream, 
        const PeriodicBoundaryBox& box) {

        stream << "PeriodicBoundaryBox{"
            << box.side_length 
        << '}';
        return stream;
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



SCENARIO("Periodic boundary box specification") {

    GIVEN("A periodic boundary box") {

        using vector_t = EuclideanVector<3, double>;
        double L = 100;
        PeriodicBoundaryBox pacman_box(L);

        CAPTURE(pacman_box);
        REQUIRE(pacman_box.side_length == 100);


        WHEN("Boundary conditions are applied") {

            THEN("Vectors within it's boundary are left unchanged") { 

                vector_t middle = {L/2, L/2, L/2}; // The middle of the box
                vector_t farmost_corner = {L, L, L}; // A corner of the box
                vector_t nearmost_corner = {0., 0., 0.}; // The opposite corner
    
                vector_t wrapped;
                wrapped = pacman_box.apply_boundary_conditions(middle);
                CHECK(wrapped == middle);
    
                wrapped = pacman_box.apply_boundary_conditions(farmost_corner);
                CHECK(wrapped == farmost_corner);
    
                wrapped = pacman_box.apply_boundary_conditions(nearmost_corner);
                CHECK(wrapped == nearmost_corner);
            }
    
            THEN("Vectors outside it's boundary are wrapped") {
    
                vector_t outside_boundary_1 = {0., 0., (L + L/2)};
                vector_t wrap_expected_1 = {0., 0., L/2};
    
                vector_t outside_boundary_2 = {L/2, (0 - L/2), L/2};
                vector_t wrap_expected_2 = {L/2, L/2, L/2};
    
    
                vector_t wrapped;
                wrapped = pacman_box.apply_boundary_conditions(outside_boundary_1);
                CHECK(wrapped == wrap_expected_1);
    
                wrapped = pacman_box.apply_boundary_conditions(outside_boundary_2);
                CHECK(wrapped == wrap_expected_2);
            }
        }

        WHEN("It is used to measure distance btw vectors") {

            THEN("If objects are close enough within the boundaries,"
                 "the distance is the usual distance.") {

                vector_t center = {L/2, L/2, L/2}; // Center of the box
                vector_t v = {L/4, L/4, L/4}; // Halfway towards the center

                CHECK(pacman_box.distance_vector(center, v) == (center - v));
            }

            THEN("If objects are separated enough within the boundaries,"
                 "the distance takes into account the periodic boundaries.") {
            
                vector_t u = {L/4, L/4, L/4}; // Halfway towards the center
                vector_t v = {L, L, L}; // The farthest corner
                vector_t expected_distance = {L/4, L/4, L/4}; // Wrapped around
                
                CHECK(pacman_box.distance_vector(u, v) == expected_distance);
            }
        }
    }
}

#endif
