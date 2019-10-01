/* 
## Clase `Container`

El espacio donde se mueven las partículas. Contiene la información sobre las 
condiciones de frontera y la distancia entre partículas. Está diseñada para 
vivir en el Device. 
*/

#ifndef CONTAINER_HEADER
#define CONTAINER_HEADER

#include "Vector.cu"


template < typename VectorT=Vector<> >
class Container {
    private:

    public:
        using vector_type = VectorT;
        static constexpr int dimensions = VectorT::dimensions;

        __host__ __device__ 
        VectorT apply_boundary_conditions(const VectorT& position) const;

        __host__ __device__
        VectorT distance_vector(const VectorT& p1, const VectorT& p2) const;
};

template< typename VectorT >
constexpr int Container<VectorT>::dimensions;

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

template< typename VectorT=Vector<> >
class PeriodicBoundaryBox : public Container<VectorT> {

    public:

    double side_length;

    __host__ __device__ PeriodicBoundaryBox( double L )
            : side_length(L) {}

    __host__ __device__ 
    VectorT apply_boundary_conditions(const VectorT& position) const {
        VectorT new_pos(position);
        for (int i=0; i<new_pos.dimensions;i++){
            new_pos[i] -= (new_pos[i] > side_length) * side_length;
            new_pos[i] += (new_pos[i] < 0) * side_length;
        }
        return new_pos;
    }

    __host__ __device__ 
    VectorT box_size() { 
        VectorT side_lengths;
        for (int i=0; i<side_lengths.dimensions; i++)
            side_lengths[i] = side_length;

        return side_lengths;
    }

    __host__ __device__ 
    VectorT distance_vector(const VectorT& r1, const VectorT& r2) const { /*
        * Get the distance to the minimum image.
        */
        double half_length = side_length/2;
        VectorT dr = r2 - r1;

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
        const PeriodicBoundaryBox<VectorT>& box) {

        stream << "PeriodicBoundary{";
        for (int i=0; i < box.dimensions-1; i++)
            stream << box.side_length << ", ";
                    
        stream << box.side_length << '}';
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

#include "doctest.h"
#include <typeinfo>   // operator typeid

SCENARIO("Container specification") {

    GIVEN("A container") {
        using vector_t = Vector<3, double>;
        Container<vector_t> box;

        THEN("It exists in a system of coordinates") {
            REQUIRE(box.dimensions == 3);

            using box_vector_t = typename decltype(box)::vector_type;
            REQUIRE(typeid(box_vector_t) == typeid(vector_t));
        }
    }
    
    SUBCASE("The raw 'Container' class is only an interface, "
            "it doesn't contain code"){
        // --- The following doesn't compile as there is no implementation.
        // box.apply_boundary_conditions(Vector<>{});
        // box.distance_vector(Vector<>(), Vector<>());
    }
}



SCENARIO("Empty space specification") {

    GIVEN("Empty, infinite space as a container") {

        using vector_t = Vector<3, double>;
        EmptySpace<vector_t> empty_box;

        REQUIRE(empty_box.dimensions == 3);

        using empty_box_vector_t = typename decltype(empty_box)::vector_type;
        REQUIRE(typeid(empty_box_vector_t) == typeid(vector_t));


        WHEN("Boundary conditions are applied") {

            THEN("All vectors are left unchanged") { 

                Vector<> random1 = {102., 0.54, -23.4};
                Vector<> random2 = {0., 0., 0.};
                Vector<> random3 = {5e-12, 0.54, -.4};
    
                Vector<> wrapped;
                wrapped = empty_box.apply_boundary_conditions(random1);
                CHECK(wrapped == random1);
    
                wrapped = empty_box.apply_boundary_conditions(random2);
                CHECK(wrapped == random2);
    
                wrapped = empty_box.apply_boundary_conditions(random3);
                CHECK(wrapped == random3);
            }
        }

        WHEN("It is used to measure distance btw vectors") {

            THEN("The distance vector is vector difference") {

                Vector<> u = {0., 0., 0.};
                Vector<> v = {102., 0.54, -23.4};

                CHECK(empty_box.distance_vector(u, v) == (v - u));
                CHECK(empty_box.distance_vector(u, v) == v);
                CHECK(empty_box.distance_vector(v, u) == (u - v));
                CHECK(empty_box.distance_vector(v, u) == -v);
            }
        }
    }
}



SCENARIO("Periodic boundary box specification") {

    GIVEN("A periodic boundary box") {

        using vector_t = Vector<>;
        double L = 100;
        PeriodicBoundaryBox<vector_t> pacman_box(L);

        CAPTURE(pacman_box);
        REQUIRE(pacman_box.side_length == 100);
        REQUIRE(pacman_box.dimensions == 3);

        using pacman_box_vector_t = typename decltype(pacman_box)::vector_type;
        REQUIRE(typeid(pacman_box_vector_t) == typeid(vector_t));


        WHEN("Boundary conditions are applied") {

            THEN("Vectors within it's boundary are left unchanged") { 

                Vector<> middle = {L/2, L/2, L/2}; // The middle of the box
                Vector<> farmost_corner = {L, L, L}; // A corner of the box
                Vector<> nearmost_corner = {0., 0., 0.}; // The opposite corner
    
                Vector<> wrapped;
                wrapped = pacman_box.apply_boundary_conditions(middle);
                CHECK(wrapped == middle);
    
                wrapped = pacman_box.apply_boundary_conditions(farmost_corner);
                CHECK(wrapped == farmost_corner);
    
                wrapped = pacman_box.apply_boundary_conditions(nearmost_corner);
                CHECK(wrapped == nearmost_corner);
            }
    
            THEN("Vectors outside it's boundary are wrapped") {
    
                Vector<> outside_boundary_1 = {0., 0., (L + L/2)};
                Vector<> wrap_expected_1 = {0., 0., L/2};
    
                Vector<> outside_boundary_2 = {L/2, (0 - L/2), L/2};
                Vector<> wrap_expected_2 = {L/2, L/2, L/2};
    
    
                Vector<> wrapped;
                wrapped = pacman_box.apply_boundary_conditions(outside_boundary_1);
                CHECK(wrapped == wrap_expected_1);
    
                wrapped = pacman_box.apply_boundary_conditions(outside_boundary_2);
                CHECK(wrapped == wrap_expected_2);
            }
        }

        WHEN("It is used to measure distance btw vectors") {

            THEN("If objects are close enough within the boundaries,"
                 "the distance is the usual distance.") {

                Vector<> center = {L/2, L/2, L/2}; // Center of the box
                Vector<> v = {L/4, L/4, L/4}; // Halfway towards the center

                CHECK(pacman_box.distance_vector(center, v) == (v - center));
            }

            THEN("If objects are separated enough within the boundaries,"
                 "the distance takes into account the periodic boundaries.") {
            
                Vector<> u = {L/4, L/4, L/4}; // Halfway towards the center
                Vector<> v = {L, L, L}; // The farthest corner
                Vector<> expected_distance = {L/4, L/4, L/4}; // Wrapped around
                
                CHECK(pacman_box.distance_vector(v, u) == expected_distance);
            }
        }
    }
}

#endif
#endif