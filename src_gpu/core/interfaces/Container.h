#ifndef CONTAINER_INTERFACE_HEADER
#define CONTAINER_INTERFACE_HEADER

template < typename VectorT >
class Container {
    private:

    public:
        using vector_type = VectorT;
        static constexpr int dimensions = VectorT::dimensions;

        VectorT apply_boundary_conditions(const VectorT& position) const;

        VectorT distance_vector(const VectorT& p1, const VectorT& p2) const;
};

template< typename VectorT >
constexpr int Container<VectorT>::dimensions;



/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "doctest.h"
#include "src_gpu/core/Vector.cu"
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

#endif
#endif