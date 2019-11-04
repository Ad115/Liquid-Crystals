#ifndef LENNARD_JONES_HEADER
#define LENNARD_JONES_HEADER

#include "SimpleParticle.cu"
#include "pcuditas/interactions/LennardJones.cu"

template <class ParticleT, class InteractionT>
class InteractingParticle : public ParticleT {

public:

    using interaction = InteractionT;
    using vector_type = typename ParticleT::vector_type;

    template<class VectorT>
    __host__ __device__
    static VectorT force_law(VectorT distance_vector) {
        return interaction::force_law(distance_vector);
    }

    template< typename ContainerT >
    __host__ __device__
    vector_type interaction_force_with(
            const InteractingParticle<ParticleT, InteractionT>& other, 
            const ContainerT& box) { 

        return interaction::interaction_force(*this, other, box);
    }
};

using LennardJonesParticle = InteractingParticle<SimpleParticle, LennardJones>;

/* -----------------------------------------------------------------------

 The following is executable documentation as described in Kevlin Henney's talk 
    "Structure and Interpretation of Test Cases" (https://youtu.be/tWn8RA_DEic)
    written using the doctest framework (https://github.com/onqtam/doctest). 

 Run with `make test`.
*/

#ifdef __TESTING__

#include "doctest.h"
#include <typeinfo>   // typeid
#include <math.h> // sqrt, pow
#include "pcuditas/environments/EmptySpace.cu"
#include "pcuditas/vectors/EuclideanVector.cu"

SCENARIO("Lennard-Jones particle specification") {

    using vector_t = EuclideanVector<3,double>;

    auto magnitude = [](vector_t vec) {
        return sqrt(vec * vec);
    };
    auto unit = [magnitude](vector_t vec) {
        return vec / magnitude(vec);
    };

    
    SUBCASE("Integration test: using with a container") {
        GIVEN("A Lennard-Jones particle at the origin, in a given environment") {

            LennardJonesParticle p;
            EmptySpace environment;
    
            double R0 = 1.122462048309373;
            vector_t small_dr = {.5, .5, .5}; // Magnitude < R0
            auto critical_dr = R0 * unit(vector_t{13.5, -24.5, .5}); // Mag == R0
            vector_t large_dr = {-2.5, -6., -4.3}; // Magnitude > R0
    
            WHEN("It interacts with another particle at a distance < R0 := 2^(1/6)") {
    
                LennardJonesParticle other;
                other.position = small_dr;
    
                auto dr = environment.distance_vector(p.position, other.position);
                CHECK(magnitude(dr) < R0); // Distance less than R0
    
                THEN("The force it experiments is repulsive") {
    
                    auto force = p.interaction_force_with(other, environment);
    
                    // Repulsive force: force goes in the same direction of dr
                    CHECK(force * dr > 0); 
                }
            }
    
            WHEN("It interacts with another particle at a distance == R0 := 2^(1/6)") {
    
                LennardJonesParticle other;
                other.position = critical_dr;
    
                auto dr = environment.distance_vector(p.position, other.position);
                CHECK(magnitude(dr) == doctest::Approx(R0)); // Distance is exactly R0
    
                THEN("The force it experiments is exactly zero") {
    
                    auto force = p.interaction_force_with(other, environment);
                    CHECK(magnitude(force) == doctest::Approx(0)); // Exactly zero
                }
            }
    
            WHEN("It interacts with another particle at a distance > R0 := 2^(1/6)") {
    
                LennardJonesParticle other;
                other.position = large_dr;
    
                auto dr = environment.distance_vector(p.position, other.position);
                CHECK(magnitude(dr) > R0); // Farther than distance R0
    
                THEN("The force it experiments is attractive") {
    
                    auto force = p.interaction_force_with(other, environment);
    
                    // Attractive: force and dr point in opposite directions
                    CHECK(force * dr < 0); 
                }
            }
        }
    }
    
}

#endif
#endif