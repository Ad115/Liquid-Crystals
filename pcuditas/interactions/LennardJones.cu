#pragma once

#include "pcuditas/gpu/gpu_object.cu"


template <typename ParticleT>
using Vector_t = typename ParticleT::vector_type;


template<class InteractionT, class EnvironmentT>
struct ConstrainedInteraction {
    EnvironmentT *environment;
    
    using interaction = InteractionT;

    ConstrainedInteraction(EnvironmentT *env)
        : environment(env) {}
    
    template <typename ParticleT>
    __host__ __device__
    Vector_t<ParticleT> operator()(ParticleT &p1, ParticleT &p2) {
        return interaction::interaction_force(p1, p2, *environment);
    }
};

class LennardJones {

public:
    template<class VectorT>
    __host__ __device__
    static VectorT force_law(VectorT distance_vector) { /*
        * The force law: Lennard Jones
        * ============================
        * 
        * Given the distance vector (dr) to the other particle, the
        * force experimented by the current particle is given by:
        *  48[(1 / r)^12 - 1/2(1 / r)^6] * dr/r^2
        * 
        * See: http://www.pages.drexel.edu/~cfa22/msim/node26.html
        */
        auto dr = distance_vector;

        double r2 = (dr*dr);
        double r6_i = 1./(r2*r2*r2);
        double r12_i = (r6_i * r6_i);

        return 48*(r12_i - 0.5*r6_i) * (dr / r2);
    }

    template<class ParticleT, class ContainerT>
    __host__ __device__
    static Vector_t<ParticleT> interaction_force(
            const ParticleT p1, const ParticleT p2,
            const ContainerT& box) { 

        auto dr = box.distance_vector(p1.position, p2.position);
        return force_law(dr);
    }

    template<class ParticleT>
    __host__ __device__
    static Vector_t<ParticleT> interaction_force(
            const ParticleT p1, const ParticleT p2) { 

        auto dr = p1.position - p2.position;
        return force_law(dr);
    }

    template <typename ParticleT>
    __host__ __device__
    Vector_t<ParticleT> operator()(ParticleT &p1, ParticleT &p2) {
        return interaction_force(p1, p2);
    }

    template<class EnvironmentT>
    __host__
    static ConstrainedInteraction<LennardJones, EnvironmentT>
    constrained_by(gpu_object<EnvironmentT> &env){
        return 
            ConstrainedInteraction<
                LennardJones, 
                EnvironmentT>{env.gpu_pointer()};
    }

    template<class EnvironmentT>
    __host__
    static ConstrainedInteraction<LennardJones, EnvironmentT>
    constrained_by(EnvironmentT &env){
        return 
            ConstrainedInteraction<
                LennardJones, 
                EnvironmentT>{&env};
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
#include <typeinfo>   // typeid
#include <math.h> // sqrt, pow
#include "pcuditas/vectors/EuclideanVector.cu"

TEST_SUITE("Lennard-Jones interaction") {

    SCENARIO("Lennard-Jones force with no constraints") {

        using vector_t = EuclideanVector<3,double>;

        auto magnitude = [](vector_t vec) {
            return sqrt(vec * vec);
        };
        auto force = [](vector_t dist_vector) {
            return LennardJones::force_law(dist_vector);
        };
        auto unit = [magnitude](vector_t vec) {
            return vec / magnitude(vec);
        };

        GIVEN("A distance vector, given as the separation between particles A and B") {

            vector_t distances[3] = {
                {145.5, 0.54, -1.24e2},
                {1., 1., 1.},
                {-34., -4.6, -0.0004}
            };

            WHEN("The force between the particles is calculated") {

                THEN("The force has as magnitude:"
                    "48(r_AB^(-12) - 0.5 r_AB^(-6))/r_AB") {

                    auto expected_force_magnitude = [magnitude](vector_t dist_vector) {
                        auto r = magnitude(dist_vector);
                        auto force = 48*(pow(r, -12) - 0.5*pow(r, -6))/r;
                        return abs(force);
                    };

                    vector_t dr;
                    dr = distances[0];
                    CHECK(
                        magnitude(force(dr))
                        == 
                        doctest::Approx( expected_force_magnitude(dr) )
                    );

                    dr = distances[1];
                    CHECK(
                        magnitude(force(dr))
                        == 
                        doctest::Approx( expected_force_magnitude(dr) )
                    );

                    dr = distances[2];
                    CHECK(
                        magnitude(force(dr))
                        == 
                        doctest::Approx( expected_force_magnitude(dr) )
                    );
                }

                AND_THEN("The force is on the direction of the vector"
                    " that connects both particles") {

                    vector_t dr;
                    dr = distances[0];
                    CHECK(
                        abs( unit(force(dr)) * unit(dr) )
                        == 
                        doctest::Approx( 1 )
                    );

                    dr = distances[1];
                    CHECK(
                        abs( unit(force(dr)) * unit(dr) )
                        == 
                        doctest::Approx( 1 )
                    );

                    dr = distances[2];
                    CHECK(
                        abs( unit(force(dr)) * unit(dr) )
                        == 
                        doctest::Approx( 1 )
                    );
                }
            }
        }
        
        GIVEN("Distance vectors of sizes respectively >R0, =R0 and <R0 where R0=2^(1/6)") {

            double R0 = 1.122462048309373;

            vector_t small_dr = {.5, .5, .5}; // Magnitude < R0
            auto critical_dr = R0 * unit(vector_t{13.5, -24.5, .5}); // Mag == R0
            vector_t large_dr = {-2.5, -6., -4.3}; // Magnitude > R0

            WHEN("The distance is less than R0 := 2^(1/6)") {

                THEN("The force is repulsive, i.e. force and distance vector "
                     "are in the same direction") {
                        
                        auto dr = small_dr;
                        CHECK( force(dr)*dr > 0 );
                    }
                }

            AND_WHEN("The distance equals R0 := 2^(1/6)") {

                THEN("The force is zero") {

                    CHECK(
                        magnitude(force(critical_dr))
                        == doctest::Approx(0)
                    );
                }
            }

            AND_WHEN("The distance is larger than R0 := 2^(1/6)") {

                THEN("The force is attractive, i.e. force and distance vector "
                     "are in opposite directions") {

                    auto dr = large_dr;
                    CHECK( force(dr)*dr < 0. );
                }
            }
        }
    }
}

#endif