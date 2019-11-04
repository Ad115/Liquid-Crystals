#pragma once

#include "SimpleParticle.cu"

class LennardJonesInteraction {

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
        auto& dr = distance_vector;

        double r2 = (dr*dr);
        double r6_i = 1./(r2*r2*r2);
        double r12_i = (r6_i * r6_i);

        return 48*(r12_i - 0.5*r6_i) * (dr / r2);
    }

    template<class ParticleT>
    using Vector = typename ParticleT::vector_type;

    template<class ParticleT, class ContainerT>
    __host__ __device__
    static Vector<ParticleT> interaction_force(
            const ParticleT p1, const ParticleT p2,
            const ContainerT& box) { 

        Vector<ParticleT> dr = box.distance_vector(p1.position, p2.position);
        return force_law(dr);
    }
};


class LennardJonesParticle : public SimpleParticle {

public:

    using interaction = LennardJonesInteraction;
    using vector_type = typename SimpleParticle::vector_type;

    template<class VectorT>
    __host__ __device__
    static VectorT force_law(VectorT distance_vector) {
        return interaction::force_law(distance_vector);
    }

    template< typename ContainerT >
    __host__ __device__
    vector_type interaction_force_with(
            const LennardJonesParticle& other, 
            const ContainerT& box) { 

        return interaction::interaction_force(*this, other, box);
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
#include <typeinfo>   // typeid
#include <math.h> // sqrt, pow
#include "pcuditas/environments/EmptySpace.cu"
#include "pcuditas/vectors/EuclideanVector.cu"

TEST_SUITE("Lennard-Jones interaction and particles") {

    SCENARIO("Lennard-Jones interaction") {

        using vector_t = EuclideanVector<3,double>;

        auto magnitude = [](vector_t vec) {
            return sqrt(vec * vec);
        };
        auto force = [](vector_t dist_vector) {
            return LennardJonesInteraction::force_law(dist_vector);
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
}

#endif