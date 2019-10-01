#ifndef LENNARD_JONES_HEADER
#define LENNARD_JONES_HEADER

template< typename VectorT=Vector<> >
class LennardJones: public Particle<VectorT>{

public:

        template< typename ContainerT >
        __host__ __device__
        VectorT force_law(LennardJones *other, ContainerT *box ){ /*
            * The force law: Lennard Jones
            * ============================
            * 
            *  f_ij = 48*e*[(s / r_ij)^12 - 1/2(s / r_ij)^6] * dr/r^2
            * 
            * See: http://www.pages.drexel.edu/~cfa22/msim/node26.html
            */
            VectorT dr = box->distance_vector((*this).position, other->position);

            double r2 = (dr*dr);
            double r6_i = 1./(r2*r2*r2);
            double r12_i = (r6_i * r6_i);

            return 48*(r12_i - .5*r6_i) * (dr / r2);
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
#include "Container.cu"

SCENARIO("Lennard-Jones particle specification") {

    GIVEN("A Lennard-Jones particle, in a given environment") {
        using vector_t = Vector<3, double>;

        LennardJones<vector_t> p;
        EmptySpace<vector_t> environment;

        WHEN("It interacts with another particle at a distance > R0 := 2^(1/6)") {

            LennardJones<vector_t> other;
            other.position = {1., 1., 1.};

            auto dr = environment.distance_vector(p.position, other.position);
            double R0 = 1.122462048309373;
            CHECK(dr*dr > R0*R0); // Farther than distance R0

            THEN("The force it experiments is attractive") {

                auto force = p.force_law(&other, &environment);

                // Attractive: force and dr point in opposite directions
                CHECK(force * dr < 0); 

                auto r2 = dr * dr;
                auto r6i = 1/(r2 * r2 * r2);
                auto r12i = r6i * r6i;
                auto expected_force = 48*(r12i - .5*r6i)/r2 * dr;

                CHECK(force == expected_force);
                CHECK(expected_force * dr < 0); // Attractive
            }
        }

        WHEN("It interacts with another particle at a distance == R0 := 2^(1/6)") {

            LennardJones<vector_t> other;
            double R0 = 1.122462048309373;
            other.position = {R0, 0., 0.};

            auto dr = environment.distance_vector(p.position, other.position);
            CHECK(dr*dr == R0*R0); // Distance is exactly R0

            THEN("The force it experiments is exactly zero") {

                auto force = p.force_law(&other, &environment);
                CHECK(force*force == 0); // Exactly zero

                auto r2 = dr * dr;
                auto r6i = 1/(r2 * r2 * r2);
                auto r12i = r6i * r6i;
                auto expected_force = 48*(r12i - .5*r6i)/r2 * dr;
                CAPTURE(r2);

                CHECK(force == expected_force);
                CHECK(expected_force * expected_force == 0); // Zero force
            }
        }

        WHEN("It interacts with another particle at a distance < R0 := 2^(1/6)") {

            LennardJones<vector_t> other;
            other.position = {.5, .5, .5};

            auto dr = environment.distance_vector(p.position, other.position);
            double R0 = 1.122462048309373;
            CHECK(dr*dr < R0*R0); // Distance less than R0

            THEN("The force it experiments is repulsive") {

                auto force = p.force_law(&other, &environment);

                // Repulsive force: force goes in the same direction of dr
                CHECK(force * dr > 0); 

                auto r2 = dr * dr;
                auto r6i = 1/(r2 * r2 * r2);
                auto r12i = r6i * r6i;
                auto expected_force = 48*(r12i - .5*r6i)/r2 * dr;

                CHECK(force == expected_force);
                CHECK(expected_force * dr > 0); // Repulsive
            }
        }

        WHEN("The force is probed around a distance of R_min := (14/4)^(1/6)") {
            LennardJones<vector_t> probe;
            double R_min = 1.2321909291736532;
            vector_t v_R_min = {R_min, 0., 0.};
            probe.position = v_R_min;

            auto dr = environment.distance_vector(p.position, probe.position);
            CHECK(dr*dr == R_min*R_min); // Distance is exactly R_min

            THEN("A minimum is encountered") {
                auto f_min = p.force_law(&probe, &environment)*dr / (dr*dr);

                probe.position += {0.1, 0., 0.};
                auto f_perturbed = p.force_law(&probe, &environment)*dr / (dr*dr);
                CAPTURE(probe.position);

                CHECK(f_min < f_perturbed);

                probe.position = v_R_min;
                probe.position += {-0.1, 0., 0.};
                f_perturbed = p.force_law(&probe, &environment)*dr / (dr*dr);
                CAPTURE(probe.position);

                CHECK(f_min < f_perturbed);
            }
        }
    }
}

#endif
#endif