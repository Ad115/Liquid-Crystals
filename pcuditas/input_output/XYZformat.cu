#pragma once

#include <fstream>

class XYZ {
    std::ofstream outputf;

public:

    template <class ParticleT>
    static void write(std::ostream& stream, gpu_array<ParticleT> &particles) {/*
        * Output the positions of the particles in the XYZ format.
        * The format consists in a line with the number of particles,
        * then a comment line followed by the space-separated coordinates 
        * of each particle in different lines.
        * 
        * Example (for 3 particles in the xyz diagonal):
        * 
        *   10
        *   
        *   1.0 1.0 1.0
        *   1.5 1.5 1.5
        *   2.0 2.0 2.0
        */

        particles.to_cpu();
        stream << particles.size << "\n";
        for (auto p: particles) {
            stream << "\n";
            auto position = p.position;
            for (int D = 0; D < position.dimensions; D++)
                stream << position[D] << " ";
        }
        stream << std::endl;

        return;
    }

    XYZ(const char *filename) : outputf(filename) {}

    template <class ParticleT>
    void operator()(gpu_array<ParticleT> &particles) {
        write(outputf, particles);
        return;
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
#include <sstream> // stringstream
#include <string> // string
#include "pcuditas/particles/SimpleParticle.cu"
#include "pcuditas/vectors/EuclideanVector.cu"

TEST_SUITE("XYZ format output specification") {

    SCENARIO("Description") {
        GIVEN("A GPU array of 3D-particles at arbitrary positions") {

            using vector_t = EuclideanVector<3>;
            using particle_t = Particle<vector_t>;

            auto particles 
                = gpu_array<particle_t>( 3,
                    [] __device__ (particle_t &p, int i) {
                        p.position = vector_t{
                            (float)i, 
                            (float)(i*i), 
                            (float)(i*i + i)};
                    });

            WHEN("The system is written to the XYZ format") {
                std::stringstream stream;

                XYZ::write(stream, particles);

                THEN("The output has the correct format") {
                    std::string expected = (
                        "3\n\n"
                        "0 0 0 \n"
                        "1 1 2 \n"
                        "2 4 6 \n"
                    );
                    CHECK(stream.str() == expected);
                }
            }
        }
    }
}

#endif