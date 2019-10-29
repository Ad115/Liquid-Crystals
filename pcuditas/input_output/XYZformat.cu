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