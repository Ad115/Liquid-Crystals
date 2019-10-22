#pragma once

#include <fstream>

class XYZ {
    std::unique_ptr<std::ofstream> outputf_ptr;
    std::string filename;

public:
    XYZ(const char *fname) filename(fname);

    XYZ(const XYZ &other) = default;

    void open() {
        bool initialized = static_cast<bool>(outputf_ptr);
        if (!initialized) {
            outputf_ptr -> open(filename);
        }
    }

    template <class ParticleT>
    void operator()(gpu_array<ParticleT> &particles) { /*
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
        auto outputf = *outputf_ptr;

        particles.to_cpu();
        outputf << particles.size << "\n";
        for (auto p: particles) {
            outputf << "\n";
            auto position = p.position;
            for (int D = 0; D < position.dimensions; D++)
                outputf << position[D] << " ";
        }
        outputf << std::endl;

        return;
    }
};